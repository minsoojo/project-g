"""Dataset and video loading utilities."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageSequence
from torch.utils.data import Dataset

from .preprocessing import normalize_frames, resize_frames, temporal_sample

VideoLoader = Callable[[Union[str, Path]], np.ndarray]


@dataclass(frozen=True)
class VideoSample:
    """A single training sample."""

    path: str
    label: int


def load_video(path: Union[str, Path]) -> np.ndarray:
    """Load a video from .npy, .pt, .gif, or common video formats."""
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".npy":
        frames = np.load(source)
        return _validate_frames(frames)
    if suffix == ".pt":
        tensor = torch.load(source, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected a tensor inside the .pt file")
        return _validate_frames(tensor.numpy())

    if suffix == ".gif":
        return _load_gif(source)

    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Failed to load video '{source}': could not open file")
        frames = []
        try:
            success, frame = capture.read()
            while success:
                if frame is None:
                    raise ValueError(f"Failed to load video '{source}': decoder returned an empty frame")
                try:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                except Exception as exc:
                    raise ValueError(f"Failed to load video '{source}': {exc}") from exc
                success, frame = capture.read()
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"Failed to load video '{source}': decoded 0 frames")
        try:
            return _validate_frames(np.asarray(frames))
        except Exception as exc:
            raise ValueError(f"Failed to load video '{source}': {exc}") from exc

    raise ValueError(f"Unsupported video format: {suffix}")


def _load_gif(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            frames = np.stack(
                [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(image)],
                axis=0,
            )
        return _validate_frames(frames)
    except Exception as exc:
        raise OSError(f"Failed to load GIF '{path}': {exc}") from exc


def load_video_samples_from_manifest(
    manifest_path: Union[str, Path],
    *,
    base_dir: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
) -> list[VideoSample]:
    """Build ``VideoSample`` items from a CSV manifest.

    Expected columns include ``relative_path`` and ``label``. When ``base_dir`` is
    provided, relative paths are resolved against it; otherwise they are resolved
    against the manifest file's parent directory.
    """
    manifest = Path(manifest_path)
    root = Path(base_dir) if base_dir is not None else manifest.parent
    samples: list[VideoSample] = []

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            if split is not None and row.get("split") != split:
                continue
            if row.get("status") not in {None, "", "ok"}:
                continue
            if row.get("is_zero_byte") == "1":
                continue

            relative_path = row.get("relative_path")
            label = row.get("label")
            if not relative_path or label is None:
                raise ValueError("Manifest rows must include 'relative_path' and 'label'")

            sample_path = Path(relative_path)
            if not sample_path.is_absolute():
                sample_path = root / sample_path
            if not sample_path.exists():
                print(
                    f"[WARN] missing sample path={sample_path}",
                    file=sys.stderr,
                )
                continue
            samples.append(VideoSample(path=str(sample_path), label=int(label)))

    return samples


def _validate_frames(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected frames shaped as [T, H, W, 3]")
    if frames.shape[0] == 0:
        raise ValueError("Video contains no frames")
    return frames


class VideoDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset that loads and preprocesses videos into model-ready tensors."""

    def __init__(
        self,
        samples: Iterable[VideoSample],
        num_frames: int = 16,
        image_size: tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        video_loader: VideoLoader = load_video,
    ) -> None:
        self.samples = list(samples)
        self.num_frames = num_frames
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.video_loader = video_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.samples:
            raise IndexError("VideoDataset is empty")

        total_samples = len(self.samples)
        start_index = index % total_samples

        for offset in range(total_samples):
            sample = self.samples[(start_index + offset) % total_samples]
            try:
                frames = self.video_loader(sample.path)
                frame_tensor = torch.from_numpy(frames)
                sampled = temporal_sample(frame_tensor, self.num_frames)
                resized = resize_frames(sampled, self.image_size)
                normalized = normalize_frames(resized, self.mean, self.std)
                label = torch.tensor(float(sample.label), dtype=torch.float32)
                return normalized, label
            except Exception as exc:
                _warn_failed_sample(sample.path, exc)

        raise RuntimeError(
            f"Failed to load any sample after {total_samples} attempts starting from index {index}."
        )


def _warn_failed_sample(path: Union[str, Path], exc: Exception) -> None:
    print(
        f"[WARN] failed to load sample path={path} error={type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
