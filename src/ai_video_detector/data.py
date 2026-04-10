"""Dataset and video loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import normalize_frames, resize_frames, temporal_sample

VideoLoader = Callable[[str | Path], np.ndarray]


@dataclass(frozen=True)
class VideoSample:
    """A single training sample."""

    path: str
    label: int


def load_video(path: str | Path) -> np.ndarray:
    """Load a video from .npy, .pt, or common video formats."""
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

    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        try:
            import cv2  # type: ignore
        except ImportError:
            try:
                import imageio.v3 as iio  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Video loading requires either opencv-python or imageio[v3]."
                ) from exc
            frames = iio.imread(source)
            return _validate_frames(frames)

        capture = cv2.VideoCapture(str(source))
        frames = []
        success, frame = capture.read()
        while success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            success, frame = capture.read()
        capture.release()
        return _validate_frames(np.asarray(frames))

    raise ValueError(f"Unsupported video format: {suffix}")


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
        sample = self.samples[index]
        frames = self.video_loader(sample.path)
        frame_tensor = torch.from_numpy(frames)
        sampled = temporal_sample(frame_tensor, self.num_frames)
        resized = resize_frames(sampled, self.image_size)
        normalized = normalize_frames(resized, self.mean, self.std)
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return normalized, label
