"""Inference helpers for single-video predictions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .data import load_video
from .preprocessing import normalize_frames, resize_frames, temporal_sample_clips
from .utils import save_json

DEFAULT_FALLBACK_FPS = 30.0


@torch.no_grad()
def predict_video(
    model: torch.nn.Module,
    video_path: Union[str, Path],
    device: torch.device,
    num_frames: int = 16,
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    return_xai: bool = False,
    xai_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Predict whether a single video is real or AI-generated."""
    frames = torch.from_numpy(load_video(video_path))
    duration_seconds = _estimate_duration_seconds(video_path, total_frames=frames.shape[0])
    num_clips = _num_clips_for_duration(duration_seconds)
    clips = temporal_sample_clips(frames, num_frames, num_clips)
    model.eval()
    if return_xai and hasattr(model, "predict_with_xai"):
        clip_logits = []
        clip_outputs = []
        for clip in clips:
            normalized = _prepare_clip(clip, image_size, mean, std, device)
            outputs = model.predict_with_xai(normalized, threshold=xai_threshold)
            clip_outputs.append(outputs)
            clip_logits.append(outputs["logits"].reshape(-1)[0])
        logits = torch.stack(clip_logits)
        confidence = _aggregate_confidence(logits)
        selected_output = clip_outputs[_select_representative_clip_index(logits)]
    else:
        normalized = torch.stack([normalize_frames(resize_frames(clip, image_size), mean, std) for clip in clips], dim=0).to(device)
        logits = model(normalized).reshape(-1)
        confidence = _aggregate_confidence(logits)
        selected_output = None
    prediction = "ai_generated" if confidence >= 0.5 else "real"
    payload: Dict[str, Any] = {"prediction": prediction, "confidence": confidence}
    if selected_output is not None:
        payload["xai"] = _format_xai_output(selected_output, threshold=xai_threshold)
    return payload


def _prepare_clip(
    clip: torch.Tensor,
    image_size: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    device: torch.device,
) -> torch.Tensor:
    resized = resize_frames(clip, image_size)
    return normalize_frames(resized, mean, std).unsqueeze(0).to(device)


def _estimate_duration_seconds(video_path: Union[str, Path], total_frames: int) -> float:
    metadata_duration = _read_duration_from_metadata(video_path)
    if metadata_duration is not None and metadata_duration > 0:
        return metadata_duration
    return total_frames / DEFAULT_FALLBACK_FPS


def _read_duration_from_metadata(video_path: Union[str, Path]) -> Optional[float]:
    source = Path(video_path)
    suffix = source.suffix.lower()
    if suffix == ".gif":
        return _read_gif_duration(source)
    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        return _read_video_duration(source)
    return None


def _read_gif_duration(path: Path) -> Optional[float]:
    try:
        from PIL import Image, ImageSequence

        with Image.open(path) as image:
            milliseconds = sum(float(frame.info.get("duration", 0.0)) for frame in ImageSequence.Iterator(image))
        return milliseconds / 1000.0 if milliseconds > 0 else None
    except Exception:
        return None


def _read_video_duration(path: Path) -> Optional[float]:
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return None
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        finally:
            capture.release()
        if fps <= 0 or frame_count <= 0:
            return None
        return frame_count / fps
    except Exception:
        return None


def _num_clips_for_duration(duration_seconds: float) -> int:
    if duration_seconds <= 5:
        return 1
    if duration_seconds <= 15:
        return 3
    if duration_seconds <= 30:
        return 5
    return max(8, math.ceil(duration_seconds / 5))


def _aggregate_confidence(logits: torch.Tensor) -> float:
    scores = torch.sigmoid(logits.reshape(-1))
    top_k = max(1, math.ceil(scores.numel() * 0.3))
    return float(torch.topk(scores, k=top_k).values.mean().item())


def _select_representative_clip_index(logits: torch.Tensor) -> int:
    return int(torch.sigmoid(logits.reshape(-1)).argmax().item())


def _format_xai_output(outputs: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    frame_importance = outputs.get("frame_importance")
    frame_scores = _tensor_to_float_list(frame_importance)
    segments = outputs.get("segments", [])
    explanations = outputs.get("explanations", [])
    return {
        "method": str(outputs.get("xai_method", "")),
        "threshold": float(threshold),
        "frame_importance": frame_scores,
        "segments": segments,
        "explanations": explanations,
        "summary": {
            "num_frames": len(frame_scores),
            "num_segments": len(segments),
            "max_frame_importance": max(frame_scores) if frame_scores else None,
        },
    }


def _tensor_to_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        flattened = value.squeeze(0).detach().cpu().tolist()
        return [float(score) for score in flattened]
    if isinstance(value, list):
        return [float(score) for score in value]
    return []


def save_prediction(path: Union[str, Path], payload: Dict[str, Any]) -> Path:
    """Persist inference results to JSON."""
    return save_json(path, payload)
