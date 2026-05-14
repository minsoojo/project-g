"""Inference helpers for single-video predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

from .data import load_video
from .preprocessing import normalize_frames, resize_frames, temporal_sample
from .utils import save_json


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
    sampled = temporal_sample(frames, num_frames)
    resized = resize_frames(sampled, image_size)
    normalized = normalize_frames(resized, mean, std).unsqueeze(0).to(device)
    model.eval()
    if return_xai and hasattr(model, "predict_with_xai"):
        outputs = model.predict_with_xai(normalized, threshold=xai_threshold)
        logits = outputs["logits"]
    else:
        outputs = None
        logits = model(normalized)
    confidence = float(torch.sigmoid(logits).item())
    prediction = "ai_generated" if confidence >= 0.5 else "real"
    payload: Dict[str, Any] = {"prediction": prediction, "confidence": confidence}
    if outputs is not None:
        payload["xai"] = _format_xai_output(outputs, threshold=xai_threshold)
    return payload


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
