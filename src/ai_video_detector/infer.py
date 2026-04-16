"""Inference helpers for single-video predictions."""

from __future__ import annotations

from pathlib import Path

import torch

from .data import load_video
from .preprocessing import normalize_frames, resize_frames, temporal_sample
from .utils import save_json


@torch.no_grad()
def predict_video(
    model: torch.nn.Module,
    video_path: str | Path,
    device: torch.device,
    num_frames: int = 16,
    image_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    return_xai: bool = False,
) -> dict[str, float | str]:
    """Predict whether a single video is real or AI-generated."""
    frames = torch.from_numpy(load_video(video_path))
    sampled = temporal_sample(frames, num_frames)
    resized = resize_frames(sampled, image_size)
    normalized = normalize_frames(resized, mean, std).unsqueeze(0).to(device)
    model.eval()
    if return_xai and hasattr(model, "predict_with_xai"):
        outputs = model.predict_with_xai(normalized)
        logits = outputs["logits"]
    else:
        outputs = None
        logits = model(normalized)
    confidence = float(torch.sigmoid(logits).item())
    prediction = "ai_generated" if confidence >= 0.5 else "real"
    payload: dict[str, float | str | list[float]] = {"prediction": prediction, "confidence": confidence}
    if outputs is not None:
        frame_importance = outputs["frame_importance"]
        if isinstance(frame_importance, torch.Tensor):
            payload["frame_importance"] = [float(value) for value in frame_importance.squeeze(0).detach().cpu().tolist()]
        payload["xai_method"] = str(outputs["xai_method"])
    return payload


def save_prediction(path: str | Path, payload: dict[str, float | str]) -> Path:
    """Persist inference results to JSON."""
    return save_json(path, payload)
