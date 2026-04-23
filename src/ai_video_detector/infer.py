"""Inference helpers for single-video predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

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
) -> Dict[str, Union[float, str, List[float]]]:
    """Predict whether a single video is real or AI-generated."""
    frames = torch.from_numpy(load_video(video_path))
    sampled = temporal_sample(frames, num_frames)
    resized = resize_frames(sampled, image_size)
    normalized = normalize_frames(resized, mean, std).unsqueeze(0).to(device)
    model.eval()

    #이 부분 수정됨
    if return_xai and hasattr(model, "predict_with_xai"):
        outputs = model.predict_with_xai(normalized)
        logits = outputs["logits"]
    else:
        outputs = None
        logits = model(normalized)

    confidence = float(torch.sigmoid(logits).item())
    prediction = "ai_generated" if confidence >= 0.5 else "real"

    payload = {
        "prediction": prediction,
        "confidence": confidence
    }

    if outputs is not None:
        payload["frame_importance"] = outputs.get("frame_importance", None)
        payload["segments"] = outputs.get("segments", [])
        payload["explanations"] = outputs.get("explanations", [])
        payload["xai_method"] = outputs.get("xai_method", "")

    return payload


def save_prediction(path: Union[str, Path], payload: Dict[str, Union[float, str, List[float]]]) -> Path:
    """Persist inference results to JSON."""
    return save_json(path, payload)
