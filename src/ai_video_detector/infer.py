"""Inference helpers for single-video predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
    xai_output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Predict whether a single video is real or AI-generated."""
    frames = torch.from_numpy(load_video(video_path))
    sampled = temporal_sample(frames, num_frames)
    resized = resize_frames(sampled, image_size)
    visual_frames = resized.clone()
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
        payload["xai"] = _format_xai_output(
            outputs,
            threshold=xai_threshold,
            visualizations=_save_anomaly_heatmaps(
                video_path=video_path,
                visual_frames=visual_frames,
                outputs=outputs,
                output_root=Path(xai_output_dir),
            )
            if xai_output_dir is not None
            else [],
        )
    return payload


def _format_xai_output(outputs: Dict[str, Any], threshold: float, visualizations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
        "visualizations": visualizations or [],
        "summary": {
            "num_frames": len(frame_scores),
            "num_segments": len(segments),
            "max_frame_importance": max(frame_scores) if frame_scores else None,
            "num_visualizations": len(visualizations or []),
        },
    }


def _save_anomaly_heatmaps(
    video_path: Union[str, Path],
    visual_frames: torch.Tensor,
    outputs: Dict[str, Any],
    output_root: Path,
) -> List[Dict[str, Any]]:
    attention_map = outputs.get("attention_map")
    segments = outputs.get("segments", [])
    frame_importance = _tensor_to_float_list(outputs.get("frame_importance"))
    if not isinstance(attention_map, torch.Tensor) or attention_map.ndim != 4:
        return []

    heatmap_dir = output_root / "xai" / "heatmaps"
    overlay_dir = output_root / "xai" / "overlays"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, Any]] = []
    seen_frames: set[int] = set()
    num_frames = visual_frames.shape[0]
    num_temporal_tokens = attention_map.shape[1]
    video_stem = Path(video_path).stem
    for segment in segments:
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        anomaly_type = str(segment["type"])
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in seen_frames or frame_idx < 0 or frame_idx >= num_frames:
                continue
            seen_frames.add(frame_idx)
            token_idx = _frame_idx_to_token_idx(frame_idx, num_frames, num_temporal_tokens)
            frame_np = _frame_to_uint8_hwc(visual_frames[frame_idx])
            heat_rgb, overlay = _make_heatmap_and_overlay(frame_np, attention_map[0, token_idx])
            heatmap_path = heatmap_dir / f"{video_stem}_frame{frame_idx}.png"
            overlay_path = overlay_dir / f"{video_stem}_frame{frame_idx}.png"
            Image.fromarray(heat_rgb).save(heatmap_path)
            Image.fromarray(overlay).save(overlay_path)
            saved.append(
                {
                    "frame_idx": frame_idx,
                    "importance": frame_importance[frame_idx] if frame_idx < len(frame_importance) else None,
                    "type": anomaly_type,
                    "heatmap_path": str(heatmap_path),
                    "overlay_path": str(overlay_path),
                }
            )
    return saved


def _frame_idx_to_token_idx(frame_idx: int, num_frames: int, num_temporal_tokens: int) -> int:
    if num_frames <= 1:
        return 0
    return min(
        num_temporal_tokens - 1,
        int(round(frame_idx * (num_temporal_tokens - 1) / (num_frames - 1))),
    )


def _frame_to_uint8_hwc(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)
    array = frame.detach().cpu().numpy()
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _make_heatmap_and_overlay(frame_np: np.ndarray, token_map: torch.Tensor, alpha: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
    height, width = frame_np.shape[:2]
    heat = F.interpolate(
        token_map.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    heat = heat - heat.min()
    heat = heat / heat.max().clamp_min(1e-6)
    heat_np = heat.detach().cpu().numpy()
    heat_rgb = np.stack(
        [
            (heat_np * 255).astype(np.uint8),
            np.zeros_like(heat_np, dtype=np.uint8),
            ((1.0 - heat_np) * 255).astype(np.uint8),
        ],
        axis=-1,
    )
    overlay = ((1.0 - alpha) * frame_np.astype(np.float32) + alpha * heat_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return heat_rgb, overlay


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
