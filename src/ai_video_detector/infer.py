"""Inference helpers for single-video predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps

from .data import load_video
from .preprocessing import normalize_frames, resize_frames, temporal_sample
from .utils import save_json


def _frame_to_uint8_hwc(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)

    arr = frame.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _frame_idx_to_token_idx(frame_idx: int, num_frames: int, num_temporal_tokens: int) -> int:
    if num_frames <= 1:
        return 0
    return min(
        num_temporal_tokens - 1,
        int(round(frame_idx * (num_temporal_tokens - 1) / (num_frames - 1))),
    )


def _make_heatmap_and_overlay(
    frame_np: np.ndarray,
    token_map: torch.Tensor,
    alpha: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame_np.shape[:2]

    heat = F.interpolate(
        token_map.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    heat = heat - heat.min()
    heat = heat / heat.max().clamp_min(1e-6)
    heat_np = heat.detach().cpu().numpy()

    heat_rgb = (colormaps["jet"](heat_np)[..., :3] * 255).astype(np.uint8)
    overlay = (
        (1.0 - alpha) * frame_np.astype(np.float32) +
        alpha * heat_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return heat_rgb, overlay


def _save_anomaly_heatmaps(
    video_path: Union[str, Path],
    visual_frames: torch.Tensor,
    outputs: Dict[str, Any],
    output_root: Path,
) -> List[Dict[str, Any]]:
    attention_map = outputs.get("attention_map", None)
    segments = outputs.get("segments", [])
    frame_importance_tensor = outputs.get("frame_importance", None)

    if not isinstance(attention_map, torch.Tensor):
        return []

    # 기대 형태: [B, T_token, H_patch, W_patch]
    if attention_map.ndim != 4:
        return []

    if isinstance(frame_importance_tensor, torch.Tensor):
        frame_importance = [
            float(v) for v in frame_importance_tensor.squeeze(0).detach().cpu().tolist()
        ]
    else:
        frame_importance = []

    heatmap_dir = output_root / "xai" / "heatmaps"
    overlay_dir = output_root / "xai" / "overlay"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    anomaly_frames: list[tuple[int, str]] = []
    for seg in segments:
        start_frame = int(seg["start_frame"])
        end_frame = int(seg["end_frame"])
        seg_type = str(seg["type"])
        for idx in range(start_frame, end_frame + 1):
            anomaly_frames.append((idx, seg_type))

    # frame 중복 제거
    seen = set()
    unique_anomaly_frames = []
    for item in anomaly_frames:
        if item[0] not in seen:
            unique_anomaly_frames.append(item)
            seen.add(item[0])

    num_frames = visual_frames.shape[0]
    num_temporal_tokens = attention_map.shape[1]
    video_stem = Path(video_path).stem

    saved = []

    for frame_idx, seg_type in unique_anomaly_frames:
        token_idx = _frame_idx_to_token_idx(frame_idx, num_frames, num_temporal_tokens)
        token_map = attention_map[0, token_idx]  # [H_patch, W_patch]
        frame_np = _frame_to_uint8_hwc(visual_frames[frame_idx])

        heat_rgb, overlay = _make_heatmap_and_overlay(frame_np, token_map)

        heatmap_name = f"{video_stem}_frame{frame_idx}.png"
        overlay_name = f"{video_stem}_frame{frame_idx}.png"

        heatmap_path = heatmap_dir / heatmap_name
        overlay_path = overlay_dir / overlay_name

        Image.fromarray(heat_rgb).save(heatmap_path)
        Image.fromarray(overlay).save(overlay_path)

        saved.append(
            {
                "frame_idx": int(frame_idx),
                "importance": float(frame_importance[frame_idx]) if frame_idx < len(frame_importance) else None,
                "type": seg_type,
                "heatmap_path": str(heatmap_path),
                "overlay_path": str(overlay_path),
            }
        )

    return saved


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
    xai_output_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Predict whether a single video is real or AI-generated."""
    frames = torch.from_numpy(load_video(video_path))
    sampled = temporal_sample(frames, num_frames)
    resized = resize_frames(sampled, image_size)

    # overlay 생성용 원본 프레임
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

    payload: Dict[str, Any] = {
        "prediction": prediction,
        "confidence": confidence,
    }

    if outputs is not None:
        frame_importance = outputs.get("frame_importance", None)
        if isinstance(frame_importance, torch.Tensor):
            payload["frame_importance"] = [
                float(value)
                for value in frame_importance.squeeze(0).detach().cpu().tolist()
            ]

        payload["segments"] = outputs.get("segments", [])
        payload["explanations"] = outputs.get("explanations", [])
        payload["xai_method"] = str(outputs.get("xai_method", ""))

        output_root = Path(xai_output_dir) if xai_output_dir is not None else Path(".")
        saved_heatmaps = _save_anomaly_heatmaps(
            video_path=video_path,
            visual_frames=visual_frames,
            outputs=outputs,
            output_root=output_root,
        )

        payload["xai_visualization"] = {
            "method": str(outputs.get("xai_method", "")),
            "heatmaps": saved_heatmaps,
        }

    return payload


def save_prediction(path: Union[str, Path], payload: Dict[str, Any]) -> Path:
    """Persist inference results to JSON."""
    return save_json(path, payload)