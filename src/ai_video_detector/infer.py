"""Inference helpers for single-video predictions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import load_video
from .preprocessing import normalize_frames, resize_frames, temporal_sample_clips_with_indices
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
    xai_output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Predict whether a single video is real or AI-generated."""
    frames = torch.from_numpy(load_video(video_path))
    timing = _read_video_timing(video_path, total_frames=frames.shape[0])
    num_clips = _num_clips_for_duration(timing["duration_seconds"])
    clips, clip_frame_indices = temporal_sample_clips_with_indices(frames, num_frames, num_clips)
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
    else:
        normalized = torch.stack([normalize_frames(resize_frames(clip, image_size), mean, std) for clip in clips], dim=0).to(device)
        clip_outputs = []
        logits = model(normalized).reshape(-1)

    clip_scores = torch.sigmoid(logits.reshape(-1)).detach().cpu()
    representative_clip_index = int(torch.argmax(clip_scores).item())
    confidence = float(clip_scores[representative_clip_index].item())
    prediction = "ai_generated" if confidence >= 0.5 else "real"
    clip_predictions = _format_clip_predictions(clip_scores, clip_frame_indices, timing["fps"])
    payload: Dict[str, Any] = {
        "prediction": prediction,
        "confidence": confidence,
        "inference": {
            "duration_seconds": timing["duration_seconds"],
            "fps": timing["fps"],
            "total_frames": int(frames.shape[0]),
            "num_frames_per_clip": int(num_frames),
            "num_clips": int(num_clips),
            "video_score_strategy": "max_clip_confidence",
            "representative_clip_index": representative_clip_index,
        },
        "clip_predictions": clip_predictions,
    }
    if return_xai and clip_outputs:
        representative_output = clip_outputs[representative_clip_index]
        representative_indices = clip_frame_indices[representative_clip_index]
        visual_frames = resize_frames(frames[representative_indices], image_size).clone()
        sampled_frames = _format_sampled_frames(representative_indices, timing["fps"])
        payload["xai"] = _format_xai_output(
            representative_output,
            threshold=xai_threshold,
            clip_index=representative_clip_index,
            sampled_frames=sampled_frames,
            visualizations=_save_anomaly_heatmaps(
                video_path=video_path,
                visual_frames=visual_frames,
                outputs=representative_output,
                sampled_frames=sampled_frames,
                output_root=Path(xai_output_dir),
            )
            if xai_output_dir is not None
            else [],
        )
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


def _format_xai_output(
    outputs: Dict[str, Any],
    threshold: float,
    clip_index: int,
    sampled_frames: List[Dict[str, Any]],
    visualizations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    frame_importance = outputs.get("frame_importance")
    frame_scores = _tensor_to_float_list(frame_importance)
    segments = _map_segments_to_original_frames(outputs.get("segments", []), sampled_frames)
    explanations = outputs.get("explanations", [])
    return {
        "method": str(outputs.get("xai_method", "")),
        "threshold": float(threshold),
        "scope": "representative_clip",
        "clip_index": int(clip_index),
        "frame_importance_scope": "clip_sampled_frame_index",
        "frame_importance": frame_scores,
        "sampled_frames": _attach_importance_to_sampled_frames(sampled_frames, frame_scores),
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
    sampled_frames: List[Dict[str, Any]],
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
            original_frame_index = int(sampled_frames[frame_idx]["original_frame_index"])
            heatmap_path = heatmap_dir / f"{video_stem}_frame{original_frame_index}.png"
            overlay_path = overlay_dir / f"{video_stem}_frame{original_frame_index}.png"
            Image.fromarray(heat_rgb).save(heatmap_path)
            Image.fromarray(overlay).save(overlay_path)
            saved.append(
                {
                    "frame_idx": frame_idx,
                    "original_frame_index": original_frame_index,
                    "timestamp_sec": sampled_frames[frame_idx]["timestamp_sec"],
                    "importance": frame_importance[frame_idx] if frame_idx < len(frame_importance) else None,
                    "type": anomaly_type,
                    "heatmap_path": str(heatmap_path),
                    "overlay_path": str(overlay_path),
                }
            )
    return saved


def _read_video_timing(video_path: Union[str, Path], total_frames: int) -> Dict[str, float]:
    fps = _read_fps_from_metadata(video_path) or DEFAULT_FALLBACK_FPS
    return {
        "fps": float(fps),
        "duration_seconds": float(total_frames / fps),
    }


def _read_fps_from_metadata(video_path: Union[str, Path]) -> Optional[float]:
    source = Path(video_path)
    suffix = source.suffix.lower()
    if suffix == ".gif":
        try:
            with Image.open(source) as image:
                duration_ms = float(image.info.get("duration", 0.0))
            return 1000.0 / duration_ms if duration_ms > 0 else None
        except Exception:
            return None
    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        try:
            import cv2  # type: ignore

            capture = cv2.VideoCapture(str(source))
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            finally:
                capture.release()
            return fps if fps > 0 else None
        except Exception:
            return None
    return None


def _num_clips_for_duration(duration_seconds: float) -> int:
    if duration_seconds <= 5:
        return 1
    if duration_seconds <= 15:
        return 3
    if duration_seconds <= 30:
        return 5
    return max(8, math.ceil(duration_seconds / 5))


def _format_clip_predictions(
    clip_scores: torch.Tensor,
    clip_frame_indices: List[torch.Tensor],
    fps: float,
) -> List[Dict[str, Any]]:
    predictions = []
    for clip_index, score in enumerate(clip_scores.tolist()):
        indices = clip_frame_indices[clip_index]
        confidence = float(score)
        predictions.append(
            {
                "clip_index": clip_index,
                "confidence": confidence,
                "prediction": "ai_generated" if confidence >= 0.5 else "real",
                "start_original_frame_index": int(indices[0].item()),
                "end_original_frame_index": int(indices[-1].item()),
                "start_timestamp_sec": _frame_to_timestamp(indices[0], fps),
                "end_timestamp_sec": _frame_to_timestamp(indices[-1], fps),
                "sampled_frames": _format_sampled_frames(indices, fps),
            }
        )
    return predictions


def _format_sampled_frames(indices: torch.Tensor, fps: float) -> List[Dict[str, Any]]:
    return [
        {
            "sampled_frame_index": sampled_index,
            "original_frame_index": int(original_index.item()),
            "timestamp_sec": _frame_to_timestamp(original_index, fps),
        }
        for sampled_index, original_index in enumerate(indices)
    ]


def _attach_importance_to_sampled_frames(sampled_frames: List[Dict[str, Any]], frame_scores: List[float]) -> List[Dict[str, Any]]:
    frames = []
    for frame in sampled_frames:
        sampled_index = int(frame["sampled_frame_index"])
        frames.append(
            {
                **frame,
                "importance": frame_scores[sampled_index] if sampled_index < len(frame_scores) else None,
            }
        )
    return frames


def _map_segments_to_original_frames(segments: List[Dict[str, Any]], sampled_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapped = []
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        start_frame = sampled_frames[start] if 0 <= start < len(sampled_frames) else None
        end_frame = sampled_frames[end] if 0 <= end < len(sampled_frames) else None
        mapped.append(
            {
                **segment,
                "start_sampled_frame_index": start,
                "end_sampled_frame_index": end,
                "start_original_frame_index": start_frame["original_frame_index"] if start_frame else None,
                "end_original_frame_index": end_frame["original_frame_index"] if end_frame else None,
                "start_timestamp_sec": start_frame["timestamp_sec"] if start_frame else None,
                "end_timestamp_sec": end_frame["timestamp_sec"] if end_frame else None,
            }
        )
    return mapped


def _frame_to_timestamp(frame_index: Union[int, torch.Tensor], fps: float) -> float:
    value = int(frame_index.item()) if isinstance(frame_index, torch.Tensor) else int(frame_index)
    return float(value / fps) if fps > 0 else 0.0


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
