"""HTTP API for video analysis requests."""

from __future__ import annotations

import os
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .data import load_video
from .infer import predict_video
from .model import VideoClassifier, VideoClassifierConfig, load_video_classifier_state_dict


DEFAULT_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
SUPPORTED_DOWNLOAD_SCHEMES = {"http", "https"}


class AnalyzeRequest(BaseModel):
    """Request payload sent by the backend after uploading a video to S3."""

    s3_url: Optional[str] = Field(default=None, description="S3 or presigned S3 URL for the video file")
    video_url: Optional[str] = Field(default=None, description="Alias for s3_url")
    return_xai: Optional[bool] = Field(default=None, description="Whether to include XAI evidence and heatmaps")
    request_id: Optional[str] = Field(default=None, description="Optional caller-side request id")

    @model_validator(mode="after")
    def normalize_video_url(self) -> "AnalyzeRequest":
        url = self.s3_url or self.video_url
        if not url:
            raise ValueError("Either s3_url or video_url is required")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in SUPPORTED_DOWNLOAD_SCHEMES:
            raise ValueError("s3_url/video_url must be an http or https URL")
        if not parsed.netloc:
            raise ValueError("s3_url/video_url must include a host")
        self.s3_url = url
        return self


class Evidence(BaseModel):
    frame_importance: List[float]
    sampled_frames: List[Dict[str, Any]] = Field(default_factory=list)
    segments: List[Dict[str, Any]]
    explanations: List[str]
    clip_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    representative_clip: Optional[Dict[str, Any]] = None


class HeatmapInfo(BaseModel):
    frame_idx: int
    original_frame_index: Optional[int] = None
    timestamp_sec: Optional[float] = None
    importance: float
    focus_area: List[str]
    heatmap_url: str
    overlay_url: str


class XAIVisualization(BaseModel):
    method: str
    heatmaps: List[HeatmapInfo]


class AnalyzeResponse(BaseModel):
    decision: str
    t2v_prob: float
    model_used: str
    evidence: Evidence
    xai_visualization: XAIVisualization


@dataclass(frozen=True)
class AnalyzerConfig:
    checkpoint_path: Path
    encoder_name: str = "MCG-NJU/videomae-base"
    use_pretrained: bool = True
    freeze_encoder: bool = False
    head_type: str = "mlp"
    transformer_head_layers: int = 2
    transformer_head_heads: int = 8
    transformer_head_ff_dim: int = 2048
    num_frames: int = 16
    image_size: int = 224
    max_clips: int = 8
    with_xai: bool = True
    xai_threshold: float = 0.6
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES
    max_heatmaps: int = 5
    xai_output_dir: Path = Path("outputs/xai")
    model_used: str = "VideoMAE"

    @classmethod
    def from_env(cls, prefix: str) -> "AnalyzerConfig":
        checkpoint = os.getenv(f"{prefix}_CHECKPOINT_PATH")
        if not checkpoint:
            raise RuntimeError(f"{prefix}_CHECKPOINT_PATH is required")
        return cls(
            checkpoint_path=Path(checkpoint),
            encoder_name=os.getenv(f"{prefix}_ENCODER_NAME", cls.encoder_name),
            use_pretrained=_env_bool(f"{prefix}_USE_PRETRAINED", cls.use_pretrained),
            freeze_encoder=_env_bool(f"{prefix}_FREEZE_ENCODER", cls.freeze_encoder),
            head_type=os.getenv(f"{prefix}_HEAD_TYPE", cls.head_type),
            transformer_head_layers=_env_int(f"{prefix}_TRANSFORMER_HEAD_LAYERS", cls.transformer_head_layers),
            transformer_head_heads=_env_int(f"{prefix}_TRANSFORMER_HEAD_HEADS", cls.transformer_head_heads),
            transformer_head_ff_dim=_env_int(f"{prefix}_TRANSFORMER_HEAD_FF_DIM", cls.transformer_head_ff_dim),
            num_frames=_env_int(f"{prefix}_NUM_FRAMES", cls.num_frames),
            image_size=_env_int(f"{prefix}_IMAGE_SIZE", cls.image_size),
            max_clips=_env_int(f"{prefix}_MAX_CLIPS", cls.max_clips),
            with_xai=_env_bool(f"{prefix}_WITH_XAI", cls.with_xai),
            xai_threshold=_env_float(f"{prefix}_XAI_THRESHOLD", cls.xai_threshold),
            max_download_bytes=_env_int(f"{prefix}_MAX_DOWNLOAD_BYTES", cls.max_download_bytes),
            max_heatmaps=_env_int(f"{prefix}_MAX_HEATMAPS", cls.max_heatmaps),
            xai_output_dir=get_xai_output_dir(),
            model_used=os.getenv(f"{prefix}_MODEL_USED", cls.model_used),
        )


class ModelAnalyzer:
    """Lazy-loaded model wrapper used by API endpoints."""

    def __init__(self, name: str, config: AnalyzerConfig) -> None:
        self.name = name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None

    def analyze_url(
        self,
        url: str,
        *,
        request_id: Optional[str] = None,
        return_xai: Optional[bool] = None,
    ) -> AnalyzeResponse:
        model = self._load_model()
        suffix = _suffix_from_url(url)
        with tempfile.NamedTemporaryFile(prefix=f"{self.name}_", suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            download_url(url, temp_path, max_bytes=self.config.max_download_bytes)
            should_return_xai = self.config.with_xai if return_xai is None else return_xai
            result = predict_video(
                model,
                temp_path,
                device=self.device,
                num_frames=self.config.num_frames,
                image_size=(self.config.image_size, self.config.image_size),
                return_xai=should_return_xai,
                xai_threshold=self.config.xai_threshold,
                xai_output_dir=None,
                max_clips=self.config.max_clips,
            )
            return self._format_response(result, temp_path, request_id=request_id)
        finally:
            temp_path.unlink(missing_ok=True)

    def _format_response(
        self,
        result: Dict[str, Any],
        video_path: Path,
        *,
        request_id: Optional[str],
    ) -> AnalyzeResponse:
        t2v_prob = float(result["confidence"])
        xai = result.get("xai", {})
        frame_importance = [float(value) for value in result.get("frame_importance", xai.get("frame_importance", []))]
        sampled_frames = list(xai.get("sampled_frames", []))
        method = str(result.get("xai_method", xai.get("method", "none")))
        heatmaps = self._build_heatmaps(
            video_path,
            frame_importance,
            sampled_frames=sampled_frames,
            request_id=request_id,
        )
        return AnalyzeResponse(
            decision="FAKE" if t2v_prob >= 0.5 else "REAL",
            t2v_prob=t2v_prob,
            model_used=self.config.model_used,
            evidence=Evidence(
                frame_importance=frame_importance,
                sampled_frames=sampled_frames,
                segments=list(result.get("segments", xai.get("segments", []))),
                explanations=list(result.get("explanations", xai.get("explanations", []))),
                clip_predictions=list(result.get("clip_predictions", [])),
                representative_clip=_representative_clip(result),
            ),
            xai_visualization=XAIVisualization(
                method=_normalize_xai_method(method),
                heatmaps=heatmaps,
            ),
        )

    def _build_heatmaps(
        self,
        video_path: Path,
        frame_importance: List[float],
        *,
        sampled_frames: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str],
    ) -> List[HeatmapInfo]:
        if not frame_importance:
            return []

        frames = torch.from_numpy(load_video(video_path))
        selected_indices = _top_frame_indices(frame_importance, self.config.max_heatmaps, self.config.xai_threshold)
        if not selected_indices:
            return []

        video_key = _safe_file_key(request_id or video_path.stem)
        heatmap_dir = self.config.xai_output_dir / "heatmaps"
        overlay_dir = self.config.xai_output_dir / "overlay"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        heatmaps: List[HeatmapInfo] = []
        for frame_idx in selected_indices:
            importance = float(frame_importance[frame_idx])
            frame_meta = _sampled_frame_meta(sampled_frames or [], frame_idx)
            original_frame_index = int(frame_meta.get("original_frame_index", frame_idx))
            original_frame_index = max(0, min(original_frame_index, frames.shape[0] - 1))
            frame = _uint8_frame(frames[original_frame_index].numpy())
            heatmap_path = heatmap_dir / f"{video_key}_frame{original_frame_index}.jpg"
            overlay_path = overlay_dir / f"{video_key}_frame{original_frame_index}.jpg"
            _save_heatmap_images(frame, importance, heatmap_path, overlay_path)
            heatmaps.append(
                HeatmapInfo(
                    frame_idx=frame_idx,
                    original_frame_index=original_frame_index,
                    timestamp_sec=frame_meta.get("timestamp_sec"),
                    importance=importance,
                    focus_area=[],
                    heatmap_url=f"/xai/heatmaps/{heatmap_path.name}",
                    overlay_url=f"/xai/overlay/{overlay_path.name}",
                )
            )
        return heatmaps

    def _load_model(self) -> torch.nn.Module:
        if self.model is not None:
            return self.model
        if not self.config.checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint not found: {self.config.checkpoint_path}")

        model_config = VideoClassifierConfig(
            encoder_name=self.config.encoder_name,
            use_pretrained=self.config.use_pretrained,
            freeze_encoder=self.config.freeze_encoder,
            head_type=self.config.head_type,
            transformer_head_layers=self.config.transformer_head_layers,
            transformer_head_heads=self.config.transformer_head_heads,
            transformer_head_ff_dim=self.config.transformer_head_ff_dim,
        )
        model = VideoClassifier(model_config).to(self.device)
        state_dict = torch.load(self.config.checkpoint_path, map_location=self.device)
        load_video_classifier_state_dict(model, state_dict)
        model.eval()
        self.model = model
        return model


def create_app() -> FastAPI:
    api = FastAPI(title="AI Video Detector API")
    xai_root = get_xai_output_dir()
    xai_root.mkdir(parents=True, exist_ok=True)
    api.mount("/xai", StaticFiles(directory=str(xai_root)), name="xai")

    @api.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @api.post("/t2v/analyze", response_model=AnalyzeResponse)
    def analyze_t2v(
        request: AnalyzeRequest,
        analyzer: ModelAnalyzer = Depends(get_t2v_analyzer),
    ) -> AnalyzeResponse:
        try:
            return analyzer.analyze_url(str(request.s3_url), request_id=request.request_id, return_xai=request.return_xai)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return api


@lru_cache(maxsize=1)
def get_t2v_analyzer() -> ModelAnalyzer:
    return ModelAnalyzer("t2v", AnalyzerConfig.from_env("T2V"))


def get_xai_output_dir() -> Path:
    return Path(os.getenv("XAI_OUTPUT_DIR", "outputs/xai"))


def download_url(url: str, destination: Path, *, max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "ai-video-detector/1.0"})
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError(f"Download exceeds max size of {max_bytes} bytes")
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if downloaded == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("Downloaded video is empty")
    return destination


def _suffix_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    suffix = Path(urllib.parse.unquote(parsed.path)).suffix.lower()
    return suffix if suffix in {".npy", ".pt", ".gif", ".mp4", ".avi", ".mov", ".mkv"} else ".mp4"


def _top_frame_indices(frame_importance: List[float], max_items: int, threshold: float) -> List[int]:
    ranked = sorted(range(len(frame_importance)), key=lambda index: frame_importance[index], reverse=True)
    above_threshold = [index for index in ranked if frame_importance[index] >= threshold]
    selected = above_threshold or ranked[:1]
    return sorted(selected[:max_items])


def _sampled_frame_meta(sampled_frames: List[Dict[str, Any]], sampled_frame_index: int) -> Dict[str, Any]:
    for frame in sampled_frames:
        if int(frame.get("sampled_frame_index", -1)) == sampled_frame_index:
            return frame
    return {"sampled_frame_index": sampled_frame_index, "original_frame_index": sampled_frame_index, "timestamp_sec": None}


def _representative_clip(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    inference = result.get("inference", {})
    clip_index = inference.get("representative_clip_index")
    if clip_index is None:
        return None
    for clip in result.get("clip_predictions", []):
        if clip.get("clip_index") == clip_index:
            return clip
    return None


def _safe_file_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return key or "video"


def _uint8_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    return np.clip(frame, 0, 255).astype(np.uint8)


def _save_heatmap_images(frame: np.ndarray, importance: float, heatmap_path: Path, overlay_path: Path) -> None:
    height, width = frame.shape[:2]
    intensity = int(np.clip(importance, 0.0, 1.0) * 255)
    alpha = np.full((height, width), intensity, dtype=np.uint8)
    red = np.full((height, width), 255, dtype=np.uint8)
    green = np.maximum(0, 180 - alpha // 2).astype(np.uint8)
    blue = np.zeros((height, width), dtype=np.uint8)
    heatmap_rgb = np.stack([red, green, blue], axis=-1)

    heatmap_image = Image.fromarray(heatmap_rgb, mode="RGB")
    frame_image = Image.fromarray(frame, mode="RGB")
    overlay_image = Image.blend(frame_image, heatmap_image, alpha=0.35 + min(importance, 1.0) * 0.25)

    heatmap_image.save(heatmap_path, format="JPEG", quality=90)
    overlay_image.save(overlay_path, format="JPEG", quality=90)


def _normalize_xai_method(method: str) -> str:
    if method == "attention_rollup":
        return "attention_rollout"
    return method


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


app = create_app()
