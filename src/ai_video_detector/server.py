"""HTTP API for video analysis requests."""

from __future__ import annotations

import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .infer import predict_video
from .model import VideoClassifier, VideoClassifierConfig


DEFAULT_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
SUPPORTED_DOWNLOAD_SCHEMES = {"http", "https"}


class AnalyzeRequest(BaseModel):
    """Request payload sent by the backend after uploading a video to S3."""

    s3_url: str = Field(..., description="S3 or presigned S3 URL for the video file")
    request_id: Optional[str] = Field(default=None, description="Optional caller-side request id")

    @field_validator("s3_url")
    @classmethod
    def validate_s3_url(cls, value: str) -> str:
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme not in SUPPORTED_DOWNLOAD_SCHEMES:
            raise ValueError("s3_url must be an http or https URL")
        if not parsed.netloc:
            raise ValueError("s3_url must include a host")
        return value


class AnalyzeResponse(BaseModel):
    request_id: Optional[str] = None
    model: str
    s3_url: str
    result: Dict[str, Any]


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
    with_xai: bool = False
    xai_threshold: float = 0.6
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES

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
            with_xai=_env_bool(f"{prefix}_WITH_XAI", cls.with_xai),
            xai_threshold=_env_float(f"{prefix}_XAI_THRESHOLD", cls.xai_threshold),
            max_download_bytes=_env_int(f"{prefix}_MAX_DOWNLOAD_BYTES", cls.max_download_bytes),
        )


class ModelAnalyzer:
    """Lazy-loaded model wrapper used by API endpoints."""

    def __init__(self, name: str, config: AnalyzerConfig) -> None:
        self.name = name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None

    def analyze_url(self, url: str) -> Dict[str, Any]:
        model = self._load_model()
        suffix = _suffix_from_url(url)
        with tempfile.NamedTemporaryFile(prefix=f"{self.name}_", suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            download_url(url, temp_path, max_bytes=self.config.max_download_bytes)
            return predict_video(
                model,
                temp_path,
                device=self.device,
                num_frames=self.config.num_frames,
                image_size=(self.config.image_size, self.config.image_size),
                return_xai=self.config.with_xai,
                xai_threshold=self.config.xai_threshold,
            )
        finally:
            temp_path.unlink(missing_ok=True)

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
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        return model


def create_app() -> FastAPI:
    api = FastAPI(title="AI Video Detector API")

    @api.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @api.post("/t2v/analyze", response_model=AnalyzeResponse)
    def analyze_t2v(
        request: AnalyzeRequest,
        analyzer: ModelAnalyzer = Depends(get_t2v_analyzer),
    ) -> AnalyzeResponse:
        try:
            result = analyzer.analyze_url(request.s3_url)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        return AnalyzeResponse(
            request_id=request.request_id,
            model=analyzer.name,
            s3_url=request.s3_url,
            result=result,
        )

    return api


@lru_cache(maxsize=1)
def get_t2v_analyzer() -> ModelAnalyzer:
    return ModelAnalyzer("t2v", AnalyzerConfig.from_env("T2V"))


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
