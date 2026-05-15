"""Model definitions for AI-generated video detection."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class VideoClassifierConfig:
    """Configuration for the classifier."""

    encoder_name: str = "MCG-NJU/videomae-base"
    hidden_dim: int = 768
    dropout: float = 0.1
    use_pretrained: bool = True
    freeze_encoder: bool = False
    head_type: str = "mlp"
    transformer_head_layers: int = 2
    transformer_head_heads: int = 8
    transformer_head_ff_dim: int = 2048


@dataclass
class EncoderOutputs:
    """Feature vector and optional explainability artifacts."""

    features: torch.Tensor
    sequence_features: Optional[torch.Tensor] = None
    frame_importance: Optional[torch.Tensor] = None
    attention_map: Optional[torch.Tensor] = None
    xai_method: str = "none"


class FallbackVideoEncoder(nn.Module):
    """Lightweight encoder used when VideoMAE is unavailable."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.stem = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.temporal_output = nn.Linear(32, hidden_dim)
        self.projection = nn.Sequential(
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.output = nn.Linear(64, hidden_dim)

    def encode(self, pixel_values: torch.Tensor) -> EncoderOutputs:
        x = pixel_values.permute(0, 2, 1, 3, 4)
        stem_activations = self.stem(x)
        sequence_features = self.temporal_output(stem_activations.mean(dim=(3, 4)).permute(0, 2, 1))
        features = self.projection(stem_activations).flatten(1)
        encoded = self.output(features)
        frame_importance = stem_activations.abs().mean(dim=(1, 3, 4))
        return EncoderOutputs(
            features=encoded,
            sequence_features=sequence_features,
            frame_importance=_normalize_scores(frame_importance),
            attention_map=frame_importance,
            xai_method="activation_energy",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode(pixel_values).features


class VideoMAEEncoder(nn.Module):
    """Wrapper that loads HuggingFace VideoMAE or an explicit local fallback."""

    def __init__(self, config: VideoClassifierConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.model: nn.Module

        if config.use_pretrained:
            try:
                from transformers import VideoMAEModel  # type: ignore

                self.model = VideoMAEModel.from_pretrained(config.encoder_name)
                self.hidden_dim = int(self.model.config.hidden_size)
                self.uses_transformers = True
                return
            except Exception as exc:
                logger.exception("Failed to load VideoMAE encoder from '%s'", config.encoder_name)
                raise RuntimeError(
                    f"Unable to load VideoMAE encoder from '{config.encoder_name}'. "
                    "Fix the model path/network access or run with use_pretrained=False to use the fallback encoder."
                ) from exc

        self.model = FallbackVideoEncoder(config.hidden_dim)
        self.uses_transformers = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode(pixel_values).features

    def encode(self, pixel_values: torch.Tensor, return_attention: bool = False) -> EncoderOutputs:
        if self.uses_transformers:
            if return_attention:
                _enable_eager_attention(self.model)
            outputs = self.model(pixel_values=pixel_values, output_attentions=return_attention)
            sequence_features = outputs.last_hidden_state
            features = sequence_features.mean(dim=1)
            if not return_attention:
                return EncoderOutputs(features=features, sequence_features=sequence_features)

            frame_importance, attention_map = _compute_videomae_attention_rollup(
                attentions=outputs.attentions,
                num_frames=pixel_values.shape[1],
                config=self.model.config,
            )
            xai_method = "attention_rollup" if frame_importance is not None else "unavailable"
            return EncoderOutputs(
                features=features,
                sequence_features=sequence_features,
                frame_importance=frame_importance,
                attention_map=attention_map,
                xai_method=xai_method,
            )
        return self.model.encode(pixel_values)


class MLPClassifierHead(nn.Module):
    """MLP classifier head for pooled video features."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class TransformerClassifierHead(nn.Module):
    """Transformer encoder head for token or temporal video features."""

    def __init__(self, hidden_dim: int, layers: int, heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by transformer_head_heads={heads}")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        batch_size = sequence_features.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_token, sequence_features], dim=1)
        sequence = sequence + _sinusoidal_positions(
            sequence_length=sequence.shape[1],
            hidden_dim=sequence.shape[2],
            device=sequence.device,
            dtype=sequence.dtype,
        )
        encoded = self.encoder(sequence)
        cls_features = self.norm(encoded[:, 0])
        return self.output(self.dropout(cls_features))


class VideoClassifier(nn.Module):
    """VideoMAE encoder plus selectable head for binary classification."""

    def __init__(self, config: Optional[VideoClassifierConfig] = None) -> None:
        super().__init__()
        self.config = config or VideoClassifierConfig()
        self.encoder = VideoMAEEncoder(self.config)
        if self.config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_dim = self.encoder.hidden_dim
        head_type = self.config.head_type.lower()
        if head_type == "mlp":
            self.classifier = MLPClassifierHead(hidden_dim, self.config.dropout)
        elif head_type == "transformer":
            self.classifier = TransformerClassifierHead(
                hidden_dim=hidden_dim,
                layers=self.config.transformer_head_layers,
                heads=self.config.transformer_head_heads,
                ff_dim=self.config.transformer_head_ff_dim,
                dropout=self.config.dropout,
            )
        else:
            raise ValueError(f"Unsupported head_type: {self.config.head_type}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder.encode(pixel_values)
        logits = self._classify(encoded).squeeze(-1)
        return logits

    def predict_with_xai(
        self,
        pixel_values: torch.Tensor,
        threshold: float = 0.6,
    ) -> Dict[str, Optional[Union[torch.Tensor, str, List[Dict[str, Union[int, float, str]]], List[str]]]]:
        """Return logits plus encoder-side explainability artifacts."""
        encoded = self.encoder.encode(pixel_values, return_attention=True)
        logits = self._classify(encoded).squeeze(-1)
        segments: List[Dict[str, Union[int, float, str]]] = []
        explanations: List[str] = []
        if encoded.frame_importance is not None:
            frame_importance = encoded.frame_importance.squeeze(0)
            for start_frame, end_frame in extract_segments(frame_importance, threshold=threshold):
                segment_importance = frame_importance[start_frame : end_frame + 1]
                segment_attention = _slice_segment_attention(
                    attention_map=encoded.attention_map,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    num_frames=pixel_values.shape[1],
                )
                anomaly_type = classify_anomaly(segment_importance, segment_attention)
                confidence = float(segment_importance.mean().item())
                segments.append(
                    {
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "type": anomaly_type,
                        "confidence": confidence,
                    }
                )
                explanations.append(generate_explanation((start_frame, end_frame), anomaly_type, confidence))
        return {
            "logits": logits,
            "frame_importance": encoded.frame_importance,
            "attention_map": encoded.attention_map,
            "segments": segments,
            "explanations": explanations,
            "xai_method": encoded.xai_method,
        }

    def _classify(self, encoded: EncoderOutputs) -> torch.Tensor:
        if isinstance(self.classifier, TransformerClassifierHead):
            if encoded.sequence_features is None:
                raise ValueError("Transformer head requires sequence features from the encoder")
            return self.classifier(encoded.sequence_features)
        return self.classifier(encoded.features)


def load_video_classifier_state_dict(model: VideoClassifier, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load checkpoints from both current and legacy classifier key layouts."""
    model.load_state_dict(_normalize_classifier_state_dict_keys(state_dict))


def _normalize_classifier_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "classifier.0.weight" not in state_dict or "classifier.layers.0.weight" in state_dict:
        return state_dict

    normalized = dict(state_dict)
    for suffix in ("weight", "bias"):
        legacy_first = f"classifier.0.{suffix}"
        legacy_output = f"classifier.3.{suffix}"
        if legacy_first in normalized:
            normalized[f"classifier.layers.0.{suffix}"] = normalized.pop(legacy_first)
        if legacy_output in normalized:
            normalized[f"classifier.layers.3.{suffix}"] = normalized.pop(legacy_output)
    return normalized


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    minimum = scores.min(dim=-1, keepdim=True).values
    maximum = scores.max(dim=-1, keepdim=True).values
    denominator = (maximum - minimum).clamp_min(1e-6)
    return (scores - minimum) / denominator


def _sinusoidal_positions(
    sequence_length: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=device, dtype=dtype).unsqueeze(1)
    dimensions = torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
    div_term = torch.exp(dimensions * (-math.log(10000.0) / hidden_dim))
    encoding = torch.zeros(sequence_length, hidden_dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


def _enable_eager_attention(model: nn.Module) -> None:
    set_attention = getattr(model, "set_attn_implementation", None)
    if callable(set_attention):
        set_attention("eager")
        return

    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "attn_implementation"):
        setattr(config, "attn_implementation", "eager")


def _slice_segment_attention(
    attention_map: Optional[torch.Tensor],
    start_frame: int,
    end_frame: int,
    num_frames: int,
) -> Optional[torch.Tensor]:
    if not isinstance(attention_map, torch.Tensor) or attention_map.ndim != 4:
        return attention_map

    temporal_tokens = attention_map.shape[1]
    token_start = min(
        temporal_tokens - 1,
        int(round(start_frame * (temporal_tokens - 1) / max(num_frames - 1, 1))),
    )
    token_end = min(
        temporal_tokens - 1,
        int(round(end_frame * (temporal_tokens - 1) / max(num_frames - 1, 1))),
    )
    return attention_map[:, token_start : token_end + 1]


def _compute_videomae_attention_rollup(
    attentions: Optional[Tuple[torch.Tensor, ...]],
    num_frames: int,
    config: object,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not attentions:
        return None, None

    batch_size = attentions[0].shape[0]
    num_tokens = attentions[0].shape[-1]
    device = attentions[0].device
    eye = torch.eye(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    rollout = eye

    for attention in attentions:
        layer_attention = attention.mean(dim=1)
        layer_attention = layer_attention + eye
        layer_attention = layer_attention / layer_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rollout = torch.bmm(layer_attention, rollout)

    if rollout.shape[-1] > 1:
        token_importance = rollout[:, 0, :]
    else:
        token_importance = rollout.mean(dim=1)

    temporal_tokens = max(1, num_frames // int(getattr(config, "tubelet_size", 2)))
    patch_size = int(getattr(config, "patch_size", 16))
    image_size = getattr(config, "image_size", 224)
    if isinstance(image_size, (list, tuple)):
        image_height = int(image_size[0])
        image_width = int(image_size[1])
    else:
        image_height = int(image_size)
        image_width = int(image_size)
    grid_h = max(1, image_height // patch_size)
    grid_w = max(1, image_width // patch_size)
    spatial_tokens = grid_h * grid_w
    expected_tokens = temporal_tokens * spatial_tokens
    if token_importance.shape[-1] == expected_tokens + 1:
        token_importance = token_importance[:, 1:]
    else:
        token_importance = token_importance[:, :expected_tokens]
    if token_importance.shape[-1] < expected_tokens:
        padding = expected_tokens - token_importance.shape[-1]
        token_importance = F.pad(token_importance, (0, padding))
    attention_map = token_importance.view(batch_size, temporal_tokens, grid_h, grid_w)
    frame_importance = attention_map.mean(dim=(2, 3))
    frame_importance = _normalize_scores(frame_importance)
    if temporal_tokens != num_frames:
        frame_importance = F.interpolate(
            frame_importance.unsqueeze(1),
            size=num_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    return frame_importance, attention_map


def extract_segments(frame_importance: torch.Tensor, threshold: float = 0.6) -> List[Tuple[int, int]]:
    """Group consecutive high-importance frames into segments."""
    segments: List[Tuple[int, int]] = []
    current_start: Optional[int] = None

    for index, score in enumerate(frame_importance):
        if float(score.item()) >= threshold:
            if current_start is None:
                current_start = index
        elif current_start is not None:
            segments.append((current_start, index - 1))
            current_start = None

    if current_start is not None:
        segments.append((current_start, frame_importance.shape[0] - 1))

    return segments


def classify_anomaly(frame_importance: torch.Tensor, attention_map: Optional[torch.Tensor]) -> str:
    if attention_map is None:
        return "unknown"

    temporal_var = float(frame_importance.var().item())
    spatial_var = float(attention_map.var().item()) if isinstance(attention_map, torch.Tensor) else 0.0
    mean_importance = float(frame_importance.mean().item())
    if temporal_var > 0.1:
        return "movement anomaly"
    if spatial_var > 0.1:
        return "texture jitter"
    if mean_importance > 0.7:
        return "lighting anomaly"
    return "object inconsistency"


def generate_explanation(segment: Tuple[int, int], anomaly_type: str, confidence: float) -> str:
    start_frame, end_frame = segment
    return f"Frames {start_frame} to {end_frame} show {anomaly_type} (confidence {confidence:.2f})"
