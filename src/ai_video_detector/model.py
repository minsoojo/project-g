"""Model definitions for AI-generated video detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
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


@dataclass
class EncoderOutputs:
    """Feature vector and optional explainability artifacts."""

    features: torch.Tensor
    frame_importance: Optional[torch.Tensor] = None
    attention_map: Optional[torch.Tensor] = None
    xai_method: str = "none"


class FallbackVideoEncoder(nn.Module):
    """Lightweight encoder used when VideoMAE is unavailable."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.stem = nn.Conv3d(3, 32, kernel_size=3, padding=1)
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
        features = self.projection(stem_activations).flatten(1)
        encoded = self.output(features)
        frame_importance = stem_activations.abs().mean(dim=(1, 3, 4))
        return EncoderOutputs(
            features=encoded,
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
            outputs = self.model(pixel_values=pixel_values, output_attentions=return_attention)
            features = outputs.last_hidden_state.mean(dim=1)
            if not return_attention:
                return EncoderOutputs(features=features)

            frame_importance, attention_map = _compute_videomae_attention_rollup(
                attentions=outputs.attentions,
                num_frames=pixel_values.shape[1],
                config=self.model.config,
            )
            return EncoderOutputs(
                features=features,
                frame_importance=frame_importance,
                attention_map=attention_map,
                xai_method="attention_rollup",
            )
        return self.model.encode(pixel_values)


class VideoClassifier(nn.Module):
    """VideoMAE encoder plus MLP head for binary classification."""

    def __init__(self, config: Optional[VideoClassifierConfig] = None) -> None:
        super().__init__()
        self.config = config or VideoClassifierConfig()
        self.encoder = VideoMAEEncoder(self.config)
        if self.config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_dim = self.encoder.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.encoder(pixel_values)).squeeze(-1)
        return logits

    def predict_with_xai(self, pixel_values: torch.Tensor) -> Dict[str, Optional[Union[torch.Tensor, str]]]:
        """Return logits plus encoder-side explainability artifacts."""
        encoded = self.encoder.encode(pixel_values, return_attention=True)
        logits = self.classifier(encoded.features).squeeze(-1)
        return {
            "logits": logits,
            "frame_importance": encoded.frame_importance,
            "attention_map": encoded.attention_map,
            "xai_method": encoded.xai_method,
        }


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    minimum = scores.min(dim=-1, keepdim=True).values
    maximum = scores.max(dim=-1, keepdim=True).values
    denominator = (maximum - minimum).clamp_min(1e-6)
    return (scores - minimum) / denominator


def _compute_videomae_attention_rollup(
    attentions: Optional[Tuple[torch.Tensor, ...]],
    num_frames: int,
    config: object,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not attentions:
        return None, None

    last_attention = attentions[-1].mean(dim=1)
    token_importance = last_attention.mean(dim=1)
    temporal_tokens = max(1, num_frames // int(getattr(config, "tubelet_size", 2)))
    patch_size = int(getattr(config, "patch_size", 16))
    image_size = getattr(config, "image_size", 224)
    if isinstance(image_size, (list, tuple)):
        image_height = int(image_size[0])
        image_width = int(image_size[1])
    else:
        image_height = int(image_size)
        image_width = int(image_size)
    spatial_tokens = max(1, (image_height // patch_size) * (image_width // patch_size))
    expected_tokens = temporal_tokens * spatial_tokens
    token_importance = token_importance[:, :expected_tokens]
    if token_importance.shape[-1] < expected_tokens:
        padding = expected_tokens - token_importance.shape[-1]
        token_importance = torch.nn.functional.pad(token_importance, (0, padding))
    attention_map = token_importance.view(token_importance.shape[0], temporal_tokens, spatial_tokens)
    frame_importance = attention_map.mean(dim=-1)
    frame_importance = _normalize_scores(frame_importance)
    if temporal_tokens != num_frames:
        frame_importance = torch.nn.functional.interpolate(
            frame_importance.unsqueeze(1),
            size=num_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    return frame_importance, attention_map
