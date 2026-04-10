"""Model definitions for AI-generated video detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class VideoClassifierConfig:
    """Configuration for the classifier."""

    encoder_name: str = "MCG-NJU/videomae-base"
    hidden_dim: int = 768
    dropout: float = 0.1
    use_pretrained: bool = True


class FallbackVideoEncoder(nn.Module):
    """Lightweight encoder used when VideoMAE is unavailable."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.output = nn.Linear(64, hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.permute(0, 2, 1, 3, 4)
        features = self.projection(x).flatten(1)
        return self.output(features)


class VideoMAEEncoder(nn.Module):
    """Wrapper that prefers HuggingFace VideoMAE and falls back to a local encoder."""

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
            except Exception:
                pass

        self.model = FallbackVideoEncoder(config.hidden_dim)
        self.uses_transformers = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.uses_transformers:
            outputs = self.model(pixel_values=pixel_values)
            return outputs.last_hidden_state.mean(dim=1)
        return self.model(pixel_values)


class VideoClassifier(nn.Module):
    """VideoMAE encoder plus MLP head for binary classification."""

    def __init__(self, config: VideoClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or VideoClassifierConfig()
        self.encoder = VideoMAEEncoder(self.config)
        hidden_dim = self.encoder.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.encoder(pixel_values)
        logits = self.classifier(features).squeeze(-1)
        return logits
