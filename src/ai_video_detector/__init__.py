"""Baseline package for AI-generated video detection."""

from .infer import predict_video
from .metrics import compute_classification_metrics
from .model import VideoClassifier, VideoClassifierConfig
from .train import evaluate_model, train_one_epoch

__all__ = [
    "VideoClassifier",
    "VideoClassifierConfig",
    "compute_classification_metrics",
    "evaluate_model",
    "predict_video",
    "train_one_epoch",
]
