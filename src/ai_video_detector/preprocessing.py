"""Video preprocessing helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_sample(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Sample a fixed number of frames with index interpolation."""
    total_frames = frames.shape[0]
    if total_frames == num_frames:
        return frames
    if total_frames == 0:
        raise ValueError("frames must contain at least one frame")
    if total_frames < num_frames:
        indices = torch.linspace(0, total_frames - 1, steps=num_frames).round().long()
    else:
        indices = torch.linspace(0, total_frames - 1, steps=num_frames).long()
    return frames[indices]


def resize_frames(frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize frames to the target image size."""
    frames_bchw = frames.permute(0, 3, 1, 2)
    resized = F.interpolate(frames_bchw, size=size, mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def normalize_frames(frames: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    """Normalize float frames into channel-first tensors."""
    frames = frames.float() / 255.0
    frames = frames.permute(0, 3, 1, 2)
    mean_tensor = torch.tensor(mean, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    return (frames - mean_tensor) / std_tensor
