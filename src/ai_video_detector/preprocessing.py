"""Video preprocessing helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_sample(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Sample a fixed number of frames with index interpolation."""
    return frames[temporal_sample_indices(frames.shape[0], num_frames)]


def temporal_sample_indices(total_frames: int, num_frames: int, start: int = 0, end: int | None = None) -> torch.Tensor:
    """Return original frame indices used for fixed-length temporal sampling."""
    end = total_frames if end is None else end
    segment_length = end - start
    if segment_length <= 0:
        raise ValueError("frames must contain at least one frame")
    if segment_length == num_frames:
        return torch.arange(start, end, dtype=torch.long)
    if segment_length < num_frames:
        offsets = torch.linspace(0, segment_length - 1, steps=num_frames).round().long()
    else:
        offsets = torch.linspace(0, segment_length - 1, steps=num_frames).long()
    return offsets + start


def temporal_sample_with_indices(frames: torch.Tensor, num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample frames and return their original frame indices."""
    total_frames = frames.shape[0]
    indices = temporal_sample_indices(total_frames, num_frames)
    return frames[indices], indices


def temporal_sample_clips_with_indices(
    frames: torch.Tensor,
    num_frames: int,
    num_clips: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample multiple clips while preserving original frame indices."""
    total_frames = frames.shape[0]
    clip_indices = temporal_sample_clip_indices(total_frames, num_frames, num_clips)
    return torch.stack([frames[indices] for indices in clip_indices], dim=0), clip_indices


def temporal_sample_clip_indices(
    total_frames: int,
    num_frames: int,
    num_clips: int,
) -> list[torch.Tensor]:
    """Return sampled original frame indices for one or more temporal clips."""
    if total_frames == 0:
        raise ValueError("frames must contain at least one frame")
    if num_clips <= 0:
        raise ValueError("num_clips must be positive")
    if num_clips == 1:
        return [temporal_sample_indices(total_frames, num_frames)]

    boundaries = torch.linspace(0, total_frames, steps=num_clips + 1).round().long()
    clip_indices = []
    for index in range(num_clips):
        start = int(boundaries[index].item())
        end = int(boundaries[index + 1].item())
        if end <= start:
            start = min(start, total_frames - 1)
            end = start + 1
        indices = temporal_sample_indices(total_frames, num_frames, start=start, end=end)
        clip_indices.append(indices)
    return clip_indices


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
