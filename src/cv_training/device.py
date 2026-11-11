"""Device selection and tensor movement.

Training logic asks for a ``torch.device`` and does not branch on where the
data was loaded or where artifacts are stored. ``auto`` prefers CUDA when it
is installed and falls back to CPU otherwise.
"""

from __future__ import annotations

from typing import Any

import torch

_DEVICE_CHOICES = {"cpu", "cuda", "auto"}


def resolve_device(selection: str) -> torch.device:
    """Resolve ``cpu``, ``cuda``, or ``auto`` to a concrete device."""
    normalized = selection.lower()
    if normalized not in _DEVICE_CHOICES:
        choices = ", ".join(sorted(_DEVICE_CHOICES))
        raise ValueError(f"device must be one of: {choices}. Got {selection!r}.")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors inside nested dicts and lists onto ``device``."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value
