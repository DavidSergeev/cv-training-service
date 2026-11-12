"""Serialize training artifacts through an ``ArtifactStore``."""

from __future__ import annotations

import io
import json
from typing import Any

import torch

from config import TrainConfig, dump_config
from exporting.store import ArtifactStore


class ArtifactWriter:
    """Save configs, metrics, checkpoints, and weights without naming a backend."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        checkpoint_name: str,
        weights_name: str,
        config_name: str,
        metrics_name: str,
        metadata_name: str,
    ) -> None:
        self.store = store
        self.checkpoint_name = checkpoint_name
        self.weights_name = weights_name
        self.config_name = config_name
        self.metrics_name = metrics_name
        self.metadata_name = metadata_name

    def save_config(self, config: TrainConfig) -> None:
        self.store.save(self.config_name, dump_config(config).encode("utf-8"))

    def save_metadata(self, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True)
        self.store.save(self.metadata_name, (text + "\n").encode("utf-8"))

    def reset_metrics(self) -> None:
        self.store.save(self.metrics_name, b"")

    def append_metric(self, metric: dict[str, Any]) -> None:
        line = json.dumps(metric, separators=(",", ":")) + "\n"
        existing = self.store.load(self.metrics_name) if self.store.exists(self.metrics_name) else b""
        self.store.save(self.metrics_name, existing + line.encode("utf-8"))

    def save_checkpoint(self, state: dict[str, Any]) -> None:
        self.store.save(self.checkpoint_name, _torch_bytes(state))

    def load_checkpoint(self, map_location: torch.device) -> dict[str, Any] | None:
        if not self.store.exists(self.checkpoint_name):
            return None
        return _torch_load(self.store.load(self.checkpoint_name), map_location)

    def save_weights(self, state_dict: dict[str, Any]) -> None:
        self.store.save(self.weights_name, _torch_bytes(state_dict))


def _torch_bytes(payload: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _torch_load(payload: bytes, map_location: torch.device) -> dict[str, Any]:
    buffer = io.BytesIO(payload)
    try:
        loaded = torch.load(buffer, map_location=map_location, weights_only=False)
    except TypeError:
        buffer.seek(0)
        loaded = torch.load(buffer, map_location=map_location)
    if not isinstance(loaded, dict):
        raise TypeError("Stored checkpoint or weights payload is not a dictionary.")
    return loaded
