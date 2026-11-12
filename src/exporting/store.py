"""Artifact storage contract. Implementations only move bytes."""

from __future__ import annotations

from abc import ABC, abstractmethod


def validate_relative_key(key: str) -> str:
    if not key or "\\" in key or key.startswith("/"):
        raise ValueError(f"Invalid artifact key: {key!r}.")
    parts = key.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid artifact key: {key!r}.")
    return key


class ArtifactStore(ABC):
    @abstractmethod
    def save(self, key: str, data: bytes) -> None:
        """Write ``data`` at ``key``, replacing any previous value."""

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Return the bytes stored at ``key``."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether ``key`` is present."""
