"""Data-source contract shared by filesystem and object-storage loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def has_image_suffix(key: str) -> bool:
    dot = key.rfind(".")
    slash = max(key.rfind("/"), key.rfind("\\"))
    if dot == -1 or dot < slash:
        return False
    return key[dot:].lower() in IMAGE_SUFFIXES


@dataclass(frozen=True)
class SampleRef:
    """Pointer to one supervised sample. Keys are interpreted by the source."""

    sample_id: str
    image_key: str
    annotation_key: str


class DataSource(ABC):
    """Read sample payloads without knowing how training will use them."""

    @abstractmethod
    def list_samples(self) -> list[SampleRef]:
        """Return every sample this source can load."""

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Return the raw bytes stored at ``key``."""
