"""Store artifacts under a local directory."""

from __future__ import annotations

from pathlib import Path

from cv_training.persistence.store import ArtifactStore, validate_relative_key


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def save(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def load(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def _resolve(self, key: str) -> Path:
        validate_relative_key(key)
        root = self.root.resolve()
        path = (root / key).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Artifact key escapes the store root: {key!r}.")
        return path
