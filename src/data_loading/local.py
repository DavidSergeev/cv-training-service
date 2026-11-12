"""Local-directory data source.

Expected layout::

    root/
      images/
        sample.jpg
      annotations/
        sample.json
"""

from __future__ import annotations

from pathlib import Path

from data_loading.source import IMAGE_SUFFIXES, DataSource, SampleRef


class LocalDataSource(DataSource):
    def __init__(self, root: str, images_dir: str = "images", annotations_dir: str = "annotations") -> None:
        self.root = Path(root)
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir

    def list_samples(self) -> list[SampleRef]:
        image_root = self.root / self.images_dir
        if not image_root.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_root}")
        paths = sorted(
            path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        samples: list[SampleRef] = []
        for path in paths:
            relative = path.relative_to(image_root)
            annotation = (self.root / self.annotations_dir / relative).with_suffix(".json")
            samples.append(
                SampleRef(
                    sample_id=relative.with_suffix("").as_posix(),
                    image_key=str(path.resolve()),
                    annotation_key=str(annotation.resolve()),
                )
            )
        return samples

    def read_bytes(self, key: str) -> bytes:
        path = Path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()
