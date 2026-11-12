"""Training configuration loaded from YAML.

The training loop, data sources, and storage backends read this object. They
do not open the YAML file themselves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SERVICE_VERSION = "0.1.0"


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping.")
    return value


def _reject_unknown(data: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown {path} keys: {names}.")


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    return value


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number.")
    return float(value)


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path} must be a non-empty string.")
    return value


@dataclass
class ClassConfig:
    """A class selected by dotted path, plus its constructor arguments."""

    class_path: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str) -> ClassConfig:
        _reject_unknown(data, {"class_path", "params"}, path)
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"{path}.params must be a mapping.")
        return cls(class_path=_as_str(data.get("class_path"), f"{path}.class_path"), params=dict(params))


@dataclass
class NormalizeConfig:
    mean: list[float]
    std: list[float]

    def __post_init__(self) -> None:
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("preparation.normalize mean and std must each contain 3 values.")
        if any(value == 0 for value in self.std):
            raise ValueError("preparation.normalize std values must be non-zero.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizeConfig:
        _reject_unknown(data, {"mean", "std"}, "preparation.normalize")
        return cls(
            mean=[_as_float(value, "preparation.normalize.mean") for value in data.get("mean", [])],
            std=[_as_float(value, "preparation.normalize.std") for value in data.get("std", [])],
        )


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = False
    flip_probability: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError("preparation.augmentation.flip_probability must be in the range [0, 1].")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AugmentationConfig:
        _reject_unknown(data, {"horizontal_flip", "flip_probability"}, "preparation.augmentation")
        return cls(
            horizontal_flip=_as_bool(
                data.get("horizontal_flip", False),
                "preparation.augmentation.horizontal_flip",
            ),
            flip_probability=_as_float(
                data.get("flip_probability", 0.5),
                "preparation.augmentation.flip_probability",
            ),
        )


@dataclass
class PreparationConfig:
    """Preprocessing and augmentation settings.

    ``transform_device`` is accepted so a later GPU pipeline can be selected
    from configuration. Only ``cpu`` is implemented.
    """

    image_size: tuple[int, int]
    normalize: NormalizeConfig | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    transform_device: str = "cpu"

    def __post_init__(self) -> None:
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError("preparation.image_size values must be positive.")
        self.transform_device = self.transform_device.lower()
        if self.transform_device not in {"cpu", "cuda"}:
            raise ValueError("preparation.transform_device must be 'cpu' or 'cuda'.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreparationConfig:
        _reject_unknown(
            data,
            {"image_size", "normalize", "augmentation", "transform_device"},
            "preparation",
        )
        raw_size = data.get("image_size")
        if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
            raise TypeError("preparation.image_size must be [height, width].")
        image_size = (
            _as_int(raw_size[0], "preparation.image_size[0]"),
            _as_int(raw_size[1], "preparation.image_size[1]"),
        )
        normalize = None
        if "normalize" in data and data["normalize"] is not None:
            normalize = NormalizeConfig.from_dict(_mapping(data["normalize"], "preparation.normalize"))
        augmentation = AugmentationConfig()
        if "augmentation" in data and data["augmentation"] is not None:
            augmentation = AugmentationConfig.from_dict(
                _mapping(data["augmentation"], "preparation.augmentation")
            )
        return cls(
            image_size=image_size,
            normalize=normalize,
            augmentation=augmentation,
            transform_device=_as_str(
                data.get("transform_device", "cpu"),
                "preparation.transform_device",
            ),
        )


@dataclass
class DataConfig:
    source: ClassConfig
    annotation_reader: ClassConfig
    val_ratio: float = 0.2
    seed: int = 0
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("data.val_ratio must be in the range [0, 1).")
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be zero or greater.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        _reject_unknown(
            data,
            {"source", "annotation_reader", "val_ratio", "seed", "num_workers"},
            "data",
        )
        return cls(
            source=ClassConfig.from_dict(_mapping(data.get("source"), "data.source"), "data.source"),
            annotation_reader=ClassConfig.from_dict(
                _mapping(data.get("annotation_reader"), "data.annotation_reader"),
                "data.annotation_reader",
            ),
            val_ratio=_as_float(data.get("val_ratio", 0.2), "data.val_ratio"),
            seed=_as_int(data.get("seed", 0), "data.seed"),
            num_workers=_as_int(data.get("num_workers", 0), "data.num_workers"),
        )


@dataclass
class LoopConfig:
    epochs: int
    batch_size: int
    resume: bool = False
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("training.epochs must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("training.batch_size must be at least 1.")
        if self.checkpoint_every < 0:
            raise ValueError("training.checkpoint_every must be zero or greater.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopConfig:
        _reject_unknown(data, {"epochs", "batch_size", "resume", "checkpoint_every"}, "training")
        return cls(
            epochs=_as_int(data.get("epochs"), "training.epochs"),
            batch_size=_as_int(data.get("batch_size"), "training.batch_size"),
            resume=_as_bool(data.get("resume", False), "training.resume"),
            checkpoint_every=_as_int(data.get("checkpoint_every", 1), "training.checkpoint_every"),
        )


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("optimizer.lr must be positive.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizerConfig:
        _reject_unknown(data, {"name", "lr", "params"}, "optimizer")
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("optimizer.params must be a mapping.")
        return cls(
            name=_as_str(data.get("name"), "optimizer.name"),
            lr=_as_float(data.get("lr"), "optimizer.lr"),
            params=dict(params),
        )


@dataclass
class LossConfig:
    """Loss selection.

    ``name`` is either a built-in loss (``cross_entropy``, ``mse``, ``l1``,
    ``bce_with_logits``), a dotted class path, or a task-specific name that
    the concrete trainer interprets (for example ``grid_detection``).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LossConfig:
        _reject_unknown(data, {"name", "params"}, "loss")
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("loss.params must be a mapping.")
        return cls(name=_as_str(data.get("name"), "loss.name"), params=dict(params))


@dataclass
class PersistenceConfig:
    store: ClassConfig
    checkpoint_name: str = "checkpoint.pt"
    weights_name: str = "weights.pt"
    config_name: str = "config.yaml"
    metrics_name: str = "metrics.jsonl"
    metadata_name: str = "metadata.json"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistenceConfig:
        _reject_unknown(
            data,
            {"store", "checkpoint_name", "weights_name", "config_name", "metrics_name", "metadata_name"},
            "persistence",
        )
        return cls(
            store=ClassConfig.from_dict(_mapping(data.get("store"), "persistence.store"), "persistence.store"),
            checkpoint_name=_as_str(data.get("checkpoint_name", "checkpoint.pt"), "persistence.checkpoint_name"),
            weights_name=_as_str(data.get("weights_name", "weights.pt"), "persistence.weights_name"),
            config_name=_as_str(data.get("config_name", "config.yaml"), "persistence.config_name"),
            metrics_name=_as_str(data.get("metrics_name", "metrics.jsonl"), "persistence.metrics_name"),
            metadata_name=_as_str(data.get("metadata_name", "metadata.json"), "persistence.metadata_name"),
        )


@dataclass
class TrainConfig:
    trainer: str
    model: ClassConfig
    data: DataConfig
    preparation: PreparationConfig
    training: LoopConfig
    optimizer: OptimizerConfig
    persistence: PersistenceConfig
    device: str = "cpu"
    mixed_precision: bool = False
    loss: LossConfig | None = None

    def __post_init__(self) -> None:
        self.device = self.device.lower()
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        _reject_unknown(
            data,
            {
                "trainer",
                "model",
                "data",
                "preparation",
                "training",
                "optimizer",
                "persistence",
                "device",
                "mixed_precision",
                "loss",
            },
            "config",
        )
        loss = None
        if "loss" in data and data["loss"] is not None:
            loss = LossConfig.from_dict(_mapping(data["loss"], "loss"))
        return cls(
            trainer=_as_str(data.get("trainer"), "trainer"),
            model=ClassConfig.from_dict(_mapping(data.get("model"), "model"), "model"),
            data=DataConfig.from_dict(_mapping(data.get("data"), "data")),
            preparation=PreparationConfig.from_dict(_mapping(data.get("preparation"), "preparation")),
            training=LoopConfig.from_dict(_mapping(data.get("training"), "training")),
            optimizer=OptimizerConfig.from_dict(_mapping(data.get("optimizer"), "optimizer")),
            persistence=PersistenceConfig.from_dict(_mapping(data.get("persistence"), "persistence")),
            device=_as_str(data.get("device", "cpu"), "device"),
            mixed_precision=_as_bool(data.get("mixed_precision", False), "mixed_precision"),
            loss=loss,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> TrainConfig:
    """Load a training config from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return TrainConfig.from_dict(raw)


def dump_config(config: TrainConfig) -> str:
    """Serialize a config to YAML text for artifact storage."""
    return yaml.safe_dump(config.to_dict(), sort_keys=False)
