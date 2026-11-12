"""Training document shared by YAML files and database records.

A job file and a ``train_configs.config`` JSON value are the same object.
Pydantic rejects unknown fields and illegal combinations before training starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from errors import ConfigError

SERVICE_VERSION = "0.1.0"

_MODEL_CONFIG = ConfigDict(extra="forbid", protected_namespaces=())


class RuntimeSettings(BaseModel):
    model_config = _MODEL_CONFIG

    device: Literal["cpu", "cuda", "auto"] = "cpu"
    mixed_precision: bool = False


class LocalModelSource(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["local"]
    path: str

    @field_validator("path")
    @classmethod
    def path_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model.source.path must be a non-empty path.")
        return value


class S3ModelSource(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["s3"]
    bucket: str
    key: str

    @field_validator("bucket", "key")
    @classmethod
    def text_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model source bucket and key must be non-empty.")
        return value


class HuggingFaceSource(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["huggingface"]
    repo_id: str
    revision: str = "main"

    @field_validator("repo_id", "revision")
    @classmethod
    def text_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Hugging Face repo_id and revision must be non-empty.")
        return value


class ArchitectureSpec(BaseModel):
    """Library class used to build a module around a state dict."""

    model_config = _MODEL_CONFIG

    class_path: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_path")
    @classmethod
    def class_path_is_present(cls, value: str) -> str:
        if not value.strip() or "." not in value:
            raise ValueError("torch.architecture.class_path must be a dotted class path.")
        return value


class TorchModuleWeights(BaseModel):
    model_config = _MODEL_CONFIG

    weights: Literal["module"]


class TorchStateDictWeights(BaseModel):
    model_config = _MODEL_CONFIG

    weights: Literal["state_dict"]
    architecture: ArchitectureSpec


TorchWeights = Annotated[TorchModuleWeights | TorchStateDictWeights, Field(discriminator="weights")]
CheckpointSource = Annotated[LocalModelSource | S3ModelSource, Field(discriminator="kind")]


class TorchCheckpointModel(BaseModel):
    model_config = _MODEL_CONFIG

    format: Literal["torch_checkpoint"]
    mode: Literal["finetune", "from_scratch"]
    trainable: Literal["all", "head_only"]
    source: CheckpointSource
    torch: TorchWeights


class HuggingFaceSpec(BaseModel):
    model_config = _MODEL_CONFIG

    task: Literal["image-classification", "object-detection"]
    num_labels: int | None = None
    ignore_mismatched_sizes: bool = False

    @field_validator("num_labels")
    @classmethod
    def labels_are_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("huggingface.num_labels must be at least 1.")
        return value

    @model_validator(mode="after")
    def classification_declares_labels(self) -> HuggingFaceSpec:
        if self.task == "image-classification" and self.num_labels is None:
            raise ValueError("huggingface.num_labels is required for image-classification.")
        return self


class HuggingFaceModel(BaseModel):
    model_config = _MODEL_CONFIG

    format: Literal["huggingface"]
    mode: Literal["finetune", "from_scratch"]
    trainable: Literal["all", "head_only"]
    source: HuggingFaceSource
    huggingface: HuggingFaceSpec


ModelSpec = Annotated[TorchCheckpointModel | HuggingFaceModel, Field(discriminator="format")]


class LocalDataSourceSpec(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["local"]
    root: str
    images_dir: str = "images"
    annotations_dir: str = "annotations"

    @field_validator("root", "images_dir", "annotations_dir")
    @classmethod
    def text_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("data source paths must be non-empty.")
        return value


class S3DataSourceSpec(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["s3"]
    bucket: str
    prefix: str = ""
    images_dir: str = "images"
    annotations_dir: str = "annotations"

    @field_validator("bucket", "images_dir", "annotations_dir")
    @classmethod
    def text_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("data source bucket and directory names must be non-empty.")
        return value


class HuggingFaceDataSourceSpec(BaseModel):
    """A Hub dataset row with an image column and an integer label column.

    ``max_samples`` stops the stream after that many rows. Leave it unset only
    for a dataset that is already small enough to read in full.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["huggingface"]
    repo_id: str
    revision: str = "main"
    split: str = "train"
    image_column: str = "image"
    label_column: str = "label"
    max_samples: int | None = Field(default=None, ge=1)

    @field_validator("repo_id", "revision", "split", "image_column", "label_column")
    @classmethod
    def text_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Hugging Face data source values must be non-empty.")
        return value


DataSourceSpec = Annotated[
    LocalDataSourceSpec | S3DataSourceSpec | HuggingFaceDataSourceSpec,
    Field(discriminator="kind"),
]


class DataSpec(BaseModel):
    model_config = _MODEL_CONFIG

    source: DataSourceSpec
    annotation: Literal["label", "boxes"]
    val_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    seed: int = 0
    num_workers: int = Field(default=0, ge=0)


class NormalizeConfig(BaseModel):
    model_config = _MODEL_CONFIG

    mean: list[float] = Field(min_length=3, max_length=3)
    std: list[float] = Field(min_length=3, max_length=3)

    @field_validator("std")
    @classmethod
    def std_is_nonzero(cls, value: list[float]) -> list[float]:
        if any(item == 0 for item in value):
            raise ValueError("preparation.normalize std values must be non-zero.")
        return value


class AugmentationConfig(BaseModel):
    model_config = _MODEL_CONFIG

    horizontal_flip: bool = False
    flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)


class PreparationConfig(BaseModel):
    model_config = _MODEL_CONFIG

    image_size: tuple[int, int]
    normalize: NormalizeConfig | None = None
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @field_validator("image_size", mode="before")
    @classmethod
    def image_size_pair(cls, value: Any) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("preparation.image_size must be [height, width].")
        height, width = value
        if type(height) is not int or type(width) is not int:
            raise ValueError("preparation.image_size values must be integers.")
        if height <= 0 or width <= 0:
            raise ValueError("preparation.image_size values must be positive.")
        return (height, width)


class LoopConfig(BaseModel):
    model_config = _MODEL_CONFIG

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    resume: bool = False
    checkpoint_every: int = Field(default=1, ge=0)


class OptimizerConfig(BaseModel):
    model_config = _MODEL_CONFIG

    name: Literal["adam", "adamw", "sgd"]
    lr: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class CrossEntropyLossSpec(BaseModel):
    model_config = _MODEL_CONFIG

    name: Literal["cross_entropy"]


class GridDetectionLossSpec(BaseModel):
    model_config = _MODEL_CONFIG

    name: Literal["grid_detection"]
    box_weight: float = Field(default=1.0, ge=0)


LossSpec = Annotated[CrossEntropyLossSpec | GridDetectionLossSpec, Field(discriminator="name")]


class LocalPersistence(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["local"]
    root: str

    @field_validator("root")
    @classmethod
    def root_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("persistence.root must be a non-empty path.")
        return value


class S3Persistence(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["s3"]
    bucket: str
    prefix: str = ""

    @field_validator("bucket")
    @classmethod
    def bucket_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("persistence.bucket must be non-empty.")
        return value


PersistenceSpec = Annotated[LocalPersistence | S3Persistence, Field(discriminator="kind")]


class TrainConfig(BaseModel):
    """One training job. The YAML file and the database record use this schema."""

    model_config = _MODEL_CONFIG

    schema_version: Literal[1]
    id: UUID
    task: Literal["classification", "detection"]
    runtime: RuntimeSettings
    model: ModelSpec
    data: DataSpec
    preparation: PreparationConfig
    training: LoopConfig
    optimizer: OptimizerConfig
    loss: LossSpec
    persistence: PersistenceSpec

    @model_validator(mode="after")
    def task_matches_data_loss_and_model(self) -> TrainConfig:
        if self.task == "classification":
            if self.data.annotation != "label":
                raise ValueError("classification requires data.annotation 'label'.")
            if self.loss.name != "cross_entropy":
                raise ValueError("classification requires loss.name 'cross_entropy'.")
            if self.model.format == "huggingface" and self.model.huggingface.task != "image-classification":
                raise ValueError("classification requires huggingface.task 'image-classification'.")
            return self
        if self.data.annotation != "boxes":
            raise ValueError("detection requires data.annotation 'boxes'.")
        if self.loss.name != "grid_detection":
            raise ValueError("detection requires loss.name 'grid_detection'.")
        if self.model.format == "huggingface" and self.model.huggingface.task != "object-detection":
            raise ValueError("detection requires huggingface.task 'object-detection'.")
        return self


def load_config(path: str | Path) -> TrainConfig:
    """Load a training document from a YAML file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Training config file does not exist: {file_path}")
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Training config file must contain a mapping: {file_path}")
    try:
        return TrainConfig.model_validate(raw)
    except ValidationError:
        raise


def dump_config(config: TrainConfig) -> str:
    """Serialize a training document to YAML text for artifact storage."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
