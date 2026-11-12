"""Training document schema."""

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from config import TrainConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_load() -> None:
    classification = load_config(ROOT / "configs" / "classification.yaml")
    detection = load_config(ROOT / "configs" / "object_detection.yaml")

    assert classification.task == "classification"
    assert classification.runtime.device == "cpu"
    assert classification.loss.name == "cross_entropy"
    assert classification.preparation.image_size == (128, 128)
    assert classification.model.format == "huggingface"
    assert classification.model.source.repo_id == "google/mobilenet_v2_1.0_224"
    assert classification.model.huggingface.num_labels == 3
    assert classification.data.source.kind == "huggingface"
    assert classification.data.source.max_samples == 8
    assert classification.id == UUID("11111111-1111-4111-8111-111111111111")

    assert detection.task == "detection"
    assert detection.loss.name == "grid_detection"
    assert detection.loss.box_weight == 1.0
    assert detection.model.torch.architecture.params["num_classes"] == 2


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "configs" / "classification.yaml"
    text = source.read_text(encoding="utf-8") + "\nextra: true\n"
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_config(path)


def test_classification_rejects_a_detection_loss() -> None:
    raw = load_config(ROOT / "configs" / "classification.yaml").model_dump(mode="json")
    raw["loss"] = {"name": "grid_detection"}
    with pytest.raises(ValidationError, match="cross_entropy"):
        TrainConfig.model_validate(raw)


def test_state_dict_requires_an_architecture() -> None:
    raw = load_config(ROOT / "configs" / "classification.yaml").model_dump(mode="json")
    raw["model"] = {
        "format": "torch_checkpoint",
        "mode": "finetune",
        "trainable": "all",
        "source": {"kind": "local", "path": "models/classifier.pt"},
        "torch": {"weights": "state_dict"},
    }
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)


def test_huggingface_classification_requires_labels() -> None:
    raw = load_config(ROOT / "configs" / "classification.yaml").model_dump(mode="json")
    raw["model"] = {
        "format": "huggingface",
        "mode": "finetune",
        "trainable": "head_only",
        "source": {"kind": "huggingface", "repo_id": "org/model"},
        "huggingface": {"task": "image-classification"},
    }
    with pytest.raises(ValidationError, match="num_labels"):
        TrainConfig.model_validate(raw)
