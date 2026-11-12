"""Checkpoint and Hugging Face model loading."""

import sys
import types
from pathlib import Path

import pytest
import torch

from config import TrainConfig, load_config
from model_loading.loader import load_model
from training.tiny_classifier import TinyClassifier

ROOT = Path(__file__).resolve().parents[1]


def _classification(**model: object) -> TrainConfig:
    raw = load_config(ROOT / "configs" / "classification.yaml").model_dump(mode="json")
    checkpoint = {
        "format": "torch_checkpoint",
        "mode": "from_scratch",
        "trainable": "all",
        "source": {"kind": "local", "path": "models/classifier.pt"},
        "torch": {
            "weights": "state_dict",
            "architecture": {
                "class_path": "training.tiny_classifier.TinyClassifier",
                "params": {"num_classes": 3},
            },
        },
    }
    checkpoint.update(model)
    raw["model"] = checkpoint
    return TrainConfig.model_validate(raw)


def test_from_scratch_state_dict_builds_the_architecture_without_reading_a_file(tmp_path: Path) -> None:
    config = _classification()
    config.model.source.path = str(tmp_path / "missing.pt")
    model = load_model(config)
    assert isinstance(model, TinyClassifier)
    assert model.head.out_features == 3


def test_state_dict_finetune_loads_saved_weights(tmp_path: Path) -> None:
    source = TinyClassifier(num_classes=3)
    with torch.no_grad():
        source.head.weight.fill_(0.25)
    path = tmp_path / "classifier.pt"
    torch.save(source.state_dict(), path)

    config = _classification(mode="finetune")
    config.model.source.path = str(path)
    loaded = load_model(config)
    assert torch.allclose(loaded.head.weight, source.head.weight)


def test_full_module_finetune_returns_the_saved_module(tmp_path: Path) -> None:
    source = TinyClassifier(num_classes=3)
    path = tmp_path / "classifier.pt"
    torch.save(source, path)
    config = _classification(
        mode="finetune",
        torch={"weights": "module"},
    )
    config.model.source.path = str(path)
    loaded = load_model(config)
    assert isinstance(loaded, TinyClassifier)


def test_head_only_freezes_the_backbone() -> None:
    config = _classification(trainable="head_only")
    model = load_model(config)
    assert model.features[0].weight.requires_grad is False
    assert model.head.weight.requires_grad is True


def test_huggingface_finetune_calls_from_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise AssertionError("from_scratch was not requested")

    class _Classification:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs):
            calls.append((repo_id, kwargs))
            return TinyClassifier(num_classes=kwargs["num_labels"])

        @staticmethod
        def from_config(_config):
            raise AssertionError("from_scratch was not requested")

    class _Detection:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise AssertionError("detection was not requested")

        @staticmethod
        def from_config(_config):
            raise AssertionError("detection was not requested")

    module = types.ModuleType("transformers")
    module.AutoConfig = _AutoConfig
    module.AutoModelForImageClassification = _Classification
    module.AutoModelForObjectDetection = _Detection
    monkeypatch.setitem(sys.modules, "transformers", module)

    raw = load_config(ROOT / "configs" / "classification.yaml").model_dump(mode="json")
    raw["model"] = {
        "format": "huggingface",
        "mode": "finetune",
        "trainable": "all",
        "source": {"kind": "huggingface", "repo_id": "org/model", "revision": "main"},
        "huggingface": {"task": "image-classification", "num_labels": 3, "ignore_mismatched_sizes": True},
    }
    model = load_model(TrainConfig.model_validate(raw))
    assert isinstance(model, TinyClassifier)
    assert calls == [
        (
            "org/model",
            {"revision": "main", "ignore_mismatched_sizes": True, "num_labels": 3},
        )
    ]
