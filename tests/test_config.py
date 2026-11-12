"""Config parsing."""

from pathlib import Path

import pytest

from config import TrainConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_load() -> None:
    classification = load_config(ROOT / "configs" / "classification.yaml")
    detection = load_config(ROOT / "configs" / "object_detection.yaml")

    assert classification.trainer.endswith("SupervisedTrainer")
    assert classification.device == "cpu"
    assert classification.loss is not None
    assert classification.loss.name == "cross_entropy"
    assert classification.preparation.image_size == (64, 64)

    assert detection.trainer.endswith("ObjectDetectionTrainer")
    assert detection.loss is not None
    assert detection.loss.name == "grid_detection"
    assert detection.model.params["num_classes"] == 2


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "configs" / "classification.yaml"
    text = source.read_text(encoding="utf-8") + "\nextra: true\n"
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_cuda_transform_device_is_stored_for_later_rejection() -> None:
    detection = load_config(ROOT / "configs" / "object_detection.yaml").to_dict()
    detection["preparation"]["transform_device"] = "cuda"
    config = TrainConfig.from_dict(detection)
    assert config.preparation.transform_device == "cuda"
