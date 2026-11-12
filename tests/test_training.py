"""Training loop, resume, and the composed detection path."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset

from builder import build_trainer
from main import main
from config import TrainConfig, load_config
from training.device import resolve_device
from training.tiny_classifier import TinyClassifier
from training.tiny_detector import TinyDetector
from exporting.local import LocalArtifactStore
from exporting.writer import ArtifactWriter
from training.detection import ObjectDetectionTrainer
from training.losses import build_loss
from training.optim import build_optimizer
from training.supervised import SupervisedTrainer

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, *, epochs: int, resume: bool) -> TrainConfig:
    raw = load_config(ROOT / "configs" / "classification.yaml").model_dump(mode="json")
    raw["runtime"]["device"] = "cpu"
    raw["runtime"]["mixed_precision"] = False
    raw["training"]["epochs"] = epochs
    raw["training"]["resume"] = resume
    raw["training"]["batch_size"] = 2
    raw["loss"] = {"name": "cross_entropy"}
    raw["preparation"]["image_size"] = [32, 32]
    raw["persistence"] = {"kind": "local", "root": str(tmp_path / "artifacts")}
    return TrainConfig.model_validate(raw)


def _writer(config: TrainConfig) -> ArtifactWriter:
    return ArtifactWriter(LocalArtifactStore(config.persistence.root))


def _supervised(config: TrainConfig) -> SupervisedTrainer:
    device = resolve_device(config.runtime.device)
    model = TinyClassifier(num_classes=3).to(device)
    data = TensorDataset(torch.rand(4, 3, 16, 16), torch.tensor([0, 1, 2, 1]))
    loader = DataLoader(data, batch_size=config.training.batch_size, shuffle=True)
    return SupervisedTrainer(
        model=model,
        optimizer=build_optimizer(model, config.optimizer),
        loss_fn=build_loss(config.loss),
        train_loader=loader,
        artifacts=_writer(config),
        config=config,
        device=device,
    )


def test_supervised_loop_writes_metrics_checkpoint_and_weights(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=2, resume=False)
    history = _supervised(config).train()
    assert [metric.epoch for metric in history] == [1, 2]
    assert all(metric.train_loss > 0 for metric in history)

    store = LocalArtifactStore(config.persistence.root)
    metrics = store.load("metrics.jsonl").decode("utf-8").strip().splitlines()
    assert [json.loads(line)["epoch"] for line in metrics] == [1, 2]
    checkpoint = _writer(config).load_checkpoint(torch.device("cpu"))
    assert checkpoint is not None
    assert checkpoint["completed_epochs"] == 2
    assert store.exists("weights.pt")
    assert store.exists("config.yaml")


def test_resume_continues_from_completed_epoch(tmp_path: Path) -> None:
    first = _config(tmp_path, epochs=1, resume=False)
    _supervised(first).train()
    second = _config(tmp_path, epochs=2, resume=True)
    history = _supervised(second).train()
    assert [metric.epoch for metric in history] == [2]
    store = LocalArtifactStore(second.persistence.root)
    epochs = [json.loads(line)["epoch"] for line in store.load("metrics.jsonl").decode("utf-8").splitlines() if line]
    assert epochs == [1, 2]


def test_mixed_precision_on_cpu_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1, resume=False)
    config.runtime.mixed_precision = True
    with pytest.raises(RuntimeError, match="Mixed precision"):
        _supervised(config)


class _Boxes(Dataset):
    def __init__(self) -> None:
        self.items = []
        for _ in range(2):
            image = torch.rand(3, 32, 32)
            target = {
                "boxes": torch.tensor([[4.0, 6.0, 18.0, 20.0]]),
                "labels": torch.tensor([0]),
            }
            self.items.append((image, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def test_detection_trainer_runs_a_backward_step(tmp_path: Path) -> None:
    raw = load_config(ROOT / "configs" / "object_detection.yaml").model_dump(mode="json")
    raw["training"]["epochs"] = 1
    raw["training"]["batch_size"] = 2
    raw["preparation"]["image_size"] = [32, 32]
    raw["persistence"] = {"kind": "local", "root": str(tmp_path / "artifacts")}
    config = TrainConfig.model_validate(raw)
    device = resolve_device("cpu")
    model = TinyDetector(num_classes=2, grid_size=4).to(device)
    loader = DataLoader(_Boxes(), batch_size=2, collate_fn=ObjectDetectionTrainer.batch_collate)
    trainer = ObjectDetectionTrainer(
        model=model,
        optimizer=build_optimizer(model, config.optimizer),
        loss_fn=None,
        train_loader=loader,
        artifacts=_writer(config),
        config=config,
        device=device,
    )
    history = trainer.train()
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0].train_loss))


def test_main_trains_detection_from_local_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    images = root / "images"
    annotations = root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    for index in range(4):
        Image.new("RGB", (24, 24), (index * 20, 10, 10)).save(images / f"sample-{index}.png")
        payload = {"boxes": [[2, 2, 12, 14]], "labels": [index % 2]}
        (annotations / f"sample-{index}.json").write_text(json.dumps(payload), encoding="utf-8")

    raw = load_config(ROOT / "configs" / "object_detection.yaml").model_dump(mode="json")
    raw["runtime"]["device"] = "cpu"
    raw["training"]["epochs"] = 1
    raw["training"]["batch_size"] = 2
    raw["preparation"]["image_size"] = [32, 32]
    raw["data"]["val_ratio"] = 0.25
    raw["data"]["source"]["root"] = str(root)
    raw["persistence"] = {"kind": "local", "root": str(tmp_path / "artifacts")}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("TRAIN_CONFIG", f"path:{config_path}")
    main()

    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    metadata = json.loads(store.load("metadata.json"))
    assert metadata["device"] == "cpu"
    assert metadata["train_samples"] == 3
    assert metadata["val_samples"] == 1
    assert store.exists("weights.pt")
    reloaded = yaml.safe_load(store.load("config.yaml").decode("utf-8"))
    assert reloaded["task"] == "detection"


def test_main_rejects_a_missing_train_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIN_CONFIG", raising=False)
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1


def test_auto_device_selects_an_available_device() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


def test_build_trainer_rejects_an_empty_source(tmp_path: Path) -> None:
    raw = load_config(ROOT / "configs" / "classification.yaml").model_dump(mode="json")
    raw["data"]["source"] = {
        "kind": "local",
        "root": str(tmp_path / "empty"),
        "images_dir": "images",
        "annotations_dir": "annotations",
    }
    (tmp_path / "empty" / "images").mkdir(parents=True)
    (tmp_path / "empty" / "annotations").mkdir()
    raw["persistence"] = {"kind": "local", "root": str(tmp_path / "artifacts")}
    config = TrainConfig.model_validate(raw)
    with pytest.raises(ValueError, match="did not return any samples"):
        build_trainer(config)
