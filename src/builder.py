"""Wire a config into a trainer. This is the composition root."""

from __future__ import annotations

import logging
import random

import torch
from torch.utils.data import DataLoader

from config import SERVICE_VERSION, DataSourceSpec, PersistenceSpec, TrainConfig
from data_loading.annotations import JsonBoxAnnotationReader, JsonLabelAnnotationReader
from data_loading.dataset import SupervisedImageDataset
from data_loading.huggingface import HuggingFaceDataSource
from data_loading.local import LocalDataSource
from data_loading.s3 import S3DataSource
from data_loading.source import DataSource
from exporting.local import LocalArtifactStore
from exporting.s3 import S3ArtifactStore
from exporting.writer import ArtifactWriter
from model_loading import load_model
from processing.preprocessing import build_transform_pipeline
from processing.split import split_samples
from training.base import BaseTrainer
from training.detection import ObjectDetectionTrainer
from training.device import resolve_device
from training.losses import build_loss
from training.optim import build_optimizer
from training.seed import seed_everything
from training.supervised import SupervisedTrainer

logger = logging.getLogger(__name__)

_TRAINERS: dict[str, type[BaseTrainer]] = {
    "classification": SupervisedTrainer,
    "detection": ObjectDetectionTrainer,
}


def build_trainer(config: TrainConfig) -> BaseTrainer:
    """Construct the trainer described by ``config``.

    The validation dataset is prepared with preprocessing and without
    augmentation. This service does not evaluate it.
    """
    seed_everything(config.data.seed)
    device = resolve_device(config.runtime.device)
    logger.info("Resolved device: %s", device.type)

    store = build_store(config.persistence)
    artifacts = ArtifactWriter(store)

    source = build_data_source(config.data.source)
    reader = JsonLabelAnnotationReader() if config.data.annotation == "label" else JsonBoxAnnotationReader()
    samples = source.list_samples()
    if not samples:
        raise ValueError("The data source did not return any samples.")
    train_samples, val_samples = split_samples(samples, config.data.val_ratio, config.data.seed)
    if not train_samples:
        raise ValueError("The training split is empty. Lower data.val_ratio or add samples.")

    train_dataset = SupervisedImageDataset(
        source,
        train_samples,
        reader,
        build_transform_pipeline(config.preparation, augment=True),
    )
    val_dataset = SupervisedImageDataset(
        source,
        val_samples,
        reader,
        build_transform_pipeline(config.preparation, augment=False),
    )
    logger.info(
        "Prepared %s training sample(s) and %s validation sample(s).",
        len(train_dataset),
        len(val_dataset),
    )

    trainer_cls = _TRAINERS[config.task]
    generator = torch.Generator()
    generator.manual_seed(config.data.seed)
    loader_kwargs: dict = {
        "dataset": train_dataset,
        "batch_size": config.training.batch_size,
        "shuffle": True,
        "num_workers": config.data.num_workers,
        "pin_memory": device.type == "cuda",
        "generator": generator,
    }
    collate = getattr(trainer_cls, "batch_collate", None)
    if collate is not None:
        loader_kwargs["collate_fn"] = collate
    if config.data.num_workers > 0:
        loader_kwargs["worker_init_fn"] = _seed_worker
    train_loader = DataLoader(**loader_kwargs)

    model = load_model(config)
    model.to(device)
    optimizer = build_optimizer(model, config.optimizer)
    loss_fn = build_loss(config.loss)
    run_metadata = {
        "service_version": SERVICE_VERSION,
        "config_id": str(config.id),
        "train_sample_ids": [sample.sample_id for sample in train_samples],
        "val_sample_ids": [sample.sample_id for sample in val_samples],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    return trainer_cls(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        artifacts=artifacts,
        config=config,
        device=device,
        run_metadata=run_metadata,
    )


def build_data_source(spec: DataSourceSpec) -> DataSource:
    if spec.kind == "local":
        return LocalDataSource(spec.root, spec.images_dir, spec.annotations_dir)
    if spec.kind == "huggingface":
        return HuggingFaceDataSource(
            repo_id=spec.repo_id,
            revision=spec.revision,
            split=spec.split,
            image_column=spec.image_column,
            label_column=spec.label_column,
            max_samples=spec.max_samples,
        )
    return S3DataSource(spec.bucket, spec.prefix, spec.images_dir, spec.annotations_dir)


def build_store(spec: PersistenceSpec) -> LocalArtifactStore | S3ArtifactStore:
    if spec.kind == "local":
        return LocalArtifactStore(spec.root)
    return S3ArtifactStore(spec.bucket, spec.prefix)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
