"""Wire a config into a trainer. This is the composition root."""

from __future__ import annotations

import logging
import random

import torch
from torch.utils.data import DataLoader

from config import SERVICE_VERSION, TrainConfig
from data_loading.dataset import SupervisedImageDataset
from training.device import resolve_device
from imports import instantiate, load_class
from exporting.writer import ArtifactWriter
from processing.split import split_samples
from processing.preprocessing import build_transform_pipeline
from training.seed import seed_everything
from training.base import BaseTrainer
from training.losses import build_loss
from training.optim import build_optimizer

logger = logging.getLogger(__name__)


def build_trainer(config: TrainConfig) -> BaseTrainer:
    """Construct the trainer described by ``config``.

    The validation dataset is prepared with preprocessing and without
    augmentation. This service does not evaluate it.
    """
    seed_everything(config.data.seed)
    device = resolve_device(config.device)
    logger.info("Resolved device: %s", device.type)

    persistence = config.persistence
    store = instantiate(persistence.store.class_path, persistence.store.params)
    artifacts = ArtifactWriter(
        store,
        checkpoint_name=persistence.checkpoint_name,
        weights_name=persistence.weights_name,
        config_name=persistence.config_name,
        metrics_name=persistence.metrics_name,
        metadata_name=persistence.metadata_name,
    )

    source = instantiate(config.data.source.class_path, config.data.source.params)
    reader = instantiate(config.data.annotation_reader.class_path, config.data.annotation_reader.params)
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

    trainer_cls = load_class(config.trainer)
    if not isinstance(trainer_cls, type) or not issubclass(trainer_cls, BaseTrainer):
        raise TypeError(f"{config.trainer} must be a subclass of BaseTrainer.")

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

    model = instantiate(config.model.class_path, config.model.params)
    model.to(device)
    optimizer = build_optimizer(model, config.optimizer)
    loss_fn = build_loss(config.loss)
    run_metadata = {
        "service_version": SERVICE_VERSION,
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


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
