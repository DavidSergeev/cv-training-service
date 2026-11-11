"""Shared supervised training lifecycle.

Concrete trainers inherit from ``BaseTrainer`` and replace ``compute_loss``
when a task needs a different objective. Evaluation is out of scope: this
loop records training loss only.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader

from cv_training.config import TrainConfig
from cv_training.device import move_to_device
from cv_training.persistence.writer import ArtifactWriter
from cv_training.training.metrics import EpochMetrics

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Run forward, loss, backward, and optimizer steps on CPU or CUDA.

    The constructor receives the model, optimizer, loss, data loader, and
    artifact writer. Subclasses do not open datasets or storage backends.
    Mixed precision is enabled only for CUDA and only when configured.

    ``batch_collate`` may be set on a subclass when the default PyTorch
    collate cannot stack that task's targets.
    """

    batch_collate = None

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module | None,
        train_loader: DataLoader,
        artifacts: ArtifactWriter,
        config: TrainConfig,
        device: torch.device,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.artifacts = artifacts
        self.config = config
        self.device = device
        self.run_metadata = run_metadata or {}
        self.use_amp = config.mixed_precision
        if self.use_amp and self.device.type != "cuda":
            raise RuntimeError("Mixed precision is only supported when the resolved device is CUDA.")
        self.scaler = _grad_scaler(self.device) if self.use_amp else None

    def train(self) -> list[EpochMetrics]:
        """Train through the configured epochs and return the new epoch metrics."""
        start_epoch = self._restore_checkpoint() if self.config.training.resume else 0
        epochs = self.config.training.epochs
        if start_epoch >= epochs:
            logger.info("Checkpoint already completed %s epoch(s). Nothing left to run.", start_epoch)
            return []
        if start_epoch == 0:
            self.artifacts.reset_metrics()

        self.artifacts.save_config(self.config)
        metadata = dict(self.run_metadata)
        metadata["device"] = self.device.type
        metadata["mixed_precision"] = self.use_amp
        metadata["resumed_from_epoch"] = start_epoch
        self.artifacts.save_metadata(metadata)

        history: list[EpochMetrics] = []
        for epoch_index in range(start_epoch, epochs):
            metrics = self.train_one_epoch(epoch_index + 1)
            history.append(metrics)
            self.artifacts.append_metric(metrics.to_dict())
            logger.info(
                "epoch %s/%s train_loss=%.6f duration_seconds=%.3f",
                metrics.epoch,
                epochs,
                metrics.train_loss,
                metrics.duration_seconds,
            )
            if self._should_checkpoint(metrics.epoch):
                self._save_checkpoint(metrics.epoch, metrics)

        if history and not self._should_checkpoint(history[-1].epoch):
            self._save_checkpoint(history[-1].epoch, history[-1])
        if history:
            self._save_weights()
        return history

    def train_one_epoch(self, epoch: int) -> EpochMetrics:
        self.model.train()
        total_loss = 0.0
        batches = 0
        started = time.perf_counter()
        for batch in self.train_loader:
            loss = self._step(batch)
            total_loss += float(loss.detach().item())
            batches += 1
        if batches == 0:
            raise RuntimeError("The training loader did not yield any batches.")
        return EpochMetrics(
            epoch=epoch,
            train_loss=total_loss / batches,
            duration_seconds=time.perf_counter() - started,
            num_batches=batches,
        )

    def prepare_batch(self, batch: Any) -> tuple[Any, Any]:
        inputs, targets = batch
        return move_to_device(inputs, self.device), move_to_device(targets, self.device)

    def forward(self, inputs: Any) -> Any:
        return self.model(inputs)

    @abstractmethod
    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """Return a scalar loss for one batch."""

    def _step(self, batch: Any) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        inputs, targets = self.prepare_batch(batch)
        if self.use_amp:
            with torch.autocast(device_type=self.device.type, enabled=True):
                loss = self.compute_loss(self.forward(inputs), targets)
            if self.scaler is None:
                raise RuntimeError("Mixed precision was enabled without a GradScaler.")
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.compute_loss(self.forward(inputs), targets)
            loss.backward()
            self.optimizer.step()
        if loss.ndim != 0:
            raise ValueError("compute_loss must return a scalar tensor.")
        return loss

    def _should_checkpoint(self, epoch: int) -> bool:
        every = self.config.training.checkpoint_every
        return every > 0 and epoch % every == 0

    def _save_checkpoint(self, completed_epochs: int, metrics: EpochMetrics) -> None:
        state: dict[str, Any] = {
            "completed_epochs": completed_epochs,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "latest_metrics": metrics.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }
        self.artifacts.save_checkpoint(state)

    def _save_weights(self) -> None:
        cpu_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        self.artifacts.save_weights(cpu_state)

    def _restore_checkpoint(self) -> int:
        state = self.artifacts.load_checkpoint(self.device)
        if state is None:
            logger.info("No checkpoint found. Starting from scratch.")
            return 0
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        completed = int(state["completed_epochs"])
        logger.info("Resumed from checkpoint at epoch %s.", completed)
        return completed


def _grad_scaler(device: torch.device):
    scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_cls is not None:
        try:
            return scaler_cls(device.type, enabled=True)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=True)
