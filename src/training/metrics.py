"""Per-epoch training metrics. Model quality is intentionally not recorded here."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    duration_seconds: float
    num_batches: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
