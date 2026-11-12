"""Deterministic train/validation splitting."""

from __future__ import annotations

import random
from typing import Sequence

from data_loading.source import SampleRef


def split_samples(
    samples: Sequence[SampleRef],
    val_ratio: float,
    seed: int,
) -> tuple[list[SampleRef], list[SampleRef]]:
    """Split samples into training and validation lists.

    The training list keeps at least one sample when the input is non-empty.
    This function only partitions references. It does not score a model.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in the range [0, 1).")
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    val_count = int(round(len(ordered) * val_ratio))
    if ordered and val_count == len(ordered):
        val_count -= 1
    validation = ordered[:val_count]
    training = ordered[val_count:]
    return training, validation
