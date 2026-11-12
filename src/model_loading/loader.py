"""Load a trainable ``nn.Module`` from a checkpoint or a Hugging Face repo."""

from __future__ import annotations

import io
import logging
from typing import Any

import torch
from torch import nn

from config import (
    HuggingFaceModel,
    S3ModelSource,
    TorchCheckpointModel,
    TrainConfig,
)
from imports import instantiate
from s3 import create_s3_client, is_not_found

logger = logging.getLogger(__name__)

_HEAD_NAMES = {"boxes", "classification", "classifier", "fc", "head", "heads"}


def load_model(config: TrainConfig) -> nn.Module:
    """Build the module described by ``config.model`` and apply ``trainable``."""
    spec = config.model
    if spec.format == "torch_checkpoint":
        model = load_torch_checkpoint(spec)
    else:
        model = load_huggingface(spec)
    if spec.trainable == "head_only":
        freeze_backbone(model)
    logger.info("Loaded %s model in %s mode (%s).", spec.format, spec.mode, spec.trainable)
    return model


def load_torch_checkpoint(spec: TorchCheckpointModel) -> nn.Module:
    """Load a full module, or build one and optionally fill its state dict."""
    if spec.torch.weights == "state_dict":
        model = instantiate(spec.torch.architecture.class_path, spec.torch.architecture.params)
        if not isinstance(model, nn.Module):
            raise TypeError(f"{spec.torch.architecture.class_path} did not construct an nn.Module.")
        if spec.mode == "from_scratch":
            return model
        state = _torch_load(read_model_bytes(spec.source), weights_only=True)
        if not isinstance(state, dict):
            raise TypeError("torch.weights 'state_dict' requires a dictionary saved with torch.save.")
        model.load_state_dict(state)
        return model

    loaded = _torch_load(read_model_bytes(spec.source), weights_only=False)
    if not isinstance(loaded, nn.Module):
        raise TypeError("torch.weights 'module' requires torch.save of an nn.Module.")
    if spec.mode == "from_scratch":
        _reset_parameters(loaded)
    return loaded


def load_huggingface(spec: HuggingFaceModel) -> nn.Module:
    """Load a Transformers image model. ``from_scratch`` uses the repo config only."""
    try:
        from transformers import AutoConfig, AutoModelForImageClassification, AutoModelForObjectDetection
    except ImportError as exc:
        raise ImportError("Hugging Face training requires the transformers package.") from exc

    classes = {
        "image-classification": AutoModelForImageClassification,
        "object-detection": AutoModelForObjectDetection,
    }
    model_cls = classes[spec.huggingface.task]
    source = spec.source
    if spec.mode == "from_scratch":
        hf_config = AutoConfig.from_pretrained(source.repo_id, revision=source.revision)
        if spec.huggingface.num_labels is not None:
            hf_config.num_labels = spec.huggingface.num_labels
        model = model_cls.from_config(hf_config)
    else:
        kwargs: dict[str, Any] = {
            "revision": source.revision,
            "ignore_mismatched_sizes": spec.huggingface.ignore_mismatched_sizes,
        }
        if spec.huggingface.num_labels is not None:
            kwargs["num_labels"] = spec.huggingface.num_labels
        model = model_cls.from_pretrained(source.repo_id, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("Hugging Face did not return an nn.Module.")
    return model


def read_model_bytes(source: Any) -> bytes:
    """Read a checkpoint from the local filesystem or S3."""
    if isinstance(source, S3ModelSource) or getattr(source, "kind", None) == "s3":
        try:
            response = create_s3_client().get_object(Bucket=source.bucket, Key=source.key)
        except Exception as exc:
            if is_not_found(exc):
                raise FileNotFoundError(source.key) from exc
            raise
        return response["Body"].read()
    path = source.path
    with open(path, "rb") as handle:
        return handle.read()


def freeze_backbone(model: nn.Module) -> None:
    """Train parameters whose names contain a known head, and freeze the rest."""
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        is_head = bool(set(name.split(".")) & _HEAD_NAMES)
        parameter.requires_grad = is_head
        if is_head:
            trainable.append(name)
    if not trainable:
        names = ", ".join(sorted(_HEAD_NAMES))
        raise ValueError(f"trainable 'head_only' found no head parameters. Expected a name containing: {names}.")


def _reset_parameters(model: nn.Module) -> None:
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


def _torch_load(payload: bytes, *, weights_only: bool) -> Any:
    buffer = io.BytesIO(payload)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=weights_only)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")
