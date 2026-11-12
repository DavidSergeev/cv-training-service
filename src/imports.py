"""Load classes named by configuration."""

from __future__ import annotations

import importlib
from typing import Any


def load_class(class_path: str) -> type[Any]:
    """Return the class named by a dotted path such as ``pkg.module.Class``."""
    module_name, separator, class_name = class_path.rpartition(".")
    if not separator or not module_name or not class_name:
        raise ValueError(f"Expected a dotted class path, got {class_path!r}.")
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"{class_name} was not found in {module_name}.") from exc
    if not isinstance(cls, type):
        raise TypeError(f"{class_path} does not refer to a class.")
    return cls


def instantiate(class_path: str, params: dict[str, Any] | None = None) -> Any:
    """Construct ``class_path`` with keyword arguments from ``params``."""
    return load_class(class_path)(**(params or {}))
