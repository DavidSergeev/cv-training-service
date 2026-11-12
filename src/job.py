"""Resolve ``TRAIN_CONFIG`` to the shared training document."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal
from uuid import UUID

from config import TrainConfig, load_config
from errors import ConfigError
from settings import ServiceSettings


@dataclass(frozen=True)
class ConfigRef:
    kind: Literal["path", "id"]
    value: str


def read_train_config_ref(environ: dict[str, str] | None = None) -> ConfigRef:
    """Read the required ``TRAIN_CONFIG`` variable. There is no default."""
    source = os.environ if environ is None else environ
    raw = source.get("TRAIN_CONFIG")
    if raw is None or not raw.strip():
        raise ConfigError("TRAIN_CONFIG is required and must be non-empty.")
    return parse_train_config_ref(raw)


def parse_train_config_ref(raw: str) -> ConfigRef:
    """Accept ``path:<file>`` or ``id:<uuid>``."""
    text = raw.strip()
    if text.startswith("path:"):
        path = text.removeprefix("path:").strip()
        if not path:
            raise ConfigError("TRAIN_CONFIG path: value must be a non-empty file path.")
        return ConfigRef(kind="path", value=path)
    if text.startswith("id:"):
        value = text.removeprefix("id:").strip()
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ConfigError("TRAIN_CONFIG id: value must be a UUID.") from exc
        return ConfigRef(kind="id", value=str(parsed))
    raise ConfigError("TRAIN_CONFIG must start with 'path:' or 'id:'.")


def resolve_train_config(
    ref: ConfigRef,
    settings: ServiceSettings,
    *,
    fetch: Callable[[str, UUID], Any] | None = None,
) -> TrainConfig:
    """Load the YAML file or the database record into ``TrainConfig``."""
    if ref.kind == "path":
        return load_config(ref.value)
    loader = fetch_config_document if fetch is None else fetch
    document = loader(settings.database_url, UUID(ref.value))
    config = TrainConfig.model_validate(document)
    if config.id != UUID(ref.value):
        raise ConfigError("The database record id does not match TRAIN_CONFIG.")
    return config


def fetch_config_document(database_url: str, config_id: UUID) -> Any:
    """Return the ``config`` JSON stored for ``config_id``.

    The JSON value uses the ``TrainConfig`` schema, including ``id``.
    """
    try:
        import psycopg
    except ImportError as exc:
        raise ConfigError("Reading a training config id requires the psycopg package.") from exc
    try:
        connection = psycopg.connect(database_url)
    except Exception as exc:
        raise ConfigError("Could not connect to the config database.") from exc
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT config FROM train_configs WHERE id = %s", (config_id,))
                row = cursor.fetchone()
    finally:
        connection.close()
    if row is None:
        raise ConfigError(f"Training config {config_id} was not found.")
    payload = row[0]
    if isinstance(payload, (bytes, str)):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ConfigError(f"Training config {config_id} is not a JSON object.")
    return payload
