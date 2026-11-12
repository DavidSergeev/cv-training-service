"""Process settings loaded from the local service file.

The database URL lives here. A training job points at a document with
``TRAIN_CONFIG`` and does not carry connection details.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from errors import ConfigError

SERVICE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "service.yaml"

class ServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database_url: str

    @field_validator("database_url")
    @classmethod
    def database_url_is_present(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("database_url must be a non-empty connection string.")
        return text


def load_service_settings(path: str | Path | None = None) -> ServiceSettings:
    """Load the service file shipped with the image or mounted over that path."""
    file_path = Path(path) if path is not None else SERVICE_CONFIG_PATH
    if not file_path.is_file():
        raise FileNotFoundError(f"Service config file does not exist: {file_path}")
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Service config file must contain a mapping: {file_path}")
    return ServiceSettings.model_validate(raw)
