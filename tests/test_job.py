"""TRAIN_CONFIG resolution."""

from pathlib import Path
from uuid import UUID

import pytest

from config import load_config
from errors import ConfigError
from job import parse_train_config_ref, read_train_config_ref, resolve_train_config
from settings import ServiceSettings, load_service_settings

ROOT = Path(__file__).resolve().parents[1]


def test_service_settings_load_the_database_url() -> None:
    settings = load_service_settings()
    assert settings.database_url.startswith("postgresql://")


def test_missing_train_config_is_rejected() -> None:
    with pytest.raises(ConfigError, match="TRAIN_CONFIG is required"):
        read_train_config_ref({})


def test_train_config_prefix_selects_the_source() -> None:
    assert parse_train_config_ref("path:/workspace/configs/classification.yaml").kind == "path"
    parsed = parse_train_config_ref("id:11111111-1111-4111-8111-111111111111")
    assert parsed.kind == "id"
    assert parsed.value == "11111111-1111-4111-8111-111111111111"
    with pytest.raises(ConfigError, match="must start with"):
        parse_train_config_ref("classification.yaml")
    with pytest.raises(ConfigError, match="UUID"):
        parse_train_config_ref("id:not-a-uuid")


def test_path_ref_loads_the_yaml_file() -> None:
    settings = ServiceSettings(database_url="postgresql://unused")
    ref = parse_train_config_ref(f"path:{ROOT / 'configs' / 'classification.yaml'}")
    config = resolve_train_config(ref, settings, fetch=lambda *_args: pytest.fail("database was queried"))
    assert config.task == "classification"


def test_id_ref_uses_the_database_document_and_checks_its_id() -> None:
    document = load_config(ROOT / "configs" / "classification.yaml").model_dump(mode="json")
    settings = ServiceSettings(database_url="postgresql://unused")
    ref = parse_train_config_ref(f"id:{document['id']}")
    seen: dict[str, object] = {}

    def fetch(database_url: str, config_id: UUID) -> dict:
        seen["database_url"] = database_url
        seen["config_id"] = config_id
        return document

    config = resolve_train_config(ref, settings, fetch=fetch)
    assert config.id == UUID(document["id"])
    assert seen == {"database_url": settings.database_url, "config_id": config.id}

    document["id"] = "33333333-3333-4333-8333-333333333333"
    with pytest.raises(ConfigError, match="does not match"):
        resolve_train_config(ref, settings, fetch=lambda *_args: document)
