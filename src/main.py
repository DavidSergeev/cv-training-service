"""Service entry point."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from builder import build_trainer
from errors import ConfigError
from job import read_train_config_ref, resolve_train_config
from settings import load_service_settings

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_service_settings()
        config = resolve_train_config(read_train_config_ref(), settings)
    except (ConfigError, FileNotFoundError, ValidationError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
