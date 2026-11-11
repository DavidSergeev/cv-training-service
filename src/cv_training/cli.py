"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging

from cv_training.builder import build_trainer
from cv_training.config import load_config


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run a configured computer-vision training job.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
