"""Configuration centralisée du logging (console + fichier rotatif)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_LOGGER_NAME = "market_sentinel"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: str = "INFO",
    file: Optional[str] = None,
    max_bytes: int = 5_242_880,
    backup_count: int = 5,
) -> logging.Logger:
    """Initialise le logger racine de l'application (idempotent)."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:  # déjà configuré : on ne duplique pas les handlers
        return logger

    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if file:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retourne un logger enfant du logger applicatif."""
    if name and name != _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME).getChild(name.split(".")[-1])
    return logging.getLogger(_LOGGER_NAME)
