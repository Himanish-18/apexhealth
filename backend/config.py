"""
Healthcare Knowledge Navigator — Backend Configuration.

Backend-specific configuration and dependency injection helpers
for the FastAPI application.
"""

import logging
from pathlib import Path

from configs.settings import Settings, get_settings


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    """Configure and return the application-wide logger.

    Sets up console and file handlers with a uniform format.
    Log files are written to the project's ``logs/`` directory.

    Args:
        settings: Application settings. Uses the cached singleton if None.

    Returns:
        logging.Logger: Configured root logger for the application.
    """
    if settings is None:
        settings = get_settings()

    # Ensure the logs directory exists
    log_dir: Path = settings.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file: Path = log_dir / "app.log"
    log_level: int = getattr(logging, settings.log_level.upper(), logging.INFO)

    # ── Formatter ────────────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ──────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # ── File Handler ─────────────────────────────────────────────────
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # ── Root Logger ──────────────────────────────────────────────────
    logger = logging.getLogger("hkn")
    logger.setLevel(log_level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a child logger under the ``hkn`` namespace.

    Args:
        name: Descriptive name for the logger (e.g., module name).

    Returns:
        logging.Logger: A child logger instance.
    """
    return logging.getLogger(f"hkn.{name}")
