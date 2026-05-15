"""
src/utils/logger.py
───────────────────
Centralised logger using loguru.
Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("hello")
"""

import sys
from loguru import logger
from config.settings import LOG_LEVEL, LOG_FILE


def get_logger(name: str = "app"):
    logger.remove()  # remove default stderr handler

    # Console handler – coloured, human-readable
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> – <level>{message}</level>",
        colorize=True,
    )

    # File handler – full detail, rotating at 10 MB
    logger.add(
        LOG_FILE,
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} – {message}",
    )

    return logger.bind(name=name)
