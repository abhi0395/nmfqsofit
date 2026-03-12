"""
Logging configuration for nmfqsofit package.

Provides centralized logging setup with consistent formatting and levels.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "nmfqsofit",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        name (str): Logger name (default: "nmfqsofit").
        level (int): Logging level (default: logging.INFO).
        log_file (str, optional): If provided, log to this file in addition to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        return logger

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s]:%(levelname)s: %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger by name (must be called after setup_logger).

    Args:
        name (str): Logger name.

    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name)
