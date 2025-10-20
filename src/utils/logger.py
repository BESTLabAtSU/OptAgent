"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO):
    """Setup a logger"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger