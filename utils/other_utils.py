import math
import os
import random
import shutil
import sys
import numpy as np
import torch
import re
from loguru import logger


def seed_all(seed=1029):
    """
    Sets the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value for the random number generators.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_logger(log_file: str):
    """
    Configure loguru logger to output to both console and file.

    Args:
        log_file (str): Log file path

    Returns:
        logger: Configured loguru logger instance
    """
    # Remove default handler
    logger.remove()

    # Add console handler (colored output)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
        catch=True
    )

    # Add file handler
    logger.add(
        log_file,
        # format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        backtrace=True,
        diagnose=True,
        catch=True,
        enqueue=True
    )

    return logger