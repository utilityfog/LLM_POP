import logging
import sys
import os
from uuid import uuid4

from indra.pytorch.util import (
    create_folder,
)


def get_logger(verbose: bool = False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    if verbose:
        pid = os.getpid()
        create_folder(path="libdeeplake_logs")
        name = f"libdeeplake_logs/log_{pid}_{str(uuid4())[:4]}.log"
        file_handler = logging.FileHandler(name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.addHandler(stdout_handler)
    return logger
