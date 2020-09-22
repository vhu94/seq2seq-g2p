import logging
import os
import random
import sys
import torch

from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union


def set_random(seed):
    if seed:
        logging.debug(f'Setting random seed to {seed}')
        random.seed(seed)
        torch.manual_seed(seed)


def setup_logging(log_level: Union[int, str], logfile: Union[str, Path] = None):
    handlers = []

    # console logger
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    handlers.append(stream_handler)

    # file logger
    if logfile:
        os.makedirs(Path(logfile).parent, exist_ok=True)
        file_handler = RotatingFileHandler(logfile, maxBytes=1048576, backupCount=5)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=log_format, level=log_level, handlers=handlers)
