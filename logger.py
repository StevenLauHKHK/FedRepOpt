# --------------------------------------------------------
# Re-parameterizing Your Optimizers rather than Architectures (https://arxiv.org/abs/2205.15242)
# Github source: https://github.com/DingXiaoH/RepOptimizers
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for each of the client
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
