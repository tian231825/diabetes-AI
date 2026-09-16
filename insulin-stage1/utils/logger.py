# -*- encoding: utf-8 -*-
import logging
import torch
import numpy as np
import os
import random

def setup_logging(log_file='app.log', console_output=True, log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    np.set_printoptions(threshold=np.inf, linewidth=200)
    try:
        torch.set_printoptions(threshold=10000, linewidth=200)
    except ImportError:
        pass
    return logger

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.getLogger(__name__).info(f"Random seed set to: {seed}")
