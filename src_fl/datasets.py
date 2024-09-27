import shutil
from PIL import Image
from pathlib import Path
from typing import Callable, Optional, Tuple, Any

import config
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from omegaconf import DictConfig

# from .common import create_lda_partitions
from RepOptimizers.data.build import build_dataset, build_loader



def get_dataset(config, is_train=False):
    samples, classes = build_dataset(is_train, config)
    return samples, classes
    


if __name__ == "__main__":
    config = config.config_from_dict(
        {"DATA": 
            {
            "annotations_fed": "annotations_fed_alpha_0.1_clients_10",
            "DATA_PATH": "/data_ssd/DATA/tiny-imagenet-200",
            "DATASET": "tiny_imagenet",
            "IMG_SIZE": 64,
            "TEST_SIZE":64,
            "INTERPOLATION": "bicubic"
            },
        "TEST": 
            {"CROP": False},
        "FL" : True
        }
    ) 
    config.client_id = str(9)
    samples, classes = get_dataset(config=config)
    print(len(samples), classes)


