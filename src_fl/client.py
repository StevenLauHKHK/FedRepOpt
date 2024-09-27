"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
import argparse
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional
import os
import flwr as fl
import numpy as np
from math import exp
import torch
import torch.nn as nn
import re
import ray
import time
import shutil
from flwr.common import parameter
from functools import reduce
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
    ndarrays_to_parameters,   # parameters_to_weights,
    parameters_to_ndarrays,   # weights_to_parameters,
)

from main_repopt_FL import main_repopt
from data.build import build_dataset


class Repopt_Client(fl.client.NumPyClient):
    """Flower client implementing video SSL w/ PyTorch."""

    def __init__(self, model, optimizer, args, cfg, repopt_main, annotations_client_file, logger, c_model_dir_name):       
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.cfg = cfg
        self.repopt_main = repopt_main
        self.annotations_client_file = annotations_client_file
        self.logger = logger
        self.c_model_dir_name = c_model_dir_name

    def get_parameters(self, config):
        # Return local model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        # Update local model w/ global parameters
        self.set_parameters(parameters)

        # perform the training for repog
        self.repopt_main(self.cfg, self.model, self.optimizer, self.logger, self.c_model_dir_name, self.annotations_client_file)

        print(f"*********************The annotations files are loaded here:{self.annotations_client_file}******************")
        # get the number of samples
        samples, classes = build_dataset(True, self.cfg, annotations_client_file=self.annotations_client_file)
        num_examples = len(samples)

        return self.get_parameters(self.model), num_examples, {}

    def evaluate(self, parameters, config):
        # for completion
        result = 0
        return float(0), int(1), {"accuracy": float(result)}





