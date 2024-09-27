import argparse
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import os
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import re
import time
import shutil
import time
import shutil
from flwr.common import parameter
import pdb # for debugging
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

def mkdir_or_exist(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)



class Reopt_FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        num_rounds: int = 1,
        eval_every_n:int = 5, 
        base_work_dir: str = "reopt_tiny_imagenet",
        server_logger = None,
        strt_rounds = 0,
        *args, **kwargs):

        self.num_rounds = num_rounds,
        self.eval_every_n = eval_every_n
        self.base_work_dir =base_work_dir
        self.server_logger = server_logger
        self.strt_rounds = strt_rounds
        super().__init__(*args, **kwargs)

    def evaluate(self,
        server_round:int,
        parameters:Parameters):
        """Evaluates global model every N rounds. Last round is always
        considered and flagged as such (e.g. to use global test set)"""
        # from https://github.com/jafermarq/FlowerMonthly/blob/main/src/strategy.py

        is_last_round = server_round == self.num_rounds

        if (server_round % self.eval_every_n == 0) or (server_round == self.num_rounds):
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            loss, metrics = self.evaluate_fn(server_round,
                                            parameters_ndarrays,
                                            config={},
                                            is_last_round=is_last_round)
        
            return loss, metrics
        else:
            print(f"Only evaluating every {self.eval_every_n} rounds...")
            self.server_logger.info(f"Only evaluating every {self.eval_every_n} rounds...")
            return None



    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Divide the weights based on the backbone and classification head
        ######################################################################################################3

        # Aggregate all the weights and the number of examples 
        weight_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        num_examples_total = [num_examples for _, num_examples in weight_results]

        print(f"The number of examples are {num_examples_total} ")

        weights_avg = aggregate(weight_results)  # Fedavg
        weights_avg = ndarrays_to_parameters(weights_avg)
       
        # create a directory to save the global checkpoints
        glb_dir = self.base_work_dir
        mkdir_or_exist(os.path.abspath(glb_dir))
        print("The results are saved")
        self.server_logger.info("The results are saved")

        if weights_avg is not None:
            # save weights
            print(f"round-{server_round + self.strt_rounds}-weights...",)
            np.savez(os.path.join(glb_dir, f"round-{server_round + self.strt_rounds}-weights.array"), weights_avg)
            save_path = os.path.join(glb_dir, f"round-{server_round + self.strt_rounds}-weights.array")
            self.server_logger.info("The result weight saved in " + save_path)

            parameters_ndarrays = parameters_to_ndarrays(weights_avg)
            loss, metrics = self.evaluate_fn(server_round + self.strt_rounds,
                                            parameters_ndarrays,
                                            config={},
                                            is_last_round=False)

        return weights_avg, {}



def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime