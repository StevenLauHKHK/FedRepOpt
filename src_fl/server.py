from typing import Dict, Callable, Optional, Tuple, List
import numpy as np
from collections import OrderedDict
import torch
from omegaconf import DictConfig
from flwr.common.typing import Scalar, NDArrays
import os
from build_model import build_model
from data.build import build_dataset
from main_repopt_FL import validate
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



def ndarrays_to_model(model: torch.nn.ModuleList, params: List[np.ndarray], round):
    # from https://github.com/jafermarq/FlowerMonthly/blob/main/src/model_utils.py
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model 

def get_evaluate_fn(
    model_cfg: DictConfig,
    logger
) -> Callable[[NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, 
        parameters:NDArrays, 
        config: Dict[str, Scalar], 
        is_last_round: bool=False
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # First build the model
        model, _  = build_model(model_cfg)
        # Now set the model buffers with the parameters of the global model
        model = ndarrays_to_model(model, parameters, server_round)
        model.to(device)

        # here you could use the config to parameterise how the global evaluation is performed (e.g. use a particular bach size)
        # you could also use the `is_last_round` flag to switch between a global validation set and a global test set.
        # The global test set should be used only in the last round, while the global validation set can be used in all rounds.
        print(f"Is this the last round?: {is_last_round = }")

        dataset_val, _ = build_dataset(is_train=False, config=model_cfg)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, 
            batch_size=model_cfg.DATA.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=model_cfg.DATA.NUM_WORKERS,
            pin_memory=model_cfg.DATA.PIN_MEMORY,
            drop_last=False
        )

        # testloader = torch.utils.data.DataLoader(dataset_val, batch_size=128)

        # run global evaluation
        acc1, acc5, loss = validate(model_cfg,  data_loader_val, model, logger)

        # Now you have evaluated the global model. This is the a good place to save a checkpoint if, for instance, a new
        # best global model is found (based on a global validation set).
        # If for instance you are using tensorboard to record global metrics or W&B (even better!!) this is the a good
        # place to log all the metrics you want.

        # return statistics
        return loss, {"accuracy": acc1}

    return evaluate   