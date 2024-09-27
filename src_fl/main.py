import argparse
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional
import os
import sys
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
    ndarrays_to_parameters,   # parameters_to_weight in the original FedVSSL repository
    parameters_to_ndarrays,   # weights_to_parameters in the original FedVSSL repository
)

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
rootpath = os.path.join(os.getcwd())
print('rootpath = ', rootpath)
sys.path.append(rootpath)
from config_FL import get_config
from server import get_evaluate_fn

from client import Repopt_Client
from strategy import Reopt_FedAvg

from logger import create_logger



def parse_option():
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--arch', default=None, type=str, help='arch name')
    parser.add_argument('--batch-size', default=128, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='/path/to/cf100/', type=str, help='path to dataset')
    parser.add_argument('--scales-path', default=None, type=str, help='path to the trained Hyper-Search model')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],  #TODO Note: use amp if you have it
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')


     ### hyper-parameters for FL pre-training ###
    parser.add_argument('--exp_name', default='tiny_repopt_results', type=str, help='experimental name used for SSL pre-training.')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    # FL settings
    parser.add_argument('--pool_size', default=10, type=int, help='number of dataset partitions (= number of total clients).')
    parser.add_argument('--rounds', default=50, type=int, help='number of FL rounds.')
    parser.add_argument('--num_clients_per_round', default=10, type=int, help='number of clients participating in the training.')

     # ray config
    parser.add_argument('--cpus_per_client', default=2, type=int, help='number of CPUs used for each client.')
    parser.add_argument('--gpus_per_client', default=1, type=int, help='number of GPUs used for each client.')
    parser.add_argument('--include_dashboard', default=False, type=bool, help='number of GPUs used for each client.')


    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def initial_setup(cid, args, config):

    from main_repopt_FL import main_repopt
    from build_model import build_model

    cid_plus_one = str(int(cid) + 1) # The client number 
    
    annotations_client_file =  'client_dist' + cid_plus_one + '.json'
    print(f"################### annotation file is {annotations_client_file}###############")
    c_model_dir_name = 'client_' + cid_plus_one
    log_output = os.path.join(config.OUTPUT, c_model_dir_name)
    os.makedirs(log_output, exist_ok=True)
    logger_file = create_logger(output_dir=log_output, dist_rank=int(cid_plus_one), name=f"{config.MODEL.ARCH}")
    
    model, optimizer = build_model(config)

    return model, optimizer, args, cfg, main_repopt, annotations_client_file, logger_file, c_model_dir_name

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "epoch_global": str(rnd),
    }
    return config

if __name__ == "__main__":
    args, cfg = parse_option()

    base_work_dir = os.path.join(cfg.OUTPUT,'server_model')
    rounds = args.rounds
    num_gpus = args.gpus_per_client

    seed = cfg.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)


    client_resources = {"num_cpus": args.cpus_per_client, "num_gpus": args.gpus_per_client}
    
    server_log_output = os.path.join(cfg.OUTPUT, 'server')
    os.makedirs(server_log_output, exist_ok=True)
    server_logger_file = create_logger(output_dir=server_log_output, dist_rank=0, name=f"{cfg.MODEL.ARCH}")

    if args.resume:
        model_ckpt_weight = np.load(args.resume, allow_pickle=True)
        model_ckpt_weight = model_ckpt_weight['arr_0'].item()
        initial_parameters = model_ckpt_weight
        strt_rounds = int(args.resume.split('/')[-1].split('-')[1])
        rounds = args.rounds - strt_rounds
    else:
        initial_parameters = None
        strt_rounds = 0
        rounds = args.rounds

    def client_fn(cid: str):
        model, optimizer, client_args, client_cfg, main_repopt, annotations_client_file, logger_file, c_model_dir_name = initial_setup(cid, args, cfg)
        return Repopt_Client(model, optimizer, client_args, client_cfg, main_repopt, annotations_client_file, logger_file, c_model_dir_name)

    strategy = Reopt_FedAvg(
        base_work_dir = base_work_dir,
        fraction_fit = (float(args.num_clients_per_round) / args.pool_size),
        min_fit_clients=args.num_clients_per_round,
        min_available_clients=args.pool_size,
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(cfg, server_logger_file),
        server_logger=server_logger_file,
        initial_parameters=initial_parameters,
        strt_rounds=strt_rounds
    )

    #  (optional) specify ray config
    ray_config = {"include_dashboard": args.include_dashboard}

    # start simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        ray_init_args=ray_config,
    )

     


