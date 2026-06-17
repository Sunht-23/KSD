#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry script for Distributed Adaptive Repartitioning and GNN Inference Pipeline.
Loads continuous workspace parameters and boots up the atomic client execution loops.
"""

import argparse
import json
import logging
import torch
import torch.nn.functional as F

from Model.DistGAT import DistGAT
from Model.DistGCN import DistGCN
from distribute_infer.distribute_infer import DistInferClient

# Coordinate logger bindings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core functional activation mapping register
ACTIVATION_MAP = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid
}


def load_json_config(graph_name: str, config_file: str = "configs.json") -> dict:
    """Loads dataset configuration layouts from discrete configuration repositories."""
    with open(config_file, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)
    return all_configs[graph_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Graph Neural Network Optimization Engine.")
    parser.add_argument("--graph_name", type=str, help="graph name", default='cora')
    parser.add_argument("--local_rank", type=int, default=0, help="DDP local worker process identification rank")
    parser.add_argument("--world_size", type=int, default=6, help="Cluster worker scale size")
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument('--master_addr', type=str, default='172.30.0.2')
    parser.add_argument('--ddp_port', type=int, default=12113)
    parser.add_argument("--batch_size", type=int, default=20000000)
    parser.add_argument("--ppo_dim", type=int, default=64)
    parser.add_argument("--ppo_num_seed", type=int, default=10)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--model_type", type=str, default='gcn')

    args = parser.parse_args()

    # Dynamic parsing profiles derived from dataset configurations
    config = load_json_config(args.graph_name)
    activation_fn = ACTIVATION_MAP[config.get('activation', 'relu')]

    # Instantiating the underlying targeted model architecture dynamically
    if args.model_type == 'gat':
        logger.info(f"[Rank {args.local_rank}] Initializing DistGAT network topology...")
        custom_model = DistGAT(
            in_feats=config['feat_dim'],
            n_hidden=config['hidden_dim'],
            n_classes=config['n_classes'],
            n_layers=config['num_layers'],
            activation=activation_fn,
            dropout=config['dropout'],
            n_heads=config['n_heads']
        )
    else:
        logger.info(f"[Rank {args.local_rank}] Initializing DistGCN network topology...")
        custom_model = DistGCN(
            in_feats=config['feat_dim'],
            n_hidden=config['hidden_dim'],
            n_classes=config['n_classes'],
            n_layers=config['num_layers'],
            activation=activation_fn,
            dropout=config['dropout']
        )

    # Instantiate the standard optimization pipeline executor client
    client = DistInferClient(model=custom_model, args=args)

    client.init_env()
    client.load_data_and_model()

    # Safely load the pre-trained PPO Actor weights for policy coordination
    client.load_policy_weights()

    try:
        # 🔄 Launch the automated optimization framework loop inside a single pipeline call
        final_representations = client.fit_adaptive_repartition(max_epochs=args.max_epochs)
    finally:
        client.close()