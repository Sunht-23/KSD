import argparse
import time
import dgl
import numpy as np
import torch
from env_set import LoadBalanceEnv
from ppov2.ppo_util import partition_graph_metis_no_feat


def partition_ppo(graph_name, args, node_parts):
    """
    Main function for Multi-Agent Partitioning using PPO.
    """
    # Initialize the load balancing environment
    env = LoadBalanceEnv(graph_name, args)

    # # Capability coefficients for different partitions
    comp_coeffs = torch.tensor([1.0, 1.0, 1.5, 1.5, 0.5, 0.5])
    comm_coeffs = torch.tensor([1.0, 1.0, 2.0, 2.0, 0.5, 0.5])

    # comp_coeffs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # comm_coeffs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Run the partition optimization using the PPO-GAT strategy
    assign = env.partition_source_change(
        node_parts,
        comp_coeffs,
        comm_coeffs,
        10,
        'history-ppo-gat'
    )

    # Save the resulting partition assignment
    save_dir = f'data/dgl_partition/{graph_name}/{args.num_parts}-parts'
    final_save_path = f'{save_dir}/{graph_name}{args.num_parts}-ppo-gat'
    np.save(final_save_path, assign)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO-based Graph Partitioning for Load Balancing")

    # Model and Learning Hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)  # Discount factor for future rewards
    parser.add_argument('--gae_lambda', type=float, default=0.99)  # GAE parameter for bias-variance tradeoff

    # Training and Environment Settings
    parser.add_argument('--episode_length', type=int, default=10000)  # Steps per exploration episode
    parser.add_argument('--num_parts', type=int, default=6)  # Number of target partitions
    parser.add_argument('--num_env_steps', type=int, default=10000)  # Total number of trials

    # PPO Specific Parameters
    parser.add_argument('--clip_param', type=float, default=0.2)  # PPO clipping epsilon
    parser.add_argument('--ppo_epoch', type=int, default=4)  # Optimization epochs per data batch
    parser.add_argument('--num_seed', type=int, default=10)  # Number of seed nodes to process
    parser.add_argument('--num_mini_batch', type=int, default=5)  # Mini-batch count
    parser.add_argument('--value_loss_coef', type=float, default=0.5)  # Weight for value function loss
    parser.add_argument('--max_grad_norm', type=float, default=1.0)  # Gradient clipping threshold
    parser.add_argument("--entropy_coef", type=float, default=0.3)  # Coefficient for entropy exploration bonus

    # GCN
    parser.add_argument("--base_comp_v", type=float, default=82000)
    parser.add_argument("--base_comp_e", type=float, default=1968000)
    parser.add_argument("--base_comp_t", type=float, default=0.0)
    parser.add_argument("--base_comm", type=float, default=7150)  # Communication cost coefficient

    # GAT
    # parser.add_argument("--base_comp_v", type=float, default=35200)
    # parser.add_argument("--base_comp_e", type=float, default=317080)
    # parser.add_argument("--base_comp_t", type=float, default=0.0)
    # parser.add_argument("--base_comm", type=float, default=4800)  # Communication cost coefficient

    args = parser.parse_args()

    # Load Dataset (Default: Cora)
    dataset_name = 'cora'
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    args.graph = g



    node_parts,_ = partition_graph_metis_no_feat(g, 'cora', args.num_parts, f'data/dgl_partition/{dataset_name}')

    # Execute Optimization
    start_time = time.time()
    partition_ppo(dataset_name, args, node_parts)

    print(f"Optimization finished in: {time.time() - start_time:.2f} seconds")