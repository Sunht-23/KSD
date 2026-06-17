import argparse
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

from env_set import LoadBalanceEnv


def train_ppo(graph_name, args):
    """
    Main training function for the Multi-Agent PPO load balancer.
    """
    # Initialize the load balancing environment
    env = LoadBalanceEnv(graph_name, args)

    # Start the training process
    env.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training for Graph Partitioning Load Balance")

    # Hyperparameters for the PPO algorithm
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)  # Discount factor for future rewards
    parser.add_argument('--gae_lambda', type=float, default=0.99)  # GAE parameter to balance bias and variance

    # Training and environment configuration
    parser.add_argument('--episode_length', type=int, default=16000)  # Steps per exploration episode
    parser.add_argument('--num_parts', type=int, default=2)  # Number of graph partitions
    parser.add_argument('--num_env_steps', type=int, default=100)  # Total number of environment trials

    # PPO specific optimization settings
    parser.add_argument('--clip_param', type=float, default=1.0)  # PPO clipping epsilon
    parser.add_argument('--ppo_epoch', type=int, default=2)  # Number of updates per data batch
    parser.add_argument('--num_seed', type=int, default=1)  # Number of seed nodes for sampling
    parser.add_argument('--num_mini_batch', type=int, default=160)  #
    parser.add_argument('--value_loss_coef', type=float, default=0.4)  # Weight for the value loss
    parser.add_argument('--max_grad_norm', type=float, default=2.0)  # Threshold for gradient clipping
    parser.add_argument("--entropy_coef", type=float, default=0.2)  # Coefficient for exploration bonus

    # Load balancing physical cost constants
    # parser.add_argument("--base_comp_v", type=float, default=210000)
    # parser.add_argument("--base_comp_e", type=float, default=3500000)
    # parser.add_argument("--base_comp_t", type=float, default=0)
    # parser.add_argument("--base_comm", type=float, default=28000)

    parser.add_argument("--base_comp_v", type=float, default=79000)
    parser.add_argument("--base_comp_e", type=float, default=600000)
    parser.add_argument("--base_comp_t", type=float, default=0)
    parser.add_argument("--base_comm", type=float, default=9700)

    args = parser.parse_args()

    # Load default dataset (Cora) for training
    data = DglNodePropPredDataset(name='ogbn-arxiv', root='data')
    g, _ = data[0]
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    args.graph = g

    # Execute training
    train_ppo('ogbn-arxiv', args)