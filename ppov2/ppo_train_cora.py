import argparse
import dgl

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
    parser.add_argument('--episode_length', type=int, default=2700)  # Steps per exploration episode
    parser.add_argument('--num_parts', type=int, default=6)  # Number of graph partitions
    parser.add_argument('--num_env_steps', type=int, default=100)  # Total number of environment trials

    # PPO specific optimization settings
    parser.add_argument('--clip_param', type=float, default=1.0)  # PPO clipping epsilon
    parser.add_argument('--ppo_epoch', type=int, default=2)  # Number of updates per data batch
    parser.add_argument('--num_seed', type=int, default=1)  # Number of seed nodes for sampling
    parser.add_argument('--num_mini_batch', type=int, default=2700)  #
    parser.add_argument('--value_loss_coef', type=float, default=0.4)  # Weight for the value loss
    parser.add_argument('--max_grad_norm', type=float, default=2.0)  # Threshold for gradient clipping
    parser.add_argument("--entropy_coef", type=float, default=0.2)  # Coefficient for exploration bonus

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

    # Load default dataset (Cora) for training
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    args.graph = g

    # Execute training
    train_ppo('cora', args)