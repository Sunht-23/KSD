import argparse
import time

import numpy as np
import torch

from env_set import LoadBalanceEnv


def partition_ppo(graph_name, args, node_parts):
    env = LoadBalanceEnv(graph_name, args)

    # env.partition(node_parts)
    # comp_coeffs = torch.tensor([1, 1, 1, 1, 1, 1])
    # comm_coeffs = torch.tensor([1, 1, 1, 1, 1, 1])

    # # 固定系数（直接定义为张量，形状 [6]）
    comp_coeffs = torch.tensor([1.0, 1.0, 1.5, 1.5, 0.5, 0.5])
    comm_coeffs = torch.tensor([1.0, 1.0, 2.0, 2.0, 0.5, 0.5])

    # GCN
    comp_coeffs = comp_coeffs * 5.128
    comm_coeffs = comm_coeffs * 3.09
    # # GAT
    # comp_coeffs = comp_coeffs * 2.013
    # comm_coeffs = comm_coeffs * 0.97

    assign = env.partition_source_change(node_parts, comp_coeffs, comm_coeffs, 10, 'history-cut')

    # final_save_path = f'data/dgl_partition/{graph_name}/{args.num_parts}-parts/{graph_name}{args.num_parts}-ppo-my-b'
    # # final_save_path = f'data/dgl_partition/{graph_name}/{args.num_parts}-parts/{graph_name}{args.num_parts}-ppo-my'
    # np.save(final_save_path, assign)

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser()

    # 添加训练参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)  # 折扣因子，未来value奖励的重要性。
    parser.add_argument('--gae_lambda', type=float, default=0.99)  # GAE参数，平衡偏差和方差。

    parser.add_argument('--episode_length', type=int, default=10000)  # 每轮探索长度
    parser.add_argument('--num_parts', type=int, default=6)

    parser.add_argument('--clip_param', type=float, default=0.2)  # PPO裁剪系数（如0.2）
    parser.add_argument('--ppo_epoch', type=int, default=4)  # 每轮数据的更新次数（相当于6次了）
    parser.add_argument('--num_seed', type=int, default=10)  # 迷你批数量
    parser.add_argument('--num_mini_batch', type=int, default=50)  # 迷你批数量
    parser.add_argument('--value_loss_coef', type=float, default=0.5)  # 价值损失系数
    parser.add_argument('--max_grad_norm', type=float, default=1.0)  # 最大梯度裁剪阈值
    parser.add_argument("--entropy_coef", type=float, default=0.3)  # 熵奖励系数 越大越鼓励
    parser.add_argument("--edge_radio", type=float, default=17)  # ogbn-arxiv GCN 17 GAT 7.5
    parser.add_argument("--num_env_steps", type=int, default=10000)  # 试验次数

    # 解析参数并生成args对象
    args = parser.parse_args()

    dataset_name = 'ogbn-arxiv'
    num_parts = args.num_parts

    data = np.load(f'data/dgl_partition/{dataset_name}/{num_parts}-parts/{dataset_name}{num_parts}-metis.npy')

    if len(data.shape) > 1:
        data = np.argmax(data, axis=1)
        # data = torch.from_numpy(max_indices)
    node_parts = torch.from_numpy(data).long()
    # 训练模型
    t = time.time()
    partition_ppo(dataset_name,args,node_parts)
    print("优化时间为：",time.time()-t)