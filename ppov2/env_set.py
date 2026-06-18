import os
import random
import time
import dgl
import numpy as np
import torch
import tqdm

from ppo import PPO
from trainPPO import TrainPPO
from replayBuffer import ReplayBuffer
from ppo_util import get_feat, get_load, exchange_part, partition_graph_metis_no_feat, update_action


class LoadBalanceEnv:
    def __init__(self, graph_name, args):
        self.num_parts = args.num_parts
        self.graph_name = graph_name
        self.ori_graph = None
        self.graph = args.graph

        self.base_comp_v, self.base_comp_e, self.base_comp_t = args.base_comp_v, args.base_comp_e, args.base_comp_t
        self.base_comm = args.base_comm
        self.node_parts = None

        self.num_env_steps = args.num_env_steps
        self.episode_length = args.episode_length
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda

        self.num_mini_batch = args.num_mini_batch
        self.policy = PPO(args.hidden_dim, args.lr, args.num_parts, args.num_seed)
        self.trainer = TrainPPO(args, self.policy)
        self.load()

        self.env_buffer = []
        self.stop = False
        self.not_change_step = 0
        self.init_time = 10e8
        self.computer_v = []
        self.computer_e = []
        self.transfer_list = [] #

        self.buffer = ReplayBuffer(
            self.episode_length,
            self.gamma,
            self.gae_lambda,
            self.num_parts
        )

    @torch.no_grad()
    def insert(self, reward, value, actions, action_log_probs):
        self.buffer.insert(reward, value, actions, action_log_probs)

    def run(self):
        for episode in tqdm.tqdm(range(self.num_env_steps)):
            self.reset(episode)

            for ep in range(self.episode_length // self.num_mini_batch):
                if self.stop:
                    break
                self.get_env()
                for step in range(self.num_mini_batch):
                    realstep = ep * self.num_mini_batch + step

                    value, action, action_log_prob = self.collect(step, episode)

                    up_action = update_action(action, self.buffer.graph_list[step], self.buffer.node_dict_list[step],
                                              self.buffer.max_index_list[step], self.num_parts)

                    reward = self.env_step(up_action, step, realstep)

                    self.insert(reward, value, action, action_log_prob)

                    if self.stop:
                        print("early stop,step: {}, episode: {}".format(step, episode))
                        break

                self.compute()
                self.trainer.train(self.buffer, self.graph, self.env_buffer)

                del self.buffer
                self.buffer = ReplayBuffer(
                    self.episode_length,
                    self.gamma,
                    self.gae_lambda,
                    self.num_parts
                )
                del self.env_buffer
                self.env_buffer = []

                self.save()
                print(f"Episode {episode}, model saved")

    def partition_source_change(self, node_parts, comp_coeffs=None, comm_coeffs=None, early_stop_step=10,
                                save_name='history'):
        self.ori_graph = None
        self.node_parts = node_parts

        self.computer_v = torch.as_tensor((self.base_comp_v * comp_coeffs), dtype=torch.float32)
        self.computer_e = torch.as_tensor((self.base_comp_e * comp_coeffs), dtype=torch.float32)
        self.transfer_list = torch.as_tensor((self.base_comm * comm_coeffs), dtype=torch.float32)

        self.get_env()

        best = self.buffer.true_total_max_list[0]

        best_step = 0
        load_history = []
        seed = self.trainer.policy.num_seed
        final_node_parts = self.node_parts.clone()

        eval_bar = tqdm.tqdm(range(self.episode_length), desc="Repartition")

        for eval_step in eval_bar:
            self.trainer.prep_rollout()

            comp_old_max = self.buffer.true_comp_max_list[0]
            comm_old_max = self.buffer.true_comm_max_list[0]
            E2E_latency_old = self.buffer.true_total_max_list[0]
            local_rank = self.buffer.max_index_list[0]

            load_history.append((comp_old_max, comm_old_max))

            with torch.inference_mode():
                eval_action = self.trainer.policy.act(
                    self.buffer.graph_list[0],
                    self.buffer.max_index_list[0],
                    self.buffer.node_dict_list[0],
                    self.buffer.vertex_feat_list[0],
                    self.buffer.env_feat_list[0],
                )
            up_action = update_action(eval_action, self.buffer.graph_list[0], self.buffer.node_dict_list[0],
                                      self.buffer.max_index_list[0], self.num_parts)

            exchange_part(self.graph, self.graph_name, self.num_parts, up_action, local_rank,
                          self.buffer.node_dict_list[0], self.node_parts)
            self.buffer.after_update()

            self.get_env()

            E2E_latency_new = self.buffer.true_total_max_list[0]

            reward = E2E_latency_old - E2E_latency_new

            if E2E_latency_new < best:
                self.trainer.policy.num_seed = seed
                best = E2E_latency_new
                final_node_parts = self.node_parts.clone()
                self.not_change_step = 0
                best_step = eval_step + 1
            else:
                self.not_change_step += 1
                if self.not_change_step >= early_stop_step:
                    eval_bar.write(
                        f"Early stop at step {eval_step}: No improvement for {early_stop_step} consecutive steps.")
                    break

            # Use tqdm.write to preserve progress bar layout while tracking step metrics
            eval_bar.write(
                f"Step {eval_step} | Reward: {reward:.4f} | Best Latency: {best:.4f} | Current Latency: {E2E_latency_new:.4f}"
            )

        assign = final_node_parts.numpy()
        # Resolve the destination directory path dynamically
        save_dir = f'data/dgl_partition/{self.graph_name}/{self.num_parts}-parts'

        # Create the target path directories if they do not exist on disk
        os.makedirs(save_dir, exist_ok=True)

        # Save the execution trajectory array safely to disk
        np.save(
            os.path.join(save_dir, f'{self.graph_name}{self.num_parts}-ppo-{save_name}'),
            np.array(load_history[:best_step + 1])
        )
        return assign

    def save(self):
        # Define the directory path for saving the models
        save_dir = f"data/ppo"
        os.makedirs(save_dir, exist_ok=True)

        # Save the Actor model
        policy_actor = self.trainer.policy.actor
        torch.save(
            policy_actor.state_dict(),
            f"data/ppo/{self.graph_name}_{self.num_parts}_actor.pt",
        )

        # Save the Critic model
        policy_critic = self.trainer.policy.critic
        torch.save(
            policy_critic.state_dict(),
            f"data/ppo/{self.graph_name}_{self.num_parts}_critic.pt",
        )

    def load(self):
        policy_actor = self.trainer.policy.actor
        policy_critic = self.trainer.policy.critic

        model_path = f"data/ppo/{self.graph_name}_{self.num_parts}_critic.pt"
        if os.path.exists(model_path):
            try:
                policy_critic.load_state_dict(torch.load(model_path))
                self.trainer.policy.critic = policy_critic
                print(f"Successfully loaded critic from {model_path}")
            except Exception as e:
                print(f"Failed to load critic from {model_path}: {str(e)}. Using new critic.")
        else:
            print(f"No critic model found at {model_path}, using new critic.")

        model_path = f"data/ppo/{self.graph_name}_{self.num_parts}_actor.pt"
        if os.path.exists(model_path):
            try:
                policy_actor.load_state_dict(torch.load(model_path))
                self.trainer.policy.actor = policy_actor
                print(f"Successfully loaded actor from {model_path}")
            except Exception as e:
                print(f"Failed to load actor from {model_path}: {str(e)}. Using new actor.")
        else:
            print(f"No actor model found at {model_path}, using new actor.")
            try:
                model_path = f"data/ppo/ogbn-arxiv_{self.num_parts}_actor.pt"
                policy_actor.load_state_dict(torch.load(model_path))
                self.trainer.policy.actor = policy_actor
                print(f"Successfully loaded actor from {model_path}")
            except Exception as e:
                print(f"Failed to load actor from {model_path}: {str(e)}. Using new actor.")


    def reset(self, epoch):
        del self.buffer
        self.buffer = ReplayBuffer(
            self.episode_length,
            self.gamma,
            self.gae_lambda,
            self.num_parts
        )

        del self.env_buffer
        self.env_buffer = []

        self.stop = False
        self.init_time = 10e8
        self.not_change_step = 0

        self.ori_graph = self.graph
        n = self.ori_graph.num_nodes()
        random_seq = random.sample(range(n), n)
        del self.graph
        self.graph = dgl.reorder_graph(self.ori_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_seq},
                                       store_ids=True)
        del self.node_parts
        self.node_parts, _ = partition_graph_metis_no_feat(self.graph, self.graph_name, self.num_parts, 'data/train')

        if epoch == 0:
            self.computer_v = torch.as_tensor(
                [int(self.base_comp_v * 1) for x in range(self.num_parts)],
                dtype=torch.float32)
            self.computer_e = torch.as_tensor(
                [int(self.base_comp_e * 1) for x in range(self.num_parts)],
                dtype=torch.float32)
            self.transfer_list = torch.as_tensor(
                [int(self.base_comm * 1) for x in range(self.num_parts)],
                dtype=torch.float32)

        elif epoch % 2 == 0:
            self.computer_v = torch.as_tensor(
                [int(self.base_comp_v * np.random.uniform(0.4, 1.6)) for x in range(self.num_parts)],
                dtype=torch.float32)
            self.computer_e = torch.as_tensor(
                [int(self.base_comp_e * np.random.uniform(0.4, 1.6)) for x in range(self.num_parts)],
                dtype=torch.float32)
            self.transfer_list = torch.as_tensor(
                [int(self.base_comm * np.random.uniform(0.4, 2.0)) for x in range(self.num_parts)],
                dtype=torch.float32)

    def get_env(self):
        # get all sub-graph statistics
        send_num_list, recv_num_list, node_num_list, edge_num_list, parts, node_dict_list = get_load(self.graph, self.node_parts, self.num_parts)

        # current vertex-to-device mapping as one-hot encoding
        all_vertex_feat = torch.nn.functional.one_hot(self.node_parts, num_classes=self.num_parts).float()

        # communication and computation load for each partition
        send_tensor = torch.tensor(send_num_list, dtype=torch.float32)
        recv_tensor = torch.tensor(recv_num_list, dtype=torch.float32)

        # communication time estimation
        send_time = send_tensor / self.transfer_list
        recv_time = recv_tensor / self.transfer_list
        comm_time = torch.max(recv_time, send_time)

        # computation time estimation
        node_tensor = torch.tensor(node_num_list, dtype=torch.float32)
        edge_tensor = torch.tensor(edge_num_list, dtype=torch.float32)
        comp_time = node_tensor / self.computer_v + edge_tensor / self.computer_e + self.base_comp_t

        self.buffer.true_comp_max_list.append(torch.max(comp_time).item())
        self.buffer.true_comm_max_list.append(torch.max(comm_time).item())
        self.buffer.true_total_max_list.append(torch.max(comp_time + comm_time).item())
        self.buffer.true_mean_list.append(torch.mean(comp_time + comm_time).item())

        # communication and computation time normalization
        eps = 1e-6
        total_time = comp_time + comm_time
        comp_time_norm = comp_time / (total_time + eps)
        comm_time_norm = comm_time / (total_time + eps)

        all_env_feat = torch.cat([
            node_tensor,
            edge_tensor,
            comp_time_norm,
            send_tensor,
            recv_tensor,
            comm_time_norm
        ], dim=0)
        all_time = max(comm_time) + max(comp_time)

        if all_time < self.init_time:
            self.init_time = all_time

        # bottleneck device
        total_part_time = comp_time + comm_time
        max_index = torch.argmax(total_part_time, dim=0).item()
        sub_graph = parts[max_index]

        # gc
        parts.clear()
        del parts
        node_dict = {}
        if 'ndata' in dir(sub_graph):
            for key in list(sub_graph.ndata.keys()):
                node_dict[key] = sub_graph.ndata.pop(key)
        if 'edata' in dir(sub_graph):
            for key in list(sub_graph.edata.keys()):
                sub_graph.edata.pop(key)
        node_dict_list.clear()
        del node_dict_list

        env_feat, vertex_feat = get_feat(all_env_feat, sub_graph, node_dict, max_index, self.num_parts)

        self.buffer.env_feat_list.append(env_feat)
        self.buffer.vertex_feat_list.append(vertex_feat)
        self.buffer.graph_list.append(sub_graph)
        self.buffer.max_index_list.append(max_index)
        self.buffer.node_dict_list.append(node_dict)
        self.env_buffer.append((all_vertex_feat, all_env_feat))

    @torch.no_grad()
    def collect(self, step, episode):
        self.trainer.prep_rollout()
        action, action_log_prob = self.trainer.policy.get_actions(
            self.buffer.graph_list[step],
            self.buffer.max_index_list[step],
            self.buffer.node_dict_list[step],
            self.buffer.vertex_feat_list[step],
            self.buffer.env_feat_list[step]
        )

        value = self.trainer.policy.get_values(self.graph, self.env_buffer[step])
        return (
            value,
            action,
            action_log_prob
        )



    @torch.no_grad()
    def env_step(self, action, step, real_step):
        if len(action) == 2:
            print("No migration action")
            self.stop = True
            return 0

        all_load_old = self.buffer.true_total_max_list[step]
        local_rank = self.buffer.max_index_list[step]

        exchange_part(self.graph, self.graph_name, self.num_parts, action, local_rank, self.buffer.node_dict_list[step], self.node_parts)

        self.get_env()

        comp_new_max = self.buffer.true_comp_max_list[step + 1]
        comm_new_max = self.buffer.true_comm_max_list[step + 1]
        all_load_new = self.buffer.true_total_max_list[step + 1]
        mean_new = self.buffer.true_mean_list[step + 1]

        reward = all_load_old - all_load_new

        if all_load_new / mean_new < 1.15:
            print("Performance close to optimal. Stopping current episode.")
            self.stop = True

            # 7. 每一步完美打印出真实的计算、通信、总延迟和当前步获得的物理 Reward
        print(
            "Step: {:4d} | Reward: {:+.6f}s | Compute: {:.4f}s | Comm: {:.4f}s | Total_E2E: {:.4f}s".format(
                real_step, reward, comp_new_max, comm_new_max, all_load_new
            ))
        return reward

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_value = self.trainer.policy.get_values(
            self.graph,
            self.env_buffer[-1]
        )
        self.buffer.compute_rets(next_value)
