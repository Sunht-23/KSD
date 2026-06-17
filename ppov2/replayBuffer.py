import torch

class ReplayBuffer:
    def __init__(self, episode_length, gamma, gae_lambda, num_parts):
        self.episode_length = episode_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.step = 0
        self.env_feat_list = []
        self.graph_list = []
        self.max_index_list = []
        self.vertex_feat_list = []
        self.node_dict_list = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = torch.zeros(self.episode_length, dtype=torch.float32)
        self.pre_value = torch.zeros(self.episode_length + 1, dtype=torch.float32)
        self.rets = torch.zeros(self.episode_length + 1, dtype=torch.float32)

        self.true_comp_max_list = []
        self.true_comm_max_list = []
        self.true_total_max_list = []
        self.true_mean_list = []

    def compute_rets(self, next_value):
        """
        Computes returns and Generalized Advantage Estimation (GAE) targets.
        """
        self.pre_value[len(self.actions)] = next_value
        gae = 0.0
        for step in reversed(range(len(self.actions))):
            next_value = self.pre_value[step + 1] if (step + 1 < len(self.pre_value)) else 0.0
            delta = self.rewards[step] + self.gamma * next_value - self.pre_value[step]
            gae = delta + self.gamma * self.gae_lambda * gae

            self.rets[step] = gae + self.pre_value[step]

    def insert(self, reward, value, actions, action_log_probs):
        """
        Inserts new trajectory elements into internal trackers.
        """
        self.pre_value[self.step] = value
        self.rewards[self.step] = reward
        self.actions.append(actions)
        self.action_log_probs.append(action_log_probs)

        self.step = (self.step + 1) % self.episode_length

    def feed_forward_generator(self, env_buffer, mini_batch_size):
        """
        Generates mini-batches of normalized advantages and trajectory states for training.
        """
        advantages = self.rets - self.pre_value
        advantages = advantages[:len(self.actions)]  # Truncate to the actual number of actions
        if advantages.numel() > 1:
            mean_advantages = advantages.mean()
            std_advantages = advantages.std() + 1e-5
            advantages = (advantages - mean_advantages) / std_advantages

        rand = torch.randperm(mini_batch_size)

        for indices in rand:
            env_feat = self.env_feat_list[indices]
            subgraph = self.graph_list[indices]
            vertex_feat = self.vertex_feat_list[indices]
            node_dict = self.node_dict_list[indices]

            reward = self.rewards[indices]
            adv = advantages[indices]
            ret = self.rets[indices]
            pre_value = self.pre_value[indices]
            action = self.actions[indices]
            action_log_prob = self.action_log_probs[indices]
            global_env_feat = env_buffer[indices]

            yield env_feat, subgraph, vertex_feat, global_env_feat, node_dict, reward, adv, ret, pre_value, action, action_log_prob

    def after_update(self):
        """
        Clears list caches and resets metric arrays after an optimization pass.
        """
        self.env_feat_list.clear()
        self.graph_list.clear()
        self.max_index_list.clear()
        self.vertex_feat_list.clear()
        self.node_dict_list.clear()
        self.actions.clear()
        self.action_log_probs.clear()

        self.rewards = torch.zeros(self.episode_length, dtype=torch.float32)
        self.pre_value = torch.zeros(self.episode_length + 1, dtype=torch.float32)
        self.rets = torch.zeros(self.episode_length + 1, dtype=torch.float32)

        self.true_comp_max_list.clear()
        self.true_comm_max_list.clear()
        self.true_total_max_list.clear()
        self.true_mean_list.clear()

        self.step = 0