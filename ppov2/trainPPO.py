import torch
import torch.nn as nn
import tqdm


class TrainPPO:
    """
    PPO Trainer class responsible for performing policy and value function updates
    using clipped objective functions and Generalized Advantage Estimation (GAE).
    """

    def __init__(self, args, policy):
        self.policy = policy

        # Hyperparameters for PPO optimization
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm

    def cal_value_loss(self, values, value_pre_batch, return_batch):
        """
        Calculates the clipped value function loss to prevent the value
        estimate from changing too drastically in one update.
        """
        # Clip the value prediction to stay within the clip range of the previous prediction
        value_pred_clipped = value_pre_batch + (values - value_pre_batch).clamp(-self.clip_param, self.clip_param)

        value_loss_original = (return_batch - values).pow(2)
        value_loss_clipped = (return_batch - value_pred_clipped).pow(2)

        # Take the maximum of original and clipped loss to ensure pessimistic value estimation
        value_loss = torch.max(value_loss_original, value_loss_clipped).mean()

        return value_loss

    def ppo_update(self, sample, graph):
        """
        Performs a single step of PPO optimization including Actor and Critic updates.
        """
        (env_feat, subgraph, vertex_feat, global_env_feat, _,
         _, adv, ret, pre_value, actions, old_action_log_probs) = sample

        if old_action_log_probs is None:
            return

        # Evaluate current actions to get new log probabilities and state values
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            subgraph, vertex_feat, env_feat, graph, global_env_feat, actions
        )

        # Calculate probability ratio via importance sampling
        log_imp_weights = action_log_probs - old_action_log_probs
        imp_weights = torch.exp(log_imp_weights)

        # Calculate Clipped Surrogate Objective for the Actor
        surr1 = imp_weights * adv
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # Actor backpropagation
        self.policy.actor_optimizer.zero_grad()
        actor_loss = policy_loss - dist_entropy * self.entropy_coef
        actor_loss.backward()

        # Gradient clipping to ensure training stability
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.actor_optimizer.step()

        # Critic backpropagation
        value_loss = self.cal_value_loss(values, pre_value, ret)
        critic_loss = value_loss * self.value_loss_coef

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.critic_optimizer.step()

    def train(self, buffer, graph, env_buffer):
        """
        Main training loop that iterates through epochs and mini-batches.
        """
        self.prep_training()

        for _ in range(self.ppo_epoch):
            mini_batch_size = len(buffer.actions)
            data_generator = buffer.feed_forward_generator(env_buffer, mini_batch_size)

            sample_bar = tqdm.tqdm(
                data_generator,
                total=mini_batch_size,
                desc="Updating Samples",
                leave=False
            )

            for sample in sample_bar:
                self.ppo_update(sample, graph)

        return

    def prep_training(self):
        """Sets the underlying modules to training mode."""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """Sets the underlying modules to evaluation mode for interaction."""
        self.policy.actor.eval()
        self.policy.critic.eval()