import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, GlobalAttentionPooling
from torch.distributions import Categorical
from ppov2.ppo_util import get_bound_nodes


class PPO(nn.Module):
    def __init__(self, hidden_dim, lr, num_parts, num_seed):
        super(PPO, self).__init__()
        self.lr = lr
        self.num_parts = num_parts
        self.num_seed = num_seed

        self.actor = R_AC(hidden_dim, num_parts)
        self.critic = R_C(hidden_dim, num_parts)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.selected_vertices = set()

    def reset_selected_vertices(self):
        self.selected_vertices.clear()

    def add_selected_vertices(self, vertices):
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if hasattr(vertices, 'tolist'):
            self.selected_vertices.update([int(v) for v in vertices.tolist()])
        else:
            self.selected_vertices.update([int(v) for v in vertices])

    def get_actions(self, sub_graph, local_rank, node_dict, vertex_feat, env_feat):
        seed = self.get_seed(sub_graph, local_rank, node_dict, vertex_feat, env_feat, self.num_seed)
        if len(seed) == 0:
            return [seed, torch.tensor([], device=vertex_feat.device)], None

        self.add_selected_vertices(seed)
        actions, action_log_probs = self.actor(sub_graph, vertex_feat, env_feat, seed)
        return [seed, actions], action_log_probs

    def get_values(self, graph, global_env_feat):
        return self.critic(graph, global_env_feat)

    def evaluate_actions(self, sub_graph, vertex_feat, env_feat, graph, global_env_feat, actions):
        seed = actions[0]
        action_log_probs, dist_entropy = self.actor(
            sub_graph, vertex_feat, env_feat, seed, isTrain=True, isEval=True, actions=actions
        )
        value = self.critic(graph, global_env_feat)
        return value, action_log_probs, dist_entropy

    def act(self, sub_graph, local_rank, node_dict, vertex_feat, env_feat):
        boundary_nodes = get_bound_nodes(sub_graph, local_rank, node_dict)
        num_seed = max(1, len(boundary_nodes) * self.num_seed // 100)

        seed = self.get_seed(sub_graph, local_rank, node_dict, vertex_feat, env_feat, num_seed)
        actions, _ = self.actor(sub_graph, vertex_feat, env_feat, seed, isTrain=False)
        return [seed, actions]

    @torch.no_grad()
    def get_seed(self, sub_graph, local_rank, node_dict, vertex_feat, env_feat, num_seed=1):
        """ KEYSTONE selection """
        device = vertex_feat.device
        bound_node = get_bound_nodes(sub_graph, local_rank, node_dict)
        bound_vertex_feat = vertex_feat[bound_node, :]

        send_parts = bound_vertex_feat[:, self.num_parts * 4]
        inner_counts = bound_vertex_feat[:, :self.num_parts]
        halo_all = bound_vertex_feat[:, self.num_parts * 2: self.num_parts * 3]
        halo_counts = bound_vertex_feat[:, self.num_parts * 3: self.num_parts * 4]

        send_local = env_feat[self.num_parts * 4 + local_rank]
        recv_local = env_feat[self.num_parts * 5 + local_rank]

        # Score communication balancing attributes based on local transmission loads
        if send_local > recv_local:
            diffs = inner_counts - send_parts.unsqueeze(1)
            cols = torch.arange(inner_counts.size(1), device=device)
            valid_cols = cols[cols != local_rank]
            filtered_diffs = diffs[:, valid_cols]

            row_min = filtered_diffs.min(dim=1).values
            halo_sum = halo_all.sum(dim=1)
            row_score = row_min / (halo_sum + 1e-8)
        else:
            recv_sum = 1 - halo_counts.sum(dim=1)
            halo_sum = halo_all.sum(dim=1)
            row_score = recv_sum / (halo_sum + 1e-8)

        # Allocate blacklists to prevent selection duplicates across steps
        if not hasattr(self, 'global_blacklist_tensor') or self.global_blacklist_tensor.shape[0] != \
                node_dict["_ID"].shape[0]:
            self.global_blacklist_tensor = torch.zeros(node_dict["_ID"].shape[0], dtype=torch.bool, device=device)

        if len(self.selected_vertices) == 0:
            self.global_blacklist_tensor.zero_()

        # Apply float penalties to historical records
        is_blacklisted = self.global_blacklist_tensor[bound_node]
        if is_blacklisted.any():
            row_score = torch.where(is_blacklisted, row_score + 1e10, row_score)

        # Order and extract top-k seeds
        sorted_indices = torch.argsort(row_score)
        top_k = int(max(num_seed, 1))
        candidate_indices = sorted_indices[:top_k]
        selected_local_nodes = bound_node[candidate_indices]

        self.global_blacklist_tensor[selected_local_nodes] = True
        if hasattr(selected_local_nodes, "tolist"):
            self.selected_vertices.update(selected_local_nodes.tolist())

        return selected_local_nodes


class R_AC(nn.Module):
    def __init__(self, hidden_dim, num_parts):
        super(R_AC, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_parts = num_parts
        self.activation = torch.tanh

        self.conv_first = GraphConv(num_parts * 5 + 2, self.hidden_dim, norm="right")
        self.conv_common = nn.ModuleList([GraphConv(self.hidden_dim, self.hidden_dim, norm="right")])

        self.obs = nn.Linear(num_parts * 7, self.hidden_dim)
        self.obs_common = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim)])
        self.part_actor = GATConv(self.hidden_dim * 2, 1, num_heads=num_parts, allow_zero_in_degree=True)

    def forward(self, sub_graph, x, env_feat, seed, isTrain=True, isEval=False, actions=None):
        if len(seed) == 0:
            return [[], []], None

        if not isTrain:
            torch.set_num_threads(8)

        # Construct structural target masks
        mask_part = x[seed, self.num_parts * 4 + 1: self.num_parts * 5 + 1]
        base_mask = mask_part + x[seed, self.num_parts * 2: self.num_parts * 3]
        final_mask = base_mask.clamp(0, 1)

        # Relational convolution block computations
        h = self.activation(self.conv_first(sub_graph, x))
        for conv in self.conv_common:
            h = self.activation(conv(sub_graph, h))

        # Ambient system features projections
        obs_h = self.activation(self.obs(env_feat))
        for linear in self.obs_common:
            obs_h = self.activation(linear(obs_h))

        fused_feat = torch.cat([h, obs_h.unsqueeze(0).expand_as(h)], dim=-1)

        # Restrict sub-graphs during execution passes to accelerate inference profiles
        if not isTrain:
            last_layer_subgraph = dgl.in_subgraph(sub_graph, seed)
            part_logits = self.part_actor(last_layer_subgraph, fused_feat).squeeze(-1)
        else:
            part_logits = self.part_actor(sub_graph, fused_feat).squeeze(-1)

        seed_part_logits = part_logits[seed]

        # Dimension-level localized batch normalization
        local_mean = seed_part_logits.mean(dim=-1, keepdim=True)
        local_std = seed_part_logits.std(dim=-1, keepdim=True) + 1e-8
        seed_part_logits = (seed_part_logits - local_mean) / local_std * 5

        # Mask unauthorized routing transitions with large negative offsets
        filtered_logits = seed_part_logits.masked_fill(final_mask == 0, -1e10)
        if torch.isnan(filtered_logits).any() or torch.isinf(filtered_logits).any():
            raise ValueError("Logits contain NaN or Inf values.")

        part_dist = Categorical(logits=filtered_logits)

        if isEval:
            return part_dist.log_prob(actions[1]), part_dist.entropy().mean()

        if not isTrain:
            torch.set_num_threads(1)

        if isTrain:
            sampled_actions = part_dist.sample()
            return sampled_actions, part_dist.log_prob(sampled_actions)
        else:
            return torch.argmax(filtered_logits, dim=-1), None


class R_C(nn.Module):
    def __init__(self, hidden_dim, num_parts):
        super(R_C, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_parts = num_parts
        self.activation = nn.LeakyReLU(0.01)

        self.graph_in = GraphConv(num_parts, self.hidden_dim)
        self.graph_common = nn.ModuleList([GraphConv(self.hidden_dim, self.hidden_dim)])

        self.obs = nn.Linear(num_parts * 6, self.hidden_dim)
        self.obs_common = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim)])

        self.conv_critic = GraphConv(self.hidden_dim * 2, self.hidden_dim)
        self.ga_pool = GlobalAttentionPooling(gate_nn=nn.Linear(self.hidden_dim, 1))
        self.final_critic = nn.Linear(self.hidden_dim, 1)
        self.norm_critic = nn.LayerNorm(self.hidden_dim * 2)

    def forward(self, graph, env_feat):
        v_feat = env_feat[0]
        e_feat = env_feat[1].float()

        # Extract graph structures and ambient tracking fields
        h = self.activation(self.graph_in(graph, v_feat))
        for conv in self.graph_common:
            h = self.activation(conv(graph, h))

        obs_h = self.activation(self.obs(e_feat))
        for linear in self.obs_common:
            obs_h = self.activation(linear(obs_h))

        obs_h_expanded = obs_h.unsqueeze(0).expand_as(h)
        fused_feat = self.norm_critic(torch.cat([h, obs_h_expanded], dim=-1))

        # Perform global pooling for layout quality evaluations
        h_critic = self.activation(self.conv_critic(graph, fused_feat))
        graph_representation = self.ga_pool(graph, h_critic)

        return self.final_critic(graph_representation).squeeze(-1)