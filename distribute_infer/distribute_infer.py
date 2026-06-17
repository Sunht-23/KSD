import argparse
import logging
import gc
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F
import torch.distributed as dist
import dgl  # type: ignore

from Model.DistGCN import DistGCN
from distribute_infer.comm_util import get_boundary_nodes_NID, send_and_receive_embeddings_NID, gather_data
from distribute_infer.topology_utils import (
    recv_topology_edges_from_src,
    send_topology_edges_to_dst,
    extract_topology_for_partition,
    extract_migration_plan
)
from distribute_infer.update_subgraph import optimized_insert, GlobalToLocalMapper
from ppov2.ppo import PPO
from ppov2.ppo_util import get_vertex_feat, get_feat, update_action
from distribute_infer.util import load_partition, save_and_print_perf_report
from distribute_infer.performance import PerformanceStore

logger = logging.getLogger(__name__)


class DistInferClient:
    """
    Distributed Repartitioning and Inference Pipeline Client.
    Manages subgraph inference states, repartitioning actions via PPO,
    and feature synchronization across distributed partitions.
    """

    def __init__(self, model: torch.nn.Module, args: argparse.Namespace):
        self.args = args
        self.rank = args.local_rank
        self.world_size = args.world_size

        self.model = model
        self.model.eval()
        self.args.num_layers = model.n_layers

        self.graph: Optional[dgl.DGLGraph] = None
        self.node_dict: Optional[Dict] = None
        self.num_feat: int = 0
        self.graph_partition_book = None

        self.inner_node_indices: Optional[torch.Tensor] = None
        self.boundary_nodes = None
        self.is_env_initialized = False
        self.gmap: Optional[GlobalToLocalMapper] = None

        self.policy = PPO(args.ppo_dim, 0, args.world_size, args.ppo_num_seed)

    def load_policy_weights(self) -> None:
        """Loads pre-trained policy checkpoints."""
        graph_name = self.args.graph_name
        num_parts = self.world_size
        actor_path = f"data/ppo/{graph_name}_{num_parts}_actor.pt"

        if os.path.exists(actor_path):
            try:
                state_dict = torch.load(actor_path, map_location=torch.device('cpu'))
                self.policy.actor.load_state_dict(state_dict)
                logger.info(f"[Rank {self.rank}] Successfully loaded policy weights from {actor_path}")
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Failed to load pre-trained policy weights: {str(e)}")

    def init_env(self) -> None:
        """Initializes the distributed process group via Gloo backend."""
        if not self.is_env_initialized:
            if not dist.is_initialized():
                logger.info(f"--- [Rank {self.rank}] Initializing DDP Process Group via Gloo backend...")
                dist.init_process_group(
                    backend='gloo',
                    rank=self.rank,
                    world_size=self.world_size,
                    init_method=f'tcp://{self.args.master_addr}:{self.args.ddp_port}',
                )
            self.is_env_initialized = True

    def update_local_mapping(self) -> None:
        """Updates internal indexing and filters inner node locations."""
        self.gmap = GlobalToLocalMapper(self.node_dict['_ID'])
        self.inner_node_indices = torch.nonzero(self.node_dict["inner_node"], as_tuple=True)[0]

    def load_data_and_model(self) -> None:
        """Loads targeted partition structures and formats localized feature structures."""
        self.graph, self.node_dict, self.num_feat, self.graph_partition_book = load_partition(
            f'data/dgl_partition/{self.args.graph_name}/{self.world_size}-parts',
            self.args.graph_name,
            self.rank
        )

        self.update_local_mapping()

        if "ori_feat" not in self.node_dict:
            self.node_dict["ori_feat"] = torch.zeros((len(self.node_dict[dgl.NID]), self.num_feat))
            if "features" in self.node_dict:
                self.node_dict["ori_feat"][self.inner_node_indices] = self.node_dict["features"]
                del self.node_dict["features"]
                gc.collect()

        self.boundary_nodes = get_boundary_nodes_NID(self.node_dict)
        dist.barrier()

    def _align_feature_tensors(self) -> None:
        """Pads local storage allocations to reconcile newly incoming vertices."""
        target_len = len(self.node_dict['_ID'])
        for key in ['ori_feat', 'curr_layer_feat', 'labels']:
            if key in self.node_dict:
                curr_tensor = self.node_dict[key]
                curr_len = curr_tensor.shape[0]
                if curr_len < target_len:
                    diff = target_len - curr_len
                    if curr_tensor.dim() == 1:
                        pad = torch.zeros(diff, dtype=curr_tensor.dtype, device=curr_tensor.device)
                    else:
                        pad = torch.zeros((diff, curr_tensor.shape[1]), dtype=curr_tensor.dtype, device=curr_tensor.device)
                    self.node_dict[key] = torch.cat([curr_tensor, pad], dim=0)

    def run_inference_pipeline(self) -> Tuple[torch.Tensor, PerformanceStore]:
        """Executes full-pass GNN layer tracking over localized layouts."""
        self.node_dict["curr_layer_feat"] = self.node_feat = self.node_dict["ori_feat"].clone()
        perf_store = PerformanceStore()
        active_feat_tag = "curr_layer_feat"

        for curr_layer in range(0, self.args.num_layers):
            dist.barrier()

            # Synchronize remote boundary boundaries across cluster partitions
            send_and_receive_embeddings_NID(
                layer_tag=active_feat_tag,
                node_dict=self.node_dict,
                boundary_nodes=self.boundary_nodes,
                gmap=self.gmap,
                world_size=self.world_size,
                rank=self.rank,
                perf_store=perf_store
            )

            t_comp_start = time.time()
            with torch.no_grad():
                out, _, _ = self.model.inference_n_layer(
                    self.graph, self.node_dict[active_feat_tag], self.inner_node_indices,
                    self.args.device, self.args.batch_size, curr_layer
                )

            self.node_dict[active_feat_tag] = out
            perf_store.add_local_infer_time(time.time() - t_comp_start)

        final_output = out[self.inner_node_indices]
        return final_output, perf_store

    def gather_and_report_performance(self, perf_store: PerformanceStore) -> Dict[str, np.ndarray]:
        """Gathers and cross-all-gathers localized telemetry vectors."""
        self.init_env()

        local_infer_sum = perf_store.local_infer_time
        local_comm_sum = max(perf_store.real_send_time, perf_store.real_recv_time)
        local_total_sum = local_infer_sum + local_comm_sum

        local_send_vector = torch.zeros(self.world_size, dtype=torch.float32, device='cpu')
        subgraph_part_ids = self.node_dict["part_id"]
        for target_rank in range(self.world_size):
            count = torch.nonzero(subgraph_part_ids == target_rank, as_tuple=True)[0].numel()
            local_send_vector[target_rank] = count

        local_core_edges = torch.tensor(self.graph.num_edges(), dtype=torch.float32)

        # Broadcast telemetry blocks for target alignment
        total_infer_time = gather_data(torch.tensor([local_infer_sum], dtype=torch.float32)).numpy().flatten()
        total_comm_time = gather_data(torch.tensor([local_comm_sum], dtype=torch.float32)).numpy().flatten()
        total_send_time = gather_data(torch.tensor([perf_store.real_send_time], dtype=torch.float32)).numpy().flatten()
        total_recv_time = gather_data(torch.tensor([perf_store.real_recv_time], dtype=torch.float32)).numpy().flatten()

        total_exec_time = gather_data(torch.tensor([local_total_sum], dtype=torch.float32)).numpy().flatten()
        total_subgraph_edges = gather_data(local_core_edges).numpy().flatten().astype(np.int64)

        send_matrix_flat = gather_data(local_send_vector).numpy().flatten()
        send_matrix = send_matrix_flat.reshape(self.world_size, self.world_size)
        total_subgraph_nodes = np.diag(send_matrix).astype(np.int64)
        comm_matrix = send_matrix.copy()
        np.fill_diagonal(comm_matrix, 0)
        total_send_nodes = comm_matrix.sum(axis=1).astype(np.int64)
        total_recv_nodes = comm_matrix.sum(axis=0).astype(np.int64)

        if self.rank == 0:
            save_and_print_perf_report(
                graph_name=self.args.graph_name, world_size=self.world_size,
                total_infer_time=total_infer_time, total_comm_time=total_comm_time,
                total_send_time=total_send_time, total_recv_time=total_recv_time,
                total_exec_time=total_exec_time,
                total_send_nodes=total_send_nodes, total_recv_nodes=total_recv_nodes,
                total_subgraph_nodes=total_subgraph_nodes, total_subgraph_edges=total_subgraph_edges
            )

        return {
            "infer_time": total_infer_time,
            "comm_time": total_comm_time,
            "send_time": total_send_time,
            "recv_time": total_recv_time,
            "total_time": total_exec_time,
            "send_nodes": total_send_nodes,
            "recv_nodes": total_recv_nodes,
            "subgraph_nodes": total_subgraph_nodes,
            "subgraph_edges": total_subgraph_edges
        }


    # repartitioning method
    def run_repartition_pipeline(self, global_perf: Dict[str, np.ndarray], current_epoch: int) -> bool:
        """Checks for runtime skew and initializes adaptive structural adjustments."""
        self.init_env()
        dist.barrier()

        total_times = global_perf["total_time"]
        max_infer_time = np.max(total_times)
        min_infer_time = np.min(total_times)

        imbalance_ratio = max_infer_time / min_infer_time if min_infer_time > 0 else 1.0

        if imbalance_ratio < 1.15:
            if self.rank == 0:
                print(f"\n Imbalance ratio ({imbalance_ratio:.2f}x) under threshold (1.15x). Skipping repartition.")
            return False
        else:
            print(f"\n Imbalance ratio ({imbalance_ratio:.2f}x) >= 1.15x. Launching repartition pipeline...")

        slowest_rank = int(np.argmax(total_times))

        # Perform topological modification calculations
        t_topo_start = time.time()
        # KSD migration plan
        migration_plan = self.Keystone_Domino_migration(slowest_rank, global_perf)
        dist.barrier()
        t_topo_end = time.time()

        # Synchronize corresponding structural tensors
        t_feat_start = time.time()
        self.sync_features_and_labels(slowest_rank, migration_plan, "ori_feat")
        dist.barrier()
        t_feat_end = time.time()

        topo_time_local = torch.tensor([t_topo_end - t_topo_start], dtype=torch.float32)
        feat_time_local = torch.tensor([t_feat_end - t_feat_start], dtype=torch.float32)

        topo_time_all = gather_data(topo_time_local).numpy().flatten()
        feat_time_all = gather_data(feat_time_local).numpy().flatten()
        exchange_total_time_all = topo_time_all + feat_time_all

        local_change_val = 0
        if self.rank == slowest_rank:
            if migration_plan is not None and isinstance(migration_plan, dict):
                for k, v in migration_plan.items():
                    if k != slowest_rank and v is not None:
                        local_change_val -= v.numel()
        else:
            if migration_plan is not None and isinstance(migration_plan, torch.Tensor):
                local_change_val += migration_plan.numel()

        change_local = torch.tensor([local_change_val], dtype=torch.float32, device='cpu')
        change_node_all = gather_data(change_local).numpy().flatten()

        if self.rank == 0:
            print("\n" + "=" * 20 + f" REPARTITION TIME BREAKDOWN REPORT (ROUND {current_epoch}) " + "=" * 20)
            col_w = 12
            metrics_label_width = 24

            header = "Metrics".ljust(metrics_label_width) + "".join(
                f"│ {n.center(col_w - 3)} " for n in [f"node{i}" for i in range(self.world_size)])
            print(header)
            print("-" * (metrics_label_width + col_w * self.world_size))

            def print_table_row(label: str, arr: np.ndarray, is_int: bool = False):
                row_str = label.ljust(metrics_label_width)
                for v in arr:
                    if is_int:
                        if "Change" in label:
                            val_str = f"+{int(v)}" if v > 0 else (str(int(v)) if v < 0 else "0")
                        else:
                            val_str = f"{int(v):,}"
                    else:
                        val_str = f"{v:.4f} s"
                    row_str += f"│ {val_str.center(col_w - 3)} "
                print(row_str)

            infer_time_all = global_perf["infer_time"]
            comm_time_all = global_perf["comm_time"]

            print_table_row("1. Comp", infer_time_all)
            print_table_row("2. Comm", comm_time_all)
            print("-" * (metrics_label_width + col_w * self.world_size))

            print_table_row("Initial Inner Nodes", global_perf["subgraph_nodes"], is_int=True)
            print_table_row("Change Nodes", change_node_all, is_int=True)

            print("-" * (metrics_label_width + col_w * self.world_size))
            print_table_row("3. Repart Topo", topo_time_all)
            print_table_row("4. Repart Sync", feat_time_all)
            print_table_row("5. Exchange Total", exchange_total_time_all)
            print("=" * (metrics_label_width + col_w * self.world_size) + "\n")

        return True

    def fit_adaptive_repartition(self, max_epochs: int = 100) -> torch.Tensor:
        """Executes consecutive validation passes over the convergence repartitioning loop."""
        final_output = None
        num_passes = 2

        for epoch in range(max_epochs):
            if self.rank == 0:
                print(f"\n==================== Repartitioning Operation {epoch} ====================")

            epoch_smooth_perf = PerformanceStore()
            last_output = None

            for pass_idx in range(num_passes):
                dist.barrier()
                output, perf = self.run_inference_pipeline()
                last_output = output

                epoch_smooth_perf.add_local_infer_time(perf.local_infer_time)
                epoch_smooth_perf.add_send_time(perf.real_send_time)
                epoch_smooth_perf.add_recv_time(perf.real_recv_time)

            epoch_smooth_perf.average_metrics(num_passes=num_passes)
            final_output = last_output

            global_perf = self.gather_and_report_performance(epoch_smooth_perf)
            is_triggered = self.run_repartition_pipeline(global_perf, current_epoch=epoch)

            if not is_triggered:
                break

        return final_output

    # distributed KSD
    def Keystone_Domino_migration(self, slowest_rank: int, global_perf: Dict[str, np.ndarray]):
        """Encodes context parameters on the bottleneck worker to infer partition actions via RL."""
        update_actions = None

        if self.rank == slowest_rank:
            device = self.graph.device if hasattr(self, 'graph') else torch.device("cpu")

            # computation load and time
            node_tensor = torch.tensor(global_perf["subgraph_nodes"], dtype=torch.float32, device=device)
            edge_tensor = torch.tensor(global_perf["subgraph_edges"], dtype=torch.float32, device=device)
            comp_time = torch.tensor(global_perf["infer_time"], dtype=torch.float32, device=device)

            # communication load and time
            send_tensor = torch.tensor(global_perf["send_nodes"], dtype=torch.float32, device=device)
            recv_tensor = torch.tensor(global_perf["recv_nodes"], dtype=torch.float32, device=device)

            real_send_time_all = torch.tensor(global_perf["send_time"], dtype=torch.float32, device=device)
            real_recv_time_all = torch.tensor(global_perf["recv_time"], dtype=torch.float32, device=device)
            comm_time = torch.max(real_send_time_all, real_recv_time_all)

            # Replicate get_env latency breakdown normalization
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

            env_feat, vertex_feat = get_feat(all_env_feat, self.graph, self.node_dict, self.rank, self.world_size)

            # Keystone selection + RL action
            eval_actions = self.policy.act(self.graph, self.rank, self.node_dict, vertex_feat, env_feat)

            # Domino
            update_actions = update_action(eval_actions, self.graph, self.node_dict, self.rank, self.world_size)

        migration_plan = self.dist_one_to_all_split(update_actions, slowest_rank)

        self.update_local_mapping()
        self.boundary_nodes = get_boundary_nodes_NID(self.node_dict)
        self._align_feature_tensors()

        return migration_plan

    def dist_one_to_all_split(self, actions: Optional[list], slowest_rank: int):
        """Broadcasts movement ledgers and commits regional pointer re-assignments."""
        local_rank = self.rank
        shared_vars = {'t_nodes': None, 'add_num': 0, 'local_map': None}
        global_plan_dict = None
        part_map = {}

        if local_rank == slowest_rank:
            plans = extract_migration_plan(actions, self.node_dict, slowest_rank, self.world_size)
            if plans is not None:
                local_plan, global_plan = plans
                part_map = global_plan
                global_plan_dict = global_plan

        objects = [part_map]
        dist.broadcast_object_list(objects, src=slowest_rank)
        global_migration_ledger = objects[0]

        # Shift localized structural partition tags
        for k, v in global_migration_ledger.items():
            if v.numel() == 0:
                continue
            mapped_indices = self.gmap(v)
            valid_mask = (mapped_indices != -1)

            if valid_mask.any():
                local_indices = mapped_indices[valid_mask]
                self.node_dict['part_id'][local_indices] = k
                if local_rank == slowest_rank:
                    self.node_dict['inner_node'][local_indices] = False

        dist.barrier()

        # Build active point-to-point communication channels
        i_am_receiver = local_rank in global_migration_ledger

        if local_rank == slowest_rank:
            if global_plan_dict is not None:
                outbound_packages = extract_topology_for_partition(
                    self.graph, self.node_dict, local_plan, global_plan
                )
                for dst_part_id, package in outbound_packages.items():
                    send_topology_edges_to_dst(
                        dst_part_id,
                        package['global_halo_nodes'],
                        package['halo_nodes_parts'],
                        package['edges_src'],
                        package['edges_dst']
                    )
        else:
            if i_am_receiver:
                my_global_t_nodes = global_migration_ledger[local_rank]
                add_num, local_map = recv_topology_edges_from_src(
                    self.graph, self.node_dict, slowest_rank, my_global_t_nodes, optimized_insert
                )
                shared_vars['t_nodes'] = my_global_t_nodes
                shared_vars['add_num'] = add_num
                shared_vars['local_map'] = local_map

        dist.barrier()

        if local_rank == slowest_rank:
            return global_plan_dict
        return shared_vars['t_nodes']

    def sync_features_and_labels(self, slowest_rank: int, migration_plan, feat_tag: str) -> None:
        """Transfers corresponding feature embeddings and labels for relocated nodes."""
        local_rank = self.rank

        if local_rank == slowest_rank:
            if migration_plan is None:
                return

            for dst_part_id, global_t_nodes in migration_plan.items():
                if dst_part_id == slowest_rank or len(global_t_nodes) == 0:
                    continue

                local_indices = self.gmap(global_t_nodes.long())
                t_features = self.node_dict[feat_tag][local_indices]
                t_labels = self.node_dict['labels'][local_indices] if 'labels' in self.node_dict else None

                has_label = 1 if t_labels is not None else 0
                label_dim = t_labels.shape[1] if (t_labels is not None and t_labels.dim() > 1) else 1
                feat_meta = torch.tensor([has_label, label_dim, t_features.shape[1]], dtype=torch.int32)

                dist.send(feat_meta, dst=dst_part_id)
                dist.send(t_features.contiguous(), dst=dst_part_id)
                if has_label == 1:
                    dist.send(t_labels.contiguous(), dst=dst_part_id)
        else:
            global_t_nodes = migration_plan
            if global_t_nodes is None or len(global_t_nodes) == 0:
                return

            feat_meta = torch.empty(3, dtype=torch.int32)
            dist.recv(feat_meta, src=slowest_rank)
            has_label, label_dim, feat_dim = feat_meta.tolist()

            t_features = torch.empty((len(global_t_nodes), feat_dim), dtype=torch.float32)
            dist.recv(t_features, src=slowest_rank)

            local_indices = self.gmap(global_t_nodes.long())
            self.node_dict[feat_tag][local_indices] = t_features

            if has_label == 1:
                t_labels = torch.empty(len(global_t_nodes), dtype=torch.long) if label_dim == 1 else \
                    torch.empty((len(global_t_nodes), label_dim), dtype=torch.long)
                dist.recv(t_labels, src=slowest_rank)
                self.node_dict['labels'][local_indices] = t_labels

    def close(self) -> None:
        """Terminates and destroys the distributed backend process group context."""
        if dist.is_initialized():
            dist.destroy_process_group()