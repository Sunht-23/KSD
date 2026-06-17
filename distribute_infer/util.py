import os
import gc
import logging
from typing import Dict, Tuple, Optional

import torch
import dgl  # type: ignore
from dgl.distributed import GraphPartitionBook  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


def load_partition(
        partition_dir: str,
        dataset_name: str,
        part_id: int
) -> Tuple[dgl.DGLGraph, Dict[str, torch.Tensor], int, GraphPartitionBook]:
    """
    Loads partitioned graph fragments, metadata, and mapping indices from disk.
    """
    partition_config = os.path.join(partition_dir, f"{dataset_name}.json")

    (
        sub_graph,
        node_feat,
        _,
        graph_partition_book,
        _,
        node_type,
        _,
    ) = dgl.distributed.load_partition(partition_config, part_id)

    node_type_str = node_type[0]

    # Map core graph identifiers into node feature dictionary
    node_feat[dgl.NID] = sub_graph.ndata[dgl.NID]
    node_feat["part_id"] = sub_graph.ndata["part_id"]
    node_feat["inner_node"] = sub_graph.ndata["inner_node"].bool()

    possible_feat_keys = [f"{node_type_str}/feat", f"{node_type_str}/features"]
    possible_label_keys = [f"{node_type_str}/label", f"{node_type_str}/labels"]

    # Locate and assign feature fields
    for key in possible_feat_keys:
        if key in node_feat:
            node_feat["features"] = node_feat[key]
            node_feat.pop(key)
            break
    else:
        raise KeyError(f"Missing feature channels in node_feat. Tried: {possible_feat_keys}")

    # Locate and assign label fields or fallback to zero initialization
    for key in possible_label_keys:
        if key in node_feat:
            node_feat["labels"] = node_feat[key]
            node_feat.pop(key)
            break
    else:
        logger.warning(f"Label fields missing. Initializing zero placeholders: {possible_label_keys}")
        node_feat["labels"] = torch.zeros(len(node_feat["features"]), dtype=torch.long)

    # Clear memory fields on the graph object to prevent duplicates
    sub_graph.ndata.clear()
    sub_graph.edata.clear()

    n_feat = node_feat["features"].shape[1]

    # Remove temporary or intermediate training mask features
    features_to_remove = [
        f"{node_type_str}/label", f"{node_type_str}/feat",
        f"{node_type_str}/in_deg", f"{node_type_str}/out_deg",
        f"{node_type_str}/train_mask", f"{node_type_str}/val_mask", f"{node_type_str}/test_mask"
    ]

    for feature in features_to_remove:
        if feature in node_feat:
            del node_feat[feature]

    sub_graph.ndata["labels"] = torch.full((len(sub_graph.nodes()),), -100)

    return sub_graph, node_feat, n_feat, graph_partition_book


def save_and_print_perf_report(
        graph_name: str,
        world_size: int,
        total_infer_time: np.ndarray,
        total_comm_time: np.ndarray,
        total_send_time: np.ndarray,
        total_recv_time: np.ndarray,
        total_exec_time: np.ndarray,
        total_send_nodes: np.ndarray,
        total_recv_nodes: np.ndarray,
        total_subgraph_nodes: np.ndarray,
        total_subgraph_edges: np.ndarray
) -> None:
    """
    Prints aggregated execution metrics across ranks and saves telemetry arrays to disk.
    """
    num_nodes = total_infer_time.shape[0]
    node_ids = [f"node{i}" for i in range(num_nodes)]

    metrics_label_width = 30
    col_width = max(max(len(n) for n in node_ids), 14)

    total_width = metrics_label_width + (col_width + 3) * num_nodes
    double_line = "=" * total_width
    single_line = "-" * total_width

    print(f"\n{double_line}")
    print(" DISTRIBUTED METRICS REPORT")
    print(double_line)

    header = "Performance Metrics".ljust(metrics_label_width) + "".join(f"│ {n.center(col_width)} " for n in node_ids)
    print(header)
    print(single_line)

    def print_row(label: str, arr: np.ndarray, fmt_type: str = "time"):
        row_str = label.ljust(metrics_label_width)
        for v in arr:
            if fmt_type == "time":
                val_str = f"{v:.4f} s"
            elif fmt_type == "mb":
                val_str = f"{v:.2f} MB"
            else:
                val_str = f"{int(v):,}"
            row_str += f"│ {val_str.rstrip().rjust(col_width)} "
        print(row_str)

    print_row("1. Computation Time", total_infer_time, "time")
    print_row("2. Communication Time", total_comm_time, "time")
    print_row("2a. Send Time", total_send_time, "time")
    print_row("2b. Recv Time", total_recv_time, "time")
    print_row("3. E2E Latency", total_exec_time, "time")
    print(single_line)

    print_row("4. Send Vertices Counts", total_send_nodes, "count")
    print_row("5. Recv Vertices Counts", total_recv_nodes, "count")
    print(single_line)

    print_row("6. Partition Vertex Counts", total_subgraph_nodes, "count")
    print_row("7. Partition Edges Counts", total_subgraph_edges, "count")
    print(double_line)

    # Save finalized telemetry data fields to disk
    base_path = f'data/dgl_partition/{graph_name}/{world_size}-parts'
    os.makedirs(base_path, exist_ok=True)

    save_kwargs = {
        "node_ids": np.array(node_ids),
        "total_infer_time": total_infer_time,
        "total_comm_time": total_comm_time,
        "total_send_time": total_send_time,
        "total_recv_time": total_recv_time,
        "total_time": total_exec_time,
        "total_send_nodes": total_send_nodes,
        "total_recv_nodes": total_recv_nodes,
        "total_subgraph_nodes": total_subgraph_nodes,
        "total_subgraph_edges": total_subgraph_edges
    }
    np.savez(f'{base_path}/final_results.npz', **save_kwargs)
    np.savez(f'{base_path}/gcn_final_results_ldg.npz', **save_kwargs)