import os
import torch
import numpy as np
import dgl
from dgl.convert import to_homogeneous
from dgl.data.utils import save_tensors
from dgl.distributed.partition import (
    _get_inner_node_mask,
    _get_inner_edge_mask,
    _save_graphs,
    _dump_part_config
)
from dgl.partition import partition_graph_with_halo, metis_partition_assignment
from dgl.base import EID, NID


def get_target_halo_nodes(target_rank, node_dict):
    """Retrieves localized halo node indices belonging to a specific target partition."""
    return torch.nonzero(node_dict["part_id"] == target_rank).squeeze(dim=1)


@torch.no_grad()
def one_step_nbr_tensor(graph, num_parts, all_inner_nodes, node_dict, local_rank, precomputed_degrees):
    """
    Computes a multi-channel neighborhood topology feature tensor using
    DGL message-passing primitives to determine boundary properties.
    """
    num_nodes = graph.num_nodes()
    total_cols = num_parts * 4
    device = node_dict["_ID"].device

    # Pre-allocate mask matrices and output buffers
    mask = torch.zeros((num_nodes, total_cols), dtype=torch.float32, device=device)
    send_tensor = torch.zeros(num_nodes, dtype=torch.long, device=device)

    # Convert inner node selections to boolean lookups for faster masking
    is_all_inner = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    is_all_inner[all_inner_nodes.long()] = True
    is_non_inner = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    degrees_1d = precomputed_degrees.view(-1)

    # Generate feature masks for each remote partition slice
    for part_id in range(num_parts):
        if part_id == local_rank:
            continue

        target_halo = get_target_halo_nodes(part_id, node_dict).long()
        if target_halo.numel() == 0:
            continue

        target_halo_1d = target_halo.view(-1)
        target_degree = degrees_1d[target_halo_1d].view(-1)
        filter_halo = target_halo_1d[target_degree == 1.0]

        # Extract local boundary nodes via out-edges of target halo nodes
        _, v = graph.out_edges(target_halo_1d, form='uv')
        target_bound = torch.unique(v)

        is_non_inner.zero_()
        is_non_inner[target_bound] = True
        is_non_inner[target_halo_1d] = True

        target_inner_mask = is_all_inner & (~is_non_inner)

        # Set specific channel offsets per partition
        mask[target_inner_mask, part_id] = 1.0
        mask[target_bound, part_id + num_parts] = 1.0
        mask[target_halo_1d, part_id + num_parts * 2] = 1.0
        mask[filter_halo, part_id + num_parts * 3] = 1.0

        send_tensor[target_bound] += 1

    # Perform message aggregation via DGL built-in operations
    graph.srcdata['_tmp_mask'] = mask
    graph.update_all(dgl.function.copy_u('_tmp_mask', 'm'), dgl.function.sum('m', 'nbr_sum'))
    nbr_tensor = graph.dstdata['nbr_sum'].long()

    del graph.srcdata['_tmp_mask']
    del graph.dstdata['nbr_sum']

    final_tensor = torch.cat([nbr_tensor, send_tensor.unsqueeze(1)], dim=1)
    return final_tensor


@torch.no_grad()
def get_vertex_feat(graph, node_dict, num_parts, local_rank):
    """Constructs local vertex embedding vectors matching the current topology layout."""
    device = node_dict["_ID"].device
    degrees = graph.in_degrees().float().view(-1, 1)

    all_nodes = torch.nonzero(node_dict["inner_node"] == 1, as_tuple=True)[0]
    nbr_tensor = one_step_nbr_tensor(graph, num_parts, all_nodes, node_dict, local_rank, degrees.squeeze(1))

    if not hasattr(graph, '_static_eye_cache'):
        graph._static_eye_cache = torch.eye(num_parts, dtype=torch.float32, device=device)

    safe_part_ids = torch.clamp(node_dict["part_id"].long(), min=0, max=num_parts - 1)
    part_feat = graph._static_eye_cache[safe_part_ids]

    vertex_feat = torch.cat([nbr_tensor, part_feat, degrees], dim=1)
    return vertex_feat


@torch.no_grad()
def get_feat(env_feat, sub_graph, node_dict, local_rank, num_parts):
    """Assembles localized environment features and individual node features."""
    part_indicator = torch.zeros(num_parts, dtype=torch.float32, device=env_feat.device)
    part_indicator[local_rank] = 1.0

    env_feat = torch.cat([part_indicator, env_feat], dim=0)
    vertex_feat = get_vertex_feat(sub_graph, node_dict, num_parts, local_rank)

    return env_feat, vertex_feat


@torch.no_grad()
def get_load(g, node_parts, num_parts):
    """Computes total communication volume and structure limits based on runtime partitions."""
    # get partitioned subgraphs with halo nodes
    parts, _, _ = partition_graph_with_halo(g, node_parts, 1, reshuffle=False)

    node_dict_list = []
    boundary_nodes_matrix = []
    node_num_list = []
    edge_num_list = []

    # Collect node and edge counts, and construct the boundary nodes matrix
    for i in range(num_parts):
        node_dict_list.append(parts[i].ndata)
        part_ids = parts[i].ndata['part_id']
        edge_num_list.append(parts[i].num_edges())

        # Count the number of nodes in each partition
        counts = torch.bincount(part_ids, minlength=num_parts)
        node_num_list.append(counts[i].item())
        boundary_nodes_matrix.append(counts)

    total_send = [0] * num_parts
    total_receive = [0] * num_parts

    # Calculate total send and receive counts based on the boundary nodes matrix
    for i in range(num_parts):
        for j in range(num_parts):
            node_count = boundary_nodes_matrix[i][j].item()
            if i == j:
                continue
            total_receive[i] += node_count
            total_send[j] += node_count

    return total_send, total_receive, node_num_list, edge_num_list, parts, node_dict_list


@torch.no_grad()
def exchange_part(graph, graph_name, num_parts, action, local_rank, local_node_dict, node_parts, policy=None):
    """Commits node re-assignment targets to the global routing mapping configuration."""
    if len(action) == 0:
        return False
    for p in range(len(action)):
        if len(action[p]) == 0 or local_rank == p:
            continue

        real_oid = local_node_dict[dgl.NID][action[p]]
        node_parts[real_oid] = p

    existing_parts = set(torch.unique(node_parts).tolist())
    valid_parts = set(range(num_parts))

    if not (valid_parts - existing_parts):
        return False

    missing_parts = list(valid_parts - existing_parts)
    if missing_parts:
        node_parts[:num_parts] = torch.arange(num_parts, dtype=node_parts.dtype)
        return True

    return False


def get_halo_nodes(local_rank, node_dict):
    """Returns indexes tracking nodes hosted on external remote partitions."""
    return torch.nonzero(node_dict["part_id"] != local_rank).squeeze(dim=1)


def get_bound_nodes(sub_graph, local_rank, node_dict, inner_nodes=None):
    """Locates nodes connected across boundaries to halo elements."""
    halo_nodes = get_halo_nodes(local_rank, node_dict)
    halo_nbr = sub_graph.in_edges(halo_nodes, form='uv')[0].unique()

    if inner_nodes is not None:
        mask = torch.isin(halo_nbr, inner_nodes)
    else:
        mask = torch.isin(halo_nbr, halo_nodes, invert=True)

    return halo_nbr[mask]


def partition_graph_metis_no_feat(g, graph_name, num_parts, data_dir):
    """Partitions input graphs via METIS and writes the split output files to disk."""
    num_hops = 1
    out_path = os.path.join(data_dir, 'dgl_partition', graph_name, f'{num_parts}-parts')

    sim_g = to_homogeneous(g)
    node_parts = metis_partition_assignment(sim_g, num_parts, balance_edges=True, objtype="cut")
    parts, orig_nids, _ = partition_graph_with_halo(sim_g, node_parts, num_hops, reshuffle=True)

    os.makedirs(out_path, mode=0o775, exist_ok=True)
    out_path = os.path.abspath(out_path)

    node_map_val = {}
    edge_map_val = {}

    for ntype in g.ntypes:
        ntype_id = g.get_ntype_id(ntype)
        node_map_val[ntype] = []
        for i in range(num_parts):
            mask = _get_inner_node_mask(parts[i], ntype_id)
            inner_nids = parts[i].ndata[NID][mask]
            node_map_val[ntype].append([int(inner_nids[0]), int(inner_nids[-1]) + 1])

    for etype in g.canonical_etypes:
        etype_id = g.get_etype_id(etype)
        edge_map_val[etype] = []
        for i in range(num_parts):
            mask = _get_inner_edge_mask(parts[i], etype_id)
            inner_eids = np.sort(parts[i].edata[EID][mask].numpy())
            edge_map_val[etype].append([int(inner_eids[0]), int(inner_eids[-1]) + 1])

    part_metadata = {
        "graph_name": graph_name,
        "num_nodes": g.num_nodes(),
        "num_edges": g.num_edges(),
        "num_parts": num_parts,
        "halo_hops": num_hops,
        "node_map": node_map_val,
        "edge_map": edge_map_val,
        "ntypes": {ntype: g.get_ntype_id(ntype) for ntype in g.ntypes},
        "etypes": {etype: g.get_etype_id(etype) for etype in g.canonical_etypes},
    }

    for part_id in range(num_parts):
        part = parts[part_id]
        part_dir = os.path.join(out_path, f"part{part_id}")
        os.makedirs(part_dir, mode=0o775, exist_ok=True)

        del part.ndata["orig_id"]
        del part.edata["orig_id"]

        graph_file = os.path.join(part_dir, "graph.dgl")
        save_tensors(os.path.join(part_dir, "node_feat.dgl"), {})
        save_tensors(os.path.join(part_dir, "edge_feat.dgl"), {})

        part_metadata[f"part-{part_id}"] = {
            "node_feats": os.path.relpath(os.path.join(part_dir, "node_feat.dgl"), out_path),
            "edge_feats": os.path.relpath(os.path.join(part_dir, "edge_feat.dgl"), out_path),
            "part_graph": os.path.relpath(graph_file, out_path),
        }

        _save_graphs(graph_file, [part], sort_etypes=(len(g.etypes) > 1))

    torch.save(orig_nids, os.path.join(out_path, "orig_nids.pt"))
    torch.save(node_parts, os.path.join(out_path, "node_parts.pt"))
    _dump_part_config(os.path.join(out_path, f"{graph_name}.json"), part_metadata)

    return node_parts, orig_nids


@torch.no_grad()
def get_v_group(sub_graph, node_dict, local_rank, num_parts, precomputed_degrees, alpha=0, action_len=1e10):
    """
    Performs local sub-graph delta modifications iteratively via vectorized assignment arrays.
    Updates neighborhood topology representations natively on the accelerator device.
    """
    device = node_dict["_ID"].device
    all_change = [torch.empty(0, dtype=torch.long, device=device) for _ in range(num_parts)]
    end = True

    num_nodes = sub_graph.num_nodes()
    total_cols = num_parts * 4
    degrees_1d = precomputed_degrees.view(-1)
    part_ids_all = node_dict["part_id"]

    all_inner_nodes_init = torch.where(node_dict["inner_node"] == 1)[0]
    nbr_tensor = one_step_nbr_tensor(sub_graph, num_parts, all_inner_nodes_init, node_dict, local_rank, precomputed_degrees)

    while end:
        all_inner_nodes_curr = torch.where(node_dict["inner_node"] == 1)[0]
        bound_node = get_bound_nodes(sub_graph, local_rank, node_dict, all_inner_nodes_curr)

        if bound_node.numel() == 0:
            break

        bound_vertex_feat = nbr_tensor[bound_node, :]
        inner_counts = bound_vertex_feat[:, :num_parts]
        halo_counts = bound_vertex_feat[:, num_parts * 3: num_parts * 4]
        send_parts = bound_vertex_feat[:, num_parts * 4]

        recv_score = (halo_counts - 1)
        send_score = 1 - inner_counts

        positive_nodes_per_part = [torch.empty(0, dtype=torch.long, device=device) for _ in range(num_parts)]
        end = False

        for part_idx in range(num_parts):
            if part_idx == local_rank:
                continue
            part_send_scores = send_score[:, part_idx]
            part_recv_scores = recv_score[:, part_idx]

            positive_mask = (part_send_scores >= alpha) & (part_recv_scores >= alpha) & (send_parts == 1)
            positive_nodes = bound_node[positive_mask]
            positive_nodes_per_part[part_idx] = positive_nodes

            if positive_nodes.numel() > 0:
                end = True
                all_change[part_idx] = torch.cat([all_change[part_idx], positive_nodes], dim=0)

        all_positive_nodes = []
        all_part_ids = []
        for p, nodes in enumerate(positive_nodes_per_part):
            if nodes.numel() > 0:
                all_positive_nodes.append(nodes)
                all_part_ids.append(torch.full_like(nodes, p, dtype=torch.long, device=device))

        if all_positive_nodes:
            actual_nodes = torch.cat(all_positive_nodes)
            target_parts = torch.cat(all_part_ids)

            node_dict["inner_node"][actual_nodes] = 0
            node_dict["part_id"][actual_nodes] = target_parts

            # Extract impacted neighborhood segments via incoming connections of modified nodes
            nbr_u, _ = sub_graph.in_edges(actual_nodes, form='uv')
            affected_nodes = torch.unique(torch.cat([actual_nodes, nbr_u])) if nbr_u.numel() > 0 else actual_nodes

            if affected_nodes.numel() > 0:
                aff_u, aff_v = sub_graph.out_edges(affected_nodes, form='uv')
                aff_v_parts = part_ids_all[aff_v].long()

                local_mask = torch.zeros((affected_nodes.numel(), total_cols), dtype=torch.float32, device=device)
                local_mask_flat = local_mask.view(-1)

                global_to_local = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
                global_to_local[affected_nodes] = torch.arange(affected_nodes.numel(), device=device)

                local_row_idx = global_to_local[aff_u]

                # Update channel B indicators (Outbound Boundaries)
                flat_idx_B = local_row_idx * total_cols + (aff_v_parts + num_parts)
                local_mask_flat[flat_idx_B] = 1.0

                # Update channel C indicators (Remote Halo boundaries)
                valid_v_mask = (aff_v_parts != local_rank) & (aff_v_parts >= 0)
                if valid_v_mask.any():
                    flat_idx_C = global_to_local[aff_u[valid_v_mask]] * total_cols + (aff_v_parts[valid_v_mask] + num_parts * 2)
                    local_mask_flat[flat_idx_C] = 1.0

                # Update channel D indicators (Unique cut-point fields)
                v_degrees = degrees_1d[aff_v]
                deg1_mask = (v_degrees == 1.0) & valid_v_mask
                if deg1_mask.any():
                    flat_idx_D = global_to_local[aff_u[deg1_mask]] * total_cols + (aff_v_parts[deg1_mask] + num_parts * 3)
                    local_mask_flat[flat_idx_D] = 1.0

                # Update channel A indicators (Relative Inner boundaries)
                all_inner_nodes_new = torch.where(node_dict["inner_node"] == 1)[0]
                aff_inner_nodes = all_inner_nodes_new[global_to_local[all_inner_nodes_new] >= 0]
                if aff_inner_nodes.numel() > 0:
                    for part_id in range(num_parts):
                        if part_id == local_rank:
                            continue
                        local_mask[global_to_local[aff_inner_nodes], part_id] = 1.0
                local_mask_flat[flat_idx_B - num_parts] = 0.0

                # Recompute fan-out tracking column indices
                send_tensor_local = local_mask[:, num_parts: num_parts * 2].sum(dim=1, keepdim=True)
                nbr_tensor[affected_nodes, :] = torch.cat([local_mask, send_tensor_local], dim=1).long()

    sub_graph.ndata['nbr_tensor'] = nbr_tensor
    return all_change


@torch.no_grad()
def update_action(action, sub_graph, node_dict, local_rank, num_parts):
    """Domino cascading migrations."""
    if action[0].numel() == 0 or len(action[0]) == 0:
        return action

    device = action[0].device

    node_dict["part_id"][action[0]] = action[1]
    node_dict["inner_node"][action[0]] = 0

    precomputed_degrees = sub_graph.in_degrees().float()
    v_group_changes = get_v_group(sub_graph, node_dict, local_rank, num_parts, precomputed_degrees)

    v_group_nodes_list = []
    v_group_parts_list = []
    for p in range(num_parts):
        if v_group_changes[p].numel() > 0:
            v_group_nodes_list.append(v_group_changes[p].long())
            v_group_parts_list.append(torch.full_like(v_group_changes[p], p, dtype=torch.long, device=device))

    if len(v_group_nodes_list) > 0:
        v_group_nodes = torch.cat(v_group_nodes_list, dim=0)
        v_group_parts = torch.cat(v_group_parts_list, dim=0)
    else:
        v_group_nodes = torch.empty(0, dtype=torch.long, device=device)
        v_group_parts = torch.empty(0, dtype=torch.long, device=device)

    up_action = [torch.empty(0, dtype=torch.long, device=device) for _ in range(num_parts)]

    total_nodes = torch.cat([action[0].long(), v_group_nodes], dim=0)
    total_parts = torch.cat([action[1].long(), v_group_parts], dim=0)

    if total_nodes.numel() > 0:
        # Sort values tracking target partitions to group arrays contiguously
        sorted_parts, perm_indices = torch.sort(total_parts)
        sorted_nodes = total_nodes[perm_indices]

        split_counts = torch.bincount(sorted_parts, minlength=num_parts).tolist()
        nodes_per_part_list = torch.split(sorted_nodes, split_counts, dim=0)

        for part in range(num_parts):
            if part == local_rank:
                continue
            up_action[part] = nodes_per_part_list[part]

    return up_action