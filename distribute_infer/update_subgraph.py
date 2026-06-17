import numpy as np
from typing import Dict, Tuple, Optional

import torch
import torch.distributed as dist
import dgl


class GlobalToLocalMapper:
    """
    Utility for mapping unique global IDs to local relative offsets.
    Pre-allocates flat memory blocks for O(1) inversion lookups.
    """

    def __init__(self, global_ids: torch.Tensor) -> None:
        self.min_id = global_ids.min().item()
        self.max_id = global_ids.max().item()
        self.offset = self.min_id

        # Allocate flat mapping tensor initialized with empty flags (-1)
        self.id_map = torch.full(
            (self.max_id - self.min_id + 1,), -1,
            dtype=torch.long, device=global_ids.device
        )
        self.id_map[global_ids - self.offset] = torch.arange(len(global_ids), device=global_ids.device)

    def __call__(self, query: torch.Tensor) -> torch.Tensor:
        query_offset = query - self.offset
        valid = (query_offset >= 0) & (query_offset < len(self.id_map))
        return torch.where(valid, self.id_map[torch.clamp(query_offset, 0, len(self.id_map) - 1)], -1)


def filter_duplicate_edges_cpu_robust(
        lsrc: torch.Tensor,
        ldst: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filters out duplicate edges using a hash-based sort and binary search approach.
    """
    if len(src) == 0 or len(lsrc) == 0:
        return lsrc, ldst

    # Convert to uint64 arrays to avoid integer overflow during hash multiplication
    src_np = src.cpu().numpy().astype(np.uint64)
    dst_np = dst.cpu().numpy().astype(np.uint64)
    lsrc_np = lsrc.cpu().numpy().astype(np.uint64)
    ldst_np = ldst.cpu().numpy().astype(np.uint64)

    global_max_dst = max(dst_np.max(), ldst_np.max())
    multiplier = global_max_dst + 1

    # Map 2D coordinate pairs into 1D hashed key layouts and sort
    existing_hashes = src_np * multiplier + dst_np
    existing_hashes.sort()

    chunk_size = 10_000_000
    masks = []

    # Process input collections sequentially to optimize CPU cache performance
    for i in range(0, len(lsrc_np), chunk_size):
        chunk_hash = lsrc_np[i:i + chunk_size] * multiplier + ldst_np[i:i + chunk_size]

        # Use binary search to locate matching pre-existing keys
        idx = np.searchsorted(existing_hashes, chunk_hash)
        idx = np.clip(idx, 0, len(existing_hashes) - 1)

        is_duplicate = (existing_hashes[idx] == chunk_hash)
        masks.append(~is_duplicate)

    mask = np.concatenate(masks)
    return lsrc[mask], ldst[mask]


def optimized_insert(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Inserts unique elements from tensor B into tensor A without duplicates.
    """
    if B.numel() == 0:
        return A.clone(), torch.empty(0, dtype=torch.long, device=A.device), 0

    device = A.device
    gmap_A = GlobalToLocalMapper(A)
    mapped_indices = gmap_A(B)

    # Isolate newly appearing identifiers from pre-registered indices
    mask_new = (mapped_indices == -1)
    new_elements = B[mask_new]
    num_added = new_elements.size(0)

    modified_A = torch.cat([A, new_elements]) if num_added > 0 else A.clone()
    B_indices = torch.empty_like(B, dtype=torch.long, device=device)

    # Map pre-existing elements to their current relative indexes
    existing_mask = ~mask_new
    if existing_mask.any():
        B_indices[existing_mask] = mapped_indices[existing_mask]

    # Map newly added elements to their freshly allocated tail locations
    if num_added > 0:
        B_indices[mask_new] = len(A) + torch.arange(num_added, device=device)

    return modified_A, B_indices, num_added


def reconstruct_subgraph(
        graph: dgl.DGLGraph,
        node_dict: Dict[str, torch.Tensor],
        global_halo_nodes: torch.Tensor,
        halo_nodes_parts: torch.Tensor,
        global_t_nodes: torch.Tensor,
        t_labels: Optional[torch.Tensor],
        edges_src_global: torch.Tensor,
        edges_dst_global: torch.Tensor
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Integrates newly transferred topology elements and feature layers into the local graph.
    """
    device = node_dict['_ID'].device

    # Expand node tracking lists with incoming halo and target core records
    node_dict['_ID'], part_map, add_num_halo = optimized_insert(node_dict['_ID'], global_halo_nodes)
    node_dict['_ID'], local_map_t, add_num_t = optimized_insert(node_dict['_ID'], global_t_nodes)

    total_add_num = add_num_halo + add_num_t

    # Trigger internal allocation tracking expansions on the graph instance
    if total_add_num > 0:
        graph.add_nodes(total_add_num)

    # Pad partition routing maps and assign incoming locations to designated parts
    new_part_id = torch.cat([node_dict['part_id'], torch.full((total_add_num,), -1, dtype=torch.long, device=device)])
    new_part_id[part_map] = halo_nodes_parts.long()

    current_rank = dist.get_rank() if dist.is_initialized() else 0
    new_part_id[local_map_t] = current_rank
    node_dict['part_id'] = new_part_id

    # Append new mask assignments and designate newly transferred items as local inner nodes
    new_inner = torch.cat([node_dict['inner_node'], torch.zeros(total_add_num, dtype=torch.bool, device=device)])
    new_inner[local_map_t] = True
    node_dict['inner_node'] = new_inner

    # Concatenate incoming target class structures aligned with the current layout sorting order
    if t_labels is not None:
        _, sorted_idx = torch.sort(local_map_t)
        if 'labels' in node_dict:
            if t_labels.dim() == 1:
                node_dict['labels'] = torch.cat([node_dict['labels'], t_labels[sorted_idx].to(device)])
            else:
                node_dict['labels'] = torch.cat([node_dict['labels'], t_labels[sorted_idx, :].to(device)])

    # Synchronize internal configuration maps back to the root DGL instance registers
    graph.ndata['_ID'] = node_dict['_ID']
    graph.ndata['part_id'] = node_dict['part_id']
    graph.ndata['inner_node'] = node_dict['inner_node']
    if 'labels' in node_dict:
        graph.ndata['labels'] = node_dict['labels']

    # Insert incoming edge structures after removing cross-boundary duplicates
    if len(edges_src_global) > 0:
        gmap = GlobalToLocalMapper(graph.ndata['_ID'])
        lsrc = gmap(edges_src_global)
        ldst = gmap(edges_dst_global)

        local_t_nodes = gmap(global_t_nodes)
        existing_src, existing_dst = graph.in_edges(local_t_nodes, form='uv')

        new_src_single, new_dst_single = filter_duplicate_edges_cpu_robust(lsrc, ldst, existing_src, existing_dst)

        if len(new_src_single) > 0:
            new_src_bidir = torch.cat([new_src_single, new_dst_single])
            new_dst_bidir = torch.cat([new_dst_single, new_src_single])
            graph.add_edges(new_src_bidir, new_dst_bidir)

    return global_t_nodes, total_add_num, local_map_t


def broadcast_partition_changes(part_map: Optional[Dict[int, torch.Tensor]], src_rank: int) -> Dict[int, torch.Tensor]:
    """
    Broadcasts the global partition routing map from the source rank to all workers.
    """
    objects = [part_map if part_map is not None else {}]
    dist.broadcast_object_list(objects, src=src_rank)
    return objects[0]


def update_part(change_part: Dict[int, torch.Tensor], node_dict: Dict[str, torch.Tensor]) -> None:
    """
    Updates local partition records using the provided global movement ledger.
    """
    gmap = GlobalToLocalMapper(node_dict['_ID'])
    for k, v in change_part.items():
        valid_mask = torch.isin(v, node_dict['_ID'])
        valid_indices = gmap(v[valid_mask])
        node_dict['part_id'][valid_indices] = k