from typing import Dict, Tuple, Optional

import torch
import torch.distributed as dist
import dgl

from distribute_infer.update_subgraph import GlobalToLocalMapper, filter_duplicate_edges_cpu_robust


@torch.no_grad()
def send_topology_edges_to_dst(
    dst_part_id: int,
    global_halo_nodes: torch.Tensor,
    halo_nodes_parts: torch.Tensor,
    edges_src: torch.Tensor,
    edges_dst: torch.Tensor
) -> None:
    """
    Dispatches localized topological delta updates to a specific target rank.
    Transfers edge connections and remote boundary tracking configurations, completely
    skipping core nodes already distributed via the global coordination blueprint.
    """
    metadata = torch.tensor([
        len(global_halo_nodes),
        len(edges_src)
    ], dtype=torch.int32)

    # 1. Dispatch structural chunk size descriptors
    dist.send(metadata, dst=dst_part_id)

    # 2. Streams dependent remote boundary points asynchronously
    if len(global_halo_nodes) > 0:
        dist.send(global_halo_nodes.to(torch.int32), dst=dst_part_id)
        dist.send(halo_nodes_parts.to(torch.int32), dst=dst_part_id)

    # 3. Stream strict multi-cast single-hop connection graphs
    if len(edges_src) > 0:
        dist.send(edges_src.to(torch.int32), dst=dst_part_id)
        dist.send(edges_dst.to(torch.int32), dst=dst_part_id)


@torch.no_grad()
def recv_topology_edges_from_src(
    graph: dgl.DGLGraph,
    node_dict: Dict[str, torch.Tensor],
    src_part_id: int,
    global_t_nodes: torch.Tensor,
    optimized_insert_fn
) -> Tuple[int, torch.Tensor]:
    """
    Intercepts streaming edge records from a source rank and dynamically modifies
    the local DGL sub-graph instance layouts via robust single-pass allocation padding.
    """
    metadata = torch.empty(2, dtype=torch.int32)
    dist.recv(metadata, src=src_part_id)
    halo_len, edge_dim = metadata.tolist()

    if halo_len > 0:
        global_halo_nodes = torch.empty(halo_len, dtype=torch.int32)
        halo_nodes_parts = torch.empty(halo_len, dtype=torch.int32)
        dist.recv(global_halo_nodes, src=src_part_id)
        dist.recv(halo_nodes_parts, src=src_part_id)
        global_halo_nodes = global_halo_nodes.long()
        halo_nodes_parts = halo_nodes_parts.long()
    else:
        global_halo_nodes = torch.empty(0, dtype=torch.long)
        halo_nodes_parts = torch.empty(0, dtype=torch.long)

    if edge_dim > 0:
        edges_src = torch.empty(edge_dim, dtype=torch.int32)
        edges_dst = torch.empty(edge_dim, dtype=torch.int32)
        dist.recv(edges_src, src=src_part_id)
        dist.recv(edges_dst, src=src_part_id)
        edges_src, edges_dst = edges_src.long(), edges_dst.long()

    # Calculate exact allocation targets beforehand to minimize host memory allocation loops
    node_dict['_ID'], part_map, add_num_halo = optimized_insert_fn(node_dict['_ID'], global_halo_nodes)
    node_dict['_ID'], local_map, add_num_t = optimized_insert_fn(node_dict['_ID'], global_t_nodes)

    total_add = add_num_halo + add_num_t

    # Trigger memory expansions on the underlying DGL C++ context view
    if total_add > 0:
        graph.add_nodes(total_add)

    # Perform unified single-pass padding cat operations to avoid cache misses
    new_part_id = torch.cat([node_dict['part_id'], torch.full((total_add,), -1, dtype=torch.long)])
    new_part_id[part_map] = halo_nodes_parts
    node_dict['part_id'] = new_part_id

    new_inner = torch.cat([node_dict['inner_node'], torch.zeros(total_add, dtype=torch.bool)])
    new_inner[local_map] = True  # Formally upgrades remote elements into active localized core nodes
    node_dict['inner_node'] = new_inner

    # Force immediate structural synchronizations back into the graph instance to avoid C++ state detachment
    graph.ndata['_ID'] = node_dict['_ID']
    graph.ndata['part_id'] = node_dict['part_id']
    graph.ndata['inner_node'] = node_dict['inner_node']

    # Undirected incremental connectivity generation: single-pass filtering -> mirror inversion -> append
    if edge_dim > 0:
        gmap = GlobalToLocalMapper(graph.ndata['_ID'])
        lsrc, ldst = gmap(edges_src), gmap(edges_dst)
        local_t_nodes_mapped = gmap(global_t_nodes)

        in_src, in_dst = graph.in_edges(local_t_nodes_mapped, form='uv')
        new_src_single, new_dst_single = filter_duplicate_edges_cpu_robust(lsrc, ldst, in_src, in_dst)

        if len(new_src_single) > 0:
            new_src_bidir = torch.cat([new_src_single, new_dst_single])
            new_dst_bidir = torch.cat([new_dst_single, new_src_single])
            graph.add_edges(new_src_bidir, new_dst_bidir)

    return total_add, local_map


def extract_migration_plan(
    action: Optional[list],
    local_node_dict: Dict[str, torch.Tensor],
    local_rank: int,
    num_parts: int
) -> Optional[Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]]:
    """
    Decodes raw PPO categorical choices into distinct inter-partition movement plans.
    Incorporates out-of-bound boundary tracking guards to suppress invalid ghost array
    indices before they hit downstream C++ pointer segments.

    Args:
        action (list): Local array lists returned from the PPO execution policy.
        local_node_dict (dict): Local tensor dictionary configurations containing dgl.NID.
        local_rank (int): Rank ID of the local worker process.
        num_parts (int): Global cluster size configuration (world_size).

    Returns:
        Optional[Tuple[dict, dict]]: (local_indices_plan, global_ids_plan) sorted by target ranks.
    """
    if not action or len(action) == 0:
        return None

    migration_global_plan = {}
    migration_plan = {}
    max_local_id = len(local_node_dict[dgl.NID]) - 1

    for p in range(num_parts):
        if len(action[p]) == 0 or p == local_rank:
            continue

        local_ids = action[p]
        if not isinstance(local_ids, torch.Tensor):
            local_ids = torch.tensor(local_ids, dtype=torch.long, device=local_node_dict[dgl.NID].device)

        # Boundary filtering guard tracking
        valid_mask = (local_ids >= 0) & (local_ids <= max_local_id)
        local_ids = local_ids[valid_mask]

        if len(local_ids) == 0:
            continue

        # Map relative localized pointers back to unique global cluster node indices
        global_id = local_node_dict[dgl.NID][local_ids]

        migration_global_plan[p] = global_id
        migration_plan[p] = local_ids

    if len(migration_plan) == 0:
        return None

    return migration_plan, migration_global_plan


def extract_topology_for_partition(
    graph: dgl.DGLGraph,
    local_node_dict: Dict[str, torch.Tensor],
    local_plan: Dict[int, torch.Tensor],
    global_plan: Dict[int, torch.Tensor]
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Isolates precise topology snapshots and sub-graph metadata tailored for specific target ranks.
    Slashes cross-network serialization volumes by 50% by extracting unilateral in-edges only,
    deferring symmetrical mirror conversions to destination processes.

    Args:
        graph (DGLGraph): Sub-graph snapshot hosted by the local cluster rank.
        local_node_dict (dict): Internal layout structures holding active routing attributes.
        local_plan (dict): Plan dict tracking target destinations mapped to relative local indices.
        global_plan (dict): Plan dict tracking target destinations mapped to unique cluster global IDs.

    Returns:
        Dict[int, Dict[str, torch.Tensor]]: Structured packages mapping destinations to data pools.
    """
    outbound_packages = {}

    for dst_part_id, local_t_nodes in local_plan.items():
        global_t_nodes = global_plan[dst_part_id]

        # Fetch in-bound edges in a single pass via local CSR matrix layouts
        src_local, dst_local = graph.in_edges(local_t_nodes, form='uv')

        edges_src_global = local_node_dict[dgl.NID][src_local]
        edges_dst_global = local_node_dict[dgl.NID][dst_local]

        unique_src_local = torch.unique(src_local)

        # Truncate redundant boundary updates already registered by destination nodes
        halo_mask = (local_node_dict['part_id'][unique_src_local] != dst_part_id)
        filtered_src_local = unique_src_local[halo_mask]

        global_halo_nodes = local_node_dict[dgl.NID][filtered_src_local]
        halo_nodes_parts = local_node_dict['part_id'][filtered_src_local]

        outbound_packages[dst_part_id] = {
            'global_t_nodes': global_t_nodes,
            'global_halo_nodes': global_halo_nodes,
            'halo_nodes_parts': halo_nodes_parts,
            'edges_src': edges_src_global,
            'edges_dst': edges_dst_global
        }

    return outbound_packages