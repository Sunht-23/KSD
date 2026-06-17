import time
from typing import Dict, List,  Optional, Any
import torch
import torch.distributed as dist
import dgl

def all_to_all_gloo_fixed_shape(
    input_list: List[torch.Tensor],
    feat_dim: int,
    world_size: int,
    rank: int,
    perf_store=None
) -> List[torch.Tensor]:
    if world_size == 1:
        return input_list[:]

    device = input_list[0].device
    dtype = input_list[0].dtype

    # Step 1: Exchange metadata flags to determine tensor sizes across ranks
    send_counts = torch.tensor([t.shape[0] for t in input_list], dtype=torch.long, device=device)
    recv_counts = torch.empty(world_size, dtype=torch.long, device=device)
    dist.all_to_all_single(recv_counts, send_counts)

    # Step 2: Pre-allocate receive spaces based on gathered counts
    output_list = []
    recv_tensors = []
    for i in range(world_size):
        n_nodes = recv_counts[i].item()
        if n_nodes == 0:
            output_list.append(torch.empty(0, feat_dim, dtype=dtype, device=device))
            recv_tensors.append(None)
        else:
            buf = torch.empty(n_nodes, feat_dim, dtype=dtype, device=device)
            output_list.append(buf)
            recv_tensors.append(buf)

    send_requests = []
    recv_requests = []

    # Initialize communication timeline tracking
    t_comm_start = time.time()

    # Step 3: Issue asynchronous bidirectional P2P communication pipelines
    for i in range(1, world_size):
        dst = (rank + i) % world_size
        src = (rank - i) % world_size

        if recv_tensors[src] is not None:
            req = dist.irecv(recv_tensors[src], src=src)
            recv_requests.append(req)

        send_t = input_list[dst]
        if send_t.shape[0] > 0:
            req = dist.isend(send_t.contiguous(), dst=dst)
            send_requests.append(req)

    # Process receive synchronization and log decoupled inbound latency
    if len(recv_requests) > 0:
        for req in recv_requests:
            req.wait()
        t_recv_done = time.time()
        recv_duration = t_recv_done - t_comm_start
    else:
        recv_duration = 0.0

    # Process send synchronization and log decoupled outbound full-duplex latency
    if len(send_requests) > 0:
        for req in send_requests:
            req.wait()
        t_send_done = time.time()
        send_duration = t_send_done - t_comm_start
    else:
        send_duration = 0.0

    # Commit decoupled unidirectional durations to the telemetry telemetry registry
    if perf_store:
        perf_store.add_send_time(send_duration)
        perf_store.add_recv_time(recv_duration)

    return output_list


def send_and_receive_embeddings_NID(
        layer_tag: str,
        node_dict: Dict,
        boundary_nodes: List,
        gmap: Any,
        world_size: int,
        rank: int,
        perf_store=None
) -> None:
    if world_size <= 1:
        return

    embeddings = node_dict[layer_tag]
    feat_dim = embeddings.shape[1]

    input_list = []
    send_bytes_total = 0
    for dst in range(world_size):
        gids = boundary_nodes[dst]
        if gids is None:
            gids = torch.empty(0, dtype=torch.long, device=embeddings.device)

        if gids.numel() == 0:
            send_tensor = torch.empty(0, feat_dim, device=embeddings.device, dtype=embeddings.dtype)
        else:
            local_ids = gmap(gids)
            send_tensor = embeddings[local_ids].contiguous()

        input_list.append(send_tensor)
        if perf_store:
            send_bytes_total += send_tensor.numel() * send_tensor.element_size()

    # point-to-point communication primitive to exchange features
    received_list = all_to_all_gloo_fixed_shape(
        input_list, feat_dim, world_size, rank, perf_store=perf_store
    )

    for src in range(world_size):
        recv_tensor = received_list[src]
        if recv_tensor.shape[0] == 0:
            continue

        mask = (node_dict["part_id"] == src)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        node_dict[layer_tag][indices] = recv_tensor


def get_boundary_nodes_NID(node_info_dict: Dict) -> List[Optional[torch.Tensor]]:
    """
    Identifies and retrieves cross-partition boundary nodes by exchanging
    structural metadata among distributed ranks.
    """
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = node_info_dict["part_id"].device
    boundary = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size

        belong_right = (node_info_dict["part_id"] == right)
        num_right = belong_right.sum().view(-1)
        num_left = torch.tensor([0], dtype=torch.long, device=device)

        # Exchange node counts between ranks asynchronously
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)

        v = node_info_dict[dgl.NID][belong_right]
        u = torch.zeros(num_left.item(), dtype=torch.long, device=device)

        # Wait for count buffer to be flushed safely
        req.wait()

        # Exchange global node ID (NID) payloads
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)

        boundary[left] = u

        # Wait for structural data to be intercepted securely
        req.wait()

    return boundary


def gather_data(time_data: torch.Tensor) -> torch.Tensor:
    """
    Aggregates runtime data from all distributed ranks.
    Supports input tensors with varying lengths across ranks.
    """
    assert time_data.dtype == torch.float32, "Data tracking array payload must be float32."

    # Align dimensions of the local tensor
    if time_data.dim() == 0:
        time_data = time_data.unsqueeze(0)
    if time_data.dim() == 1:
        time_data = time_data.unsqueeze(1)

    local_shape = time_data.shape

    # Gather shape metadata from all ranks
    shape_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(shape_list, local_shape)

    # Pre-allocate buffers and gather actual data payloads
    gathered_list = [torch.zeros(s, dtype=torch.float32, device=time_data.device) for s in shape_list]
    dist.all_gather(gathered_list, time_data)

    # Concatenate profiles into a single tensor
    return torch.cat(gathered_list, dim=0)