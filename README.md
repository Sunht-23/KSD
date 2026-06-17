# KSD

Repository Structure

* `main.py`: Main entry point to initialize the environment and run the repartition loop.
* `configs.json`: Stores model configurations (features, layers, hidden dimensions) for different datasets.
* `distribute_infer/`: Package for managing distributed GNN execution.
    * `distribute_infer.py`: Client that manages state transitions and runs the repartition pipeline.
    * `comm_util.py`: Peer-to-peer (P2P) communication handlers.
    * `performance.py`: Tracks thread-local execution latency and network traffic volumes.
    * `topology_utils.py`: Extracts and synchronizes sub-graph topological updates across partitions.
    * `update_subgraph.py`: Maps global IDs to local offsets and handles tensor insertions.
    * `util.py`: Functions to load graph partitions and print telemetry reports.
* `Model/`: Defines GNN model architectures.
    * `DistGCN.py`: Distributed Graph Convolutional Network implementation.
    * `DistGAT.py`: Distributed Graph Attention Network implementation.
* `ppov2/`: Reinforcement learning repartition engine.
    * `env_set.py`: Load-balancing environment that calculates states and step rewards.
    * `ppo.py`: Defines the Actor-Critic networks and seed vertex selection logic.
    * `trainPPO.py`: Training algorithms using clipped surrogate objectives.
    * `replayBuffer.py`: Handles experience storage and advantage (GAE) calculations.
    * `ppo_util.py`: Operators for graph loads, boundary nodes, and domino updates.

Environment Requirements

| Dependency | Version   |
|------------|-----------|
| Python     | 3.10.8    |
| PyTorch    | 2.1.1+cpu |
| DGL        | 2.4.0     |
| numpy      | 1.26.4    |
| pandas     | 2.3.3     |
