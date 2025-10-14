# Distributed Learned Hash Table

A production-ready Python implementation of LEAD (LEArned DHT), a distributed key-value storage system that uses machine learning models to enable efficient range queries while maintaining the benefits of traditional DHTs.

Based on the paper: **"A Distributed Learned Hash Table"** (Wang et al., 2025)

## Key Features

- **Efficient Range Queries**: 80-90% reduction in query latency compared to traditional DHTs
- **Order-Preserving Hashing**: Maintains key relationships for sequential access
- **Learned Index Structures**: Uses Recursive Model Index (RMI) for intelligent key mapping
- **Federated Learning**: Distributed model updates without central coordination
- **Load Balancing**: Virtual nodes (Shadow Balancer) for even key distribution
- **Fault Tolerance**: Automatic failure recovery with backup successors
- **Scalability**: Logarithmic routing complexity for single-key lookups

## Installation

```bash
# Install dependencies
pip install numpy

# Install LEAD library
pip install -e .
```

## Quick Start

### Single Node

```python
from lead_dht import LEADNode, LEADConfig

# Create and start a node
config = LEADConfig(ip='127.0.0.1', base_port=5000)
node = LEADNode(config)
node.start()

# Store data
node.put(key=12345, value="my data")

# Retrieve data
value = node.get(key=12345)

# Range query (the killer feature!)
results = node.range_query(start_key=10000, count=100)

# Cleanup
node.stop()
```

### Multi-Node Cluster

```python
from lead_dht import LEADCluster

# Create cluster
cluster = LEADCluster()

# Add bootstrap node
node1 = cluster.add_node(ip='127.0.0.1', base_port=5000)

# Add more nodes
node2 = cluster.add_node(ip='127.0.0.1', base_port=5100,
                        bootstrap_peer=('127.0.0.1', 5000))
node3 = cluster.add_node(ip='127.0.0.1', base_port=5200,
                        bootstrap_peer=('127.0.0.1', 5000))

# Use any node to store/retrieve
node1.put(key=123, value="data")
value = node2.get(key=123)

# Range queries work across the cluster
results = node1.range_query(start_key=100, count=50)

# View cluster stats
cluster.print_stats()

# Cleanup
cluster.stop_all()
```

## Configuration

```python
from lead_dht import LEADConfig

config = LEADConfig(
    ip='127.0.0.1',
    base_port=5000,
    num_virtual_nodes=10,           # More = better load balancing
    model_type='linear',             # 'linear' or 'cubic'
    branching_factor=100,            # RMI complexity
    model_update_threshold=0.4,      # Update at 40% new keys
    rpc_timeout=5.0,
    stabilize_interval=1.0,
    max_workers=20
)

node = LEADNode(config)
```

## API Reference

### LEADNode

#### Methods

- **`put(key: int, value: Any)`**: Store key-value pair
- **`get(key: int) -> Optional[Any]`**: Retrieve value by key
- **`range_query(start_key: int, count: int) -> List[Tuple[int, Any]]`**: Efficient range query
- **`retrain_model(sample_keys: Optional[np.ndarray] = None)`**: Retrain learned hash function
- **`federated_model_update()`**: Perform distributed model update
- **`get_stats() -> dict`**: Get node statistics
- **`start()`**: Start node and join network
- **`stop()`**: Stop node gracefully

### LEADCluster

#### Methods

- **`add_node(...) -> LEADNode`**: Add node to cluster
- **`stop_all()`**: Stop all nodes
- **`get_stats() -> dict`**: Get cluster statistics
- **`print_stats()`**: Print formatted statistics

### LEADConfig

Configuration dataclass with all tunable parameters. See docstrings for details.

## Architecture

### Components

1. **Learned Hash Function (RMI)**
   - Two-stage Recursive Model Index
   - Stage 1: Routes to bucket (always linear model)
   - Stage 2: Predicts position within bucket (linear or cubic)
   - Maintains order-preserving hash values

2. **Virtual Nodes (Shadow Balancer)**
   - Each physical node hosts multiple virtual nodes
   - Enables fine-grained load balancing
   - Supports heterogeneous node capacities

3. **DHT Overlay**
   - Chord-style finger table routing
   - Logarithmic lookup complexity: O(log N)
   - Successor/predecessor lists for fault tolerance

4. **Federated Model Updates**
   - Decentralized cooperative learning
   - Triggered when 40% (configurable) of keys are new
   - No central coordinator required

## Performance

Compared to traditional DHT (Chord):

- **Range Queries**: 10-20x faster
- **Message Cost**: 80-90% reduction
- **Single-Key Lookups**: Comparable performance
- **Scalability**: Maintains O(log N) routing

## Use Cases

- **LLM Serving**: KV cache management and sharing
- **Distributed Databases**: Range scans and sequential access
- **Content Delivery Networks**: Ordered content retrieval
- **Blockchain**: Transaction range queries
- **Time-Series Data**: Temporal range queries
- **Vector Databases**: Embedding similarity search

## Examples

See `examples.py` for comprehensive examples including:

1. Single node usage
2. Multi-node cluster
3. Range query performance
4. Heterogeneous cluster
5. Federated model updates
6. Custom configuration
7. Practical integration patterns

Run examples:

```bash
python examples.py
```

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run benchmarks
python benchmarks/performance_test.py
```

## Project Structure

```
lead_dht/
├── __init__.py          # Library exports
├── config.py            # Configuration
├── exceptions.py        # Custom exceptions
├── models.py            # Learned models (RMI)
├── utils.py             # Utility functions
├── peer.py              # Virtual node/peer
├── node.py              # Physical node
└── cluster.py           # Cluster management

examples.py              # Usage examples
README.md                # This file
setup.py                 # Package setup
requirements.txt         # Dependencies
```

## Requirements

- Python 3.7+
- NumPy >= 1.19.0

## License

MIT License - See LICENSE file for details

## Citation

If you use LEAD in your research, please cite:

```bibtex
@article{wang2025lead,
  title={A Distributed Learned Hash Table},
  author={Wang, Shengze and Liu, Yi and Zhang, Xiaoxue and Hu, Liting and Qian, Chen},
  journal={arXiv preprint arXiv:2508.14239},
  year={2025}
}
```

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Support

- GitHub Issues: [Report bugs or request features]
- Examples: See `examples.py`

## Acknowledgments

Based on research by Wang et al. at UC Santa Cruz and University of Nevada Reno.

Implementation incorporates ideas from:
- Chord DHT (Stoica et al.)
- Learned Indexes (Kraska et al.)
- Consistent Hashing

