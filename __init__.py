"""
LEAD: Distributed Learned Hash Table Library
A production-ready implementation for order-preserving distributed key-value storage

Usage:
    from lead_dht import LEADNode, LEADCluster, LEADConfig
    
    # Simple single-node usage
    config = LEADConfig(ip='127.0.0.1', base_port=5000)
    node = LEADNode(config)
    node.start()
    
    node.put(key=12345, value="my data")
    value = node.get(key=12345)
    results = node.range_query(start_key=10000, count=100)
    
    # Multi-node cluster
    cluster = LEADCluster()
    cluster.add_node(ip='127.0.0.1', base_port=5000)
    cluster.add_node(ip='127.0.0.1', base_port=5100, 
                    bootstrap_peer=('127.0.0.1', 5000))
"""

from .config import LEADConfig
from .node import LEADNode
from .cluster import LEADCluster
from .models import RecursiveModelIndex, LinearModel, CubicModel
from .peer import LEADPeer, FingerEntry
from .exceptions import (
    LEADException,
    NetworkException,
    ModelException,
    KeyNotFoundException
)

__version__ = '1.0.0'
__author__ = 'LEAD Implementation Team'

__all__ = [
    'LEADNode',
    'LEADCluster',
    'LEADConfig',
    'LEADPeer',
    'FingerEntry',
    'RecursiveModelIndex',
    'LinearModel',
    'CubicModel',
    'LEADException',
    'NetworkException',
    'ModelException',
    'KeyNotFoundException'
]