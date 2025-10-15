"""
Tests for LEAD node implementation
"""

import pytest
import time
import numpy as np
from registry_service.dlht.node import LEADNode
from registry_service.dlht.config import LEADConfig
from registry_service.dlht.exceptions import NodeNotReadyException


class TestLEADNode:
    """Test cases for LEADNode"""

    def test_initialization(self):
        """Test node initialization"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=5
        )
        node = LEADNode(config)

        assert node.ip == '127.0.0.1'
        assert node.base_port == 5000
        assert node.num_virtual_nodes == 5
        assert not node.running
        assert not node.ready

    def test_initialization_default_config(self):
        """Test node initialization with default config"""
        node = LEADNode()

        assert node.ip is not None
        assert node.base_port > 0
        assert node.num_virtual_nodes > 0

    def test_virtual_nodes_creation(self):
        """Test that virtual nodes are created on start"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=6000,
            num_virtual_nodes=3
        )
        node = LEADNode(config)

        # Start node
        node.start()
        time.sleep(0.5)  # Give it time to initialize

        try:
            assert node.running
            assert node.ready
            assert len(node.virtual_nodes) == 3

            # Check each virtual node
            for vnode in node.virtual_nodes.values():
                assert vnode.successor is not None
                # First node's successor should be itself
                if not node.bootstrap_peer:
                    assert vnode.successor.vid == vnode.vid

        finally:
            node.stop()

    def test_put_and_get(self):
        """Test basic put and get operations"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=7000,
            num_virtual_nodes=3
        )
        node = LEADNode(config)
        node.start()
        time.sleep(0.5)

        try:
            # Put a value
            node.put(key=12345, value="test_value")

            # Get the value
            value = node.get(key=12345)
            assert value == "test_value"

            # Get non-existent key
            assert node.get(key=99999) is None

        finally:
            node.stop()

    def test_put_before_ready(self):
        """Test that put raises exception when node not ready"""
        node = LEADNode()

        with pytest.raises(NodeNotReadyException):
            node.put(key=100, value="test")

    def test_get_before_ready(self):
        """Test that get raises exception when node not ready"""
        node = LEADNode()

        with pytest.raises(NodeNotReadyException):
            node.get(key=100)

    def test_multiple_puts_and_gets(self):
        """Test multiple put and get operations"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=8000,
            num_virtual_nodes=5
        )
        node = LEADNode(config)
        node.start()
        time.sleep(0.5)

        try:
            # Put multiple values
            for i in range(100):
                node.put(key=i, value=f"value_{i}")

            # Verify all values
            for i in range(100):
                value = node.get(key=i)
                assert value == f"value_{i}"

        finally:
            node.stop()

    def test_range_query_simple(self):
        """Test simple range query"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=9000,
            num_virtual_nodes=3
        )
        node = LEADNode(config)
        node.start()
        time.sleep(0.5)

        try:
            # Insert sequential keys
            for i in range(0, 100, 10):
                node.put(key=i, value=f"value_{i}")

            # Retrain model for better locality
            node.retrain_model()

            # Range query
            results = node.range_query(start_key=20, count=5)

            # Should get keys starting from 20
            assert len(results) <= 5
            # Check that we got sequential keys
            if len(results) > 0:
                keys = [k for k, v in results]
                assert all(k >= 20 for k in keys)

        finally:
            node.stop()

    def test_range_query_large(self):
        """Test range query with larger dataset"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=9100,
            num_virtual_nodes=5
        )
        node = LEADNode(config)
        node.start()
        time.sleep(0.5)

        try:
            # Insert many keys
            for i in range(1000):
                node.put(key=i, value=f"value_{i}")

            # Retrain model
            node.retrain_model()

            # Range query
            results = node.range_query(start_key=500, count=50)

            # Should get up to 50 results
            assert len(results) <= 50

            # Keys should be >= 500
            if len(results) > 0:
                keys = [k for k, v in results]
                assert all(k >= 500 for k in keys)

        finally:
            node.stop()

    def test_retrain_model(self):
        """Test model retraining"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=9200,
            num_virtual_nodes=3
        )
        node = LEADNode(config)
        node.start()
        time.sleep(1.0)  # Increase startup time

        try:
            initial_version = node.learned_hash.version

            # Insert keys
            for i in range(100):
                node.put(key=i, value=f"value_{i}")

            time.sleep(0.5)

            # Retrain
            node.retrain_model()

            # Version should increment
            assert node.learned_hash.version > initial_version

            # Keys should still be accessible
            for i in range(100):
                assert node.get(key=i) == f"value_{i}"

        finally:
            node.stop()

    def test_retrain_model_with_custom_keys(self):
        """Test model retraining with custom keys"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=9300,
            num_virtual_nodes=3
        )
        node = LEADNode(config)
        node.start()
        time.sleep(0.5)

        try:
            initial_version = node.learned_hash.version

            # Retrain with custom keys
            custom_keys = np.arange(0, 1000, dtype=float)
            node.retrain_model(sample_keys=custom_keys)

            # Version should increment
            assert node.learned_hash.version > initial_version

        finally:
            node.stop()

    def test_federated_model_update(self):
        """Test federated model update"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=9400,
            num_virtual_nodes=3,
            model_update_threshold=0.4
        )
        node = LEADNode(config)
        node.start()
        time.sleep(0.5)

        try:
            # Insert initial keys
            for i in range(10):
                node.put(key=i, value=f"value_{i}")

            # Retrain to establish baseline
            node.retrain_model()

            # Mark all vnodes as ready for update
            for vnode in node.virtual_nodes.values():
                vnode.update_ready = True

            initial_version = node.learned_hash.version

            # Trigger federated update
            node.federated_model_update()

            # Version should increment
            assert node.learned_hash.version > initial_version

            # Update flags should be reset
            for vnode in node.virtual_nodes.values():
                assert not vnode.update_ready
                assert vnode.new_keys_count == 0

        finally:
            node.stop()

    def test_get_stats(self):
        """Test get_stats returns correct information"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=9500,
            num_virtual_nodes=3
        )
        node = LEADNode(config)
        node.start()
        time.sleep(0.5)

        try:
            # Insert some keys
            for i in range(50):
                node.put(key=i, value=f"value_{i}")

            stats = node.get_stats()

            assert stats['ip'] == '127.0.0.1'
            assert stats['base_port'] == 9500
            assert stats['num_virtual_nodes'] == 3
            assert stats['total_keys'] == 50
            assert stats['running']
            assert stats['ready']
            assert len(stats['virtual_nodes']) == 3

        finally:
            node.stop()

    def test_stop_and_restart(self):
        """Test stopping and restarting a node"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=9600,
            num_virtual_nodes=3
        )
        node = LEADNode(config)

        # Start
        node.start()
        time.sleep(0.5)
        assert node.running

        # Stop
        node.stop()
        time.sleep(0.2)
        assert not node.running

        # Restart
        node.start()
        time.sleep(0.5)
        assert node.running

        node.stop()

    def test_concurrent_operations(self):
        """Test concurrent put/get operations"""
        import threading

        config = LEADConfig(
            ip='127.0.0.1',
            base_port=9700,
            num_virtual_nodes=5
        )
        node = LEADNode(config)
        node.start()
        time.sleep(0.5)

        try:
            def put_keys(start, end):
                for i in range(start, end):
                    node.put(key=i, value=f"value_{i}")

            def get_keys(start, end):
                for i in range(start, end):
                    try:
                        node.get(key=i)
                    except:
                        pass  # Key might not exist yet

            # Create threads
            threads = []
            threads.append(threading.Thread(target=put_keys, args=(0, 100)))
            threads.append(threading.Thread(target=put_keys, args=(100, 200)))
            threads.append(threading.Thread(target=get_keys, args=(0, 200)))

            # Start threads
            for t in threads:
                t.start()

            # Wait for completion
            for t in threads:
                t.join()

            # Verify data integrity
            total_keys = sum(len(vnode.storage) for vnode in node.virtual_nodes.values())
            assert total_keys == 200

        finally:
            node.stop()

    def test_hash_space_size(self):
        """Test that hash space size is respected"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=9800,
            num_virtual_nodes=3,
            hash_space_size=2**32  # Smaller hash space
        )
        node = LEADNode(config)
        node.start()
        time.sleep(0.5)

        try:
            # Insert keys
            for i in range(100):
                node.put(key=i, value=f"value_{i}")

            # Verify model uses correct hash space
            assert node.learned_hash.hash_space_size == 2**32

            # Predictions should be within bounds
            for i in range(100):
                hash_val = node.learned_hash.predict(i)
                assert 0 <= hash_val < 2**32

        finally:
            node.stop()

    def test_different_model_types(self):
        """Test node with different model types"""
        # Test with linear models
        config1 = LEADConfig(
            ip='127.0.0.1',
            base_port=9900,
            num_virtual_nodes=2,
            model_type='linear',
            stage1_model_type='linear'
        )
        node1 = LEADNode(config1)
        node1.start()
        time.sleep(0.5)

        try:
            for i in range(50):
                node1.put(key=i, value=f"value_{i}")

            # Verify retrieval
            for i in range(50):
                assert node1.get(key=i) == f"value_{i}"

        finally:
            node1.stop()

        # Test with cubic stage1 model
        config2 = LEADConfig(
            ip='127.0.0.1',
            base_port=10000,
            num_virtual_nodes=2,
            model_type='linear',
            stage1_model_type='cubic'
        )
        node2 = LEADNode(config2)
        node2.start()
        time.sleep(0.5)

        try:
            for i in range(50):
                node2.put(key=i, value=f"value_{i}")

            # Verify retrieval
            for i in range(50):
                assert node2.get(key=i) == f"value_{i}"

        finally:
            node2.stop()


class TestLEADNodeWithBootstrap:
    """Test cases for node joining via bootstrap"""

    def test_node_join_via_bootstrap(self):
        """Test node joining network via bootstrap peer"""
        # Start first node (bootstrap)
        config1 = LEADConfig(
            ip='127.0.0.1',
            base_port=11000,
            num_virtual_nodes=2
        )
        node1 = LEADNode(config1)
        node1.start()
        time.sleep(0.5)

        try:
            # Insert data in first node
            for i in range(10):
                node1.put(key=i, value=f"value_{i}")

            # Start second node with bootstrap
            config2 = LEADConfig(
                ip='127.0.0.1',
                base_port=11100,
                num_virtual_nodes=2,
                bootstrap_peer=('127.0.0.1', 11000)
            )
            node2 = LEADNode(config2)
            node2.start()
            time.sleep(1.0)  # Give more time for join protocol

            # Both nodes should be running
            assert node1.running
            assert node2.running

            # Both should have virtual nodes
            assert len(node1.virtual_nodes) == 2
            assert len(node2.virtual_nodes) == 2

        finally:
            node2.stop()
            node1.stop()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
