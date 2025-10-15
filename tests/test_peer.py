"""
Tests for LEAD peer (virtual node) implementation
"""

import pytest
import time
from registry_service.dlht.peer import LEADPeer, FingerEntry, KeyValuePair
from registry_service.dlht.config import LEADConfig
from unittest.mock import Mock, MagicMock


class TestFingerEntry:
    """Test cases for FingerEntry"""

    def test_creation(self):
        """Test FingerEntry creation"""
        entry = FingerEntry(vid=12345, ip='127.0.0.1', port=5000)
        assert entry.vid == 12345
        assert entry.ip == '127.0.0.1'
        assert entry.port == 5000

    def test_to_dict(self):
        """Test serialization to dict"""
        entry = FingerEntry(vid=12345, ip='127.0.0.1', port=5000)
        data = entry.to_dict()

        assert data['vid'] == 12345
        assert data['ip'] == '127.0.0.1'
        assert data['port'] == 5000

    def test_from_dict(self):
        """Test deserialization from dict"""
        data = {'vid': 12345, 'ip': '127.0.0.1', 'port': 5000}
        entry = FingerEntry.from_dict(data)

        assert entry.vid == 12345
        assert entry.ip == '127.0.0.1'
        assert entry.port == 5000


class TestKeyValuePair:
    """Test cases for KeyValuePair"""

    def test_creation(self):
        """Test KeyValuePair creation"""
        kv = KeyValuePair(key=100, value="test_value")
        assert kv.key == 100
        assert kv.value == "test_value"
        assert kv.timestamp > 0

    def test_custom_timestamp(self):
        """Test KeyValuePair with custom timestamp"""
        custom_time = 12345.0
        kv = KeyValuePair(key=100, value="test", timestamp=custom_time)
        assert kv.timestamp == custom_time


class TestLEADPeer:
    """Test cases for LEADPeer"""

    def setup_method(self):
        """Setup for each test"""
        self.config = LEADConfig()
        self.physical_node = Mock()
        self.physical_node.config = self.config
        self.peer = LEADPeer(
            vid=1000,
            ip='127.0.0.1',
            port=5000,
            physical_node=self.physical_node,
            hash_space_size=2**160
        )

    def test_initialization(self):
        """Test peer initialization"""
        assert self.peer.vid == 1000
        assert self.peer.ip == '127.0.0.1'
        assert self.peer.port == 5000
        assert len(self.peer.finger_table) == 160
        assert len(self.peer.storage) == 0

    def test_in_range_no_wrap(self):
        """Test in_range without ring wrap-around"""
        peer = self.peer

        # Test (10, 20] with key=15
        assert peer.in_range(15, 10, 20, inclusive_end=True)

        # Test (10, 20) with key=15
        assert peer.in_range(15, 10, 20, inclusive_end=False)

        # Test boundaries
        assert not peer.in_range(10, 10, 20, inclusive_end=False)
        assert peer.in_range(20, 10, 20, inclusive_end=True)
        assert not peer.in_range(20, 10, 20, inclusive_end=False)

    def test_in_range_with_wrap(self):
        """Test in_range with ring wrap-around"""
        peer = self.peer

        # Wrap around: (90, 10]
        assert peer.in_range(95, 90, 10, inclusive_end=True)
        assert peer.in_range(5, 90, 10, inclusive_end=True)
        assert peer.in_range(10, 90, 10, inclusive_end=True)

        # Should not be in range
        assert not peer.in_range(50, 90, 10, inclusive_end=True)

    def test_put_and_get(self):
        """Test put and get operations"""
        peer = self.peer

        # Put a value
        peer.put(100, "value1")
        assert len(peer.storage) == 1

        # Get the value
        value = peer.get(100)
        assert value == "value1"

        # Get non-existent key
        assert peer.get(999) is None

    def test_put_updates_counters(self):
        """Test that put updates key counters"""
        peer = self.peer

        # Put new keys
        peer.put(100, "value1")
        assert peer.new_keys_count == 1
        assert peer.total_keys_count == 1

        peer.put(200, "value2")
        assert peer.new_keys_count == 2
        assert peer.total_keys_count == 2

        # Update existing key (should not increment new_keys_count)
        peer.put(100, "updated_value")
        assert peer.new_keys_count == 2
        assert peer.total_keys_count == 2

    def test_put_triggers_update_ready(self):
        """Test that put triggers update_ready flag"""
        peer = self.peer
        peer.update_threshold = 0.4

        # Add keys to reach threshold
        for i in range(10):
            peer.put(i, f"value{i}")

        # Initially not ready
        assert not peer.update_ready

        # Add more keys to exceed 40% threshold
        for i in range(10, 18):
            peer.put(i, f"value{i}")

        # Should now be ready for update
        assert peer.update_ready

    def test_get_range(self):
        """Test get_range operation"""
        peer = self.peer

        # Insert keys
        for i in range(0, 100, 10):
            peer.put(i, f"value{i}")

        # Get range starting at 20, count 5
        results = peer.get_range(20, 5)

        assert len(results) == 5
        assert results[0] == (20, "value20")
        assert results[1] == (30, "value30")
        assert results[4] == (60, "value60")

    def test_get_range_with_no_keys(self):
        """Test get_range with empty storage"""
        peer = self.peer

        results = peer.get_range(0, 10)
        assert len(results) == 0

    def test_get_range_with_start_key_beyond_all(self):
        """Test get_range with start_key beyond all stored keys"""
        peer = self.peer

        for i in range(0, 50, 10):
            peer.put(i, f"value{i}")

        results = peer.get_range(100, 5)
        assert len(results) == 0

    def test_get_range_partial_results(self):
        """Test get_range when fewer keys available than requested"""
        peer = self.peer

        for i in range(0, 30, 10):
            peer.put(i, f"value{i}")

        # Request 10 keys but only 2 available starting from 10
        results = peer.get_range(10, 10)
        assert len(results) == 2
        assert results[0] == (10, "value10")
        assert results[1] == (20, "value20")

    def test_transfer_keys(self):
        """Test key transfer to another node"""
        peer = self.peer

        # Insert keys
        for i in range(0, 100, 10):
            peer.put(i, f"value{i}")

        initial_count = len(peer.storage)

        # Transfer keys in range (20, 60]
        transferred = peer.transfer_keys(20, 60)

        # Check transferred keys
        transferred_keys = [kv.key for kv in transferred]
        assert 30 in transferred_keys
        assert 40 in transferred_keys
        assert 50 in transferred_keys
        assert 60 in transferred_keys

        # Check keys removed from storage
        assert len(peer.storage) < initial_count
        assert 30 not in peer.storage
        assert 40 not in peer.storage

        # Keys outside range should still exist
        assert 0 in peer.storage
        assert 70 in peer.storage

    def test_find_successor_self(self):
        """Test find_successor when peer is responsible"""
        peer = self.peer
        peer.vid = 1000
        peer.successor = FingerEntry(vid=1000, ip='127.0.0.1', port=5000)

        # Key hash between self and successor (which is self)
        successor = peer.find_successor(1000)

        assert successor.vid == 1000

    def test_closest_preceding_node_no_fingers(self):
        """Test closest_preceding_node with empty finger table"""
        peer = self.peer
        peer.vid = 1000

        # Should return self when no fingers available
        closest = peer.closest_preceding_node(2000)
        assert closest.vid == peer.vid

    def test_notify_updates_predecessor(self):
        """Test that notify updates predecessor"""
        peer = self.peer
        peer.vid = 1000
        peer.predecessor = None

        # Notify with potential predecessor
        potential_pred = FingerEntry(vid=500, ip='127.0.0.1', port=4999)
        peer.notify(potential_pred)

        assert peer.predecessor == potential_pred

    def test_notify_replaces_better_predecessor(self):
        """Test that notify replaces predecessor with better candidate"""
        peer = self.peer
        peer.vid = 1000
        peer.predecessor = FingerEntry(vid=500, ip='127.0.0.1', port=4999)

        # Notify with better predecessor (closer to peer)
        better_pred = FingerEntry(vid=800, ip='127.0.0.1', port=4998)
        peer.notify(better_pred)

        assert peer.predecessor.vid == 800

    def test_notify_ignores_worse_predecessor(self):
        """Test that notify ignores worse predecessor"""
        peer = self.peer
        peer.vid = 1000
        peer.predecessor = FingerEntry(vid=800, ip='127.0.0.1', port=4999)

        # Notify with worse predecessor (farther from peer)
        worse_pred = FingerEntry(vid=500, ip='127.0.0.1', port=4998)
        peer.notify(worse_pred)

        # Should keep original predecessor
        assert peer.predecessor.vid == 800

    def test_handle_successor_failure_with_backup(self):
        """Test successor failure handling with backup"""
        peer = self.peer
        peer.successor = FingerEntry(vid=2000, ip='127.0.0.1', port=5001)
        backup = FingerEntry(vid=3000, ip='127.0.0.1', port=5002)
        peer.successor_list = [backup]

        peer.handle_successor_failure()

        # Should promote backup to successor
        assert peer.successor.vid == 3000
        assert len(peer.successor_list) == 0

    def test_handle_successor_failure_no_backup(self):
        """Test successor failure handling without backup"""
        peer = self.peer
        peer.vid = 1000
        peer.successor = FingerEntry(vid=2000, ip='127.0.0.1', port=5001)
        peer.successor_list = []

        peer.handle_successor_failure()

        # Should set self as successor
        assert peer.successor.vid == peer.vid

    def test_get_stats(self):
        """Test get_stats returns correct information"""
        peer = self.peer

        # Add some keys
        for i in range(10):
            peer.put(i, f"value{i}")

        peer.successor = FingerEntry(vid=2000, ip='127.0.0.1', port=5001)
        peer.predecessor = FingerEntry(vid=500, ip='127.0.0.1', port=4999)

        stats = peer.get_stats()

        assert stats['vid'] == peer.vid
        assert stats['num_keys'] == 10
        assert stats['has_successor'] is True
        assert stats['has_predecessor'] is True

    def test_concurrent_access(self):
        """Test thread-safe concurrent access"""
        import threading

        peer = self.peer

        def put_keys(start, end):
            for i in range(start, end):
                peer.put(i, f"value{i}")

        def get_keys(start, end):
            for i in range(start, end):
                peer.get(i)

        # Create threads
        threads = []
        threads.append(threading.Thread(target=put_keys, args=(0, 100)))
        threads.append(threading.Thread(target=put_keys, args=(100, 200)))
        threads.append(threading.Thread(target=get_keys, args=(0, 100)))

        # Start threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify all keys were inserted
        assert len(peer.storage) == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
