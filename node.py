"""
Physical node implementation for LEAD DHT
"""

import socket
import struct
import threading
import time
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .config import LEADConfig
from .models import RecursiveModelIndex
from .peer import LEADPeer, FingerEntry, KeyValuePair
from .utils import peer_hash, distance, RPCMessage
from .exceptions import NetworkException, RPCException, NodeNotReadyException

logger = logging.getLogger(__name__)


class LEADNode:
    """Physical node that hosts multiple virtual nodes"""
    
    def __init__(self, config: Optional[LEADConfig] = None):
        """
        Initialize a LEAD node
        
        Args:
            config: LEADConfig object with node configuration
        """
        self.config = config or LEADConfig()
        
        self.ip = self.config.ip
        self.base_port = self.config.base_port
        self.num_virtual_nodes = self.config.num_virtual_nodes
        self.bootstrap_peer = self.config.bootstrap_peer
        self.hash_space_size = self.config.hash_space_size
        
        # Virtual nodes
        self.virtual_nodes: Dict[int, LEADPeer] = {}
        
        # Learned hash function (shared across virtual nodes)
        self.learned_hash = RecursiveModelIndex(
            branching_factor=self.config.branching_factor,
            model_type=self.config.model_type,
            hash_space_size=self.hash_space_size
        )
        self.learned_hash_lock = threading.RLock()
        
        # Network
        self.server_socket = None
        self.running = False
        self.ready = False
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Stabilization
        self.stabilize_interval = self.config.stabilize_interval
        
    def start(self):
        """Start the node and join the network"""
        if self.running:
            logger.warning("Node already running")
            return
            
        logger.info(f"Starting LEAD node at {self.ip}:{self.base_port}")
        
        # Create virtual nodes
        for i in range(self.num_virtual_nodes):
            port = self.base_port + i
            vid = peer_hash(self.ip, port)
            vnode = LEADPeer(
                vid, self.ip, port, self,
                self.hash_space_size,
                self.config.model_update_threshold
            )
            self.virtual_nodes[vid] = vnode
            
        logger.info(f"Created {len(self.virtual_nodes)} virtual nodes")
        
        # Initialize learned hash with empty training
        self.learned_hash.train(np.array([]))
        
        # Start network server
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        # Join network
        if self.bootstrap_peer:
            self._join_network()
        else:
            # First node in network
            for vnode in self.virtual_nodes.values():
                vnode.successor = FingerEntry(vnode.vid, vnode.ip, vnode.port)
                vnode.predecessor = FingerEntry(vnode.vid, vnode.ip, vnode.port)
            logger.info("Initialized as first node in network")
                
        # Start stabilization
        self.stabilize_thread = threading.Thread(target=self._stabilize_loop, daemon=True)
        self.stabilize_thread.start()
        
        self.ready = True
        logger.info(f"Node started successfully")
        
    def stop(self):
        """Stop the node"""
        if not self.running:
            return
            
        logger.info("Stopping LEAD node")
        self.running = False
        self.ready = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
                
        self.executor.shutdown(wait=False)
        logger.info("Node stopped")
        
    def _join_network(self):
        """Join existing network via bootstrap peer"""
        logger.info(f"Joining network via bootstrap peer {self.bootstrap_peer}")
        
        for vnode in self.virtual_nodes.values():
            try:
                # Find successor via bootstrap
                successor = self.rpc_find_successor(
                    self.bootstrap_peer[0], self.bootstrap_peer[1], vnode.vid)
                    
                vnode.successor = successor
                vnode.predecessor = None
                
                # Update finger table
                vnode.update_finger_table()
                
                # Notify successor
                self.rpc_notify(successor.ip, successor.port,
                               FingerEntry(vnode.vid, vnode.ip, vnode.port))
                               
                # Request key transfer
                self._rpc_request_keys(successor.ip, successor.port, 
                                      successor.vid, vnode.vid)
                
                logger.debug(f"Virtual node {vnode.vid} joined network")
                
            except Exception as e:
                logger.error(f"Failed to join network for vnode {vnode.vid}: {e}")
                
    def _run_server(self):
        """Run RPC server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.ip, self.base_port))
            self.server_socket.listen(100)
            
            logger.info(f"RPC server listening on {self.ip}:{self.base_port}")
            
            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    self.executor.submit(self._handle_connection, conn)
                except Exception as e:
                    if self.running:
                        logger.error(f"Server accept error: {e}")
                    break
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.running = False
                
    def _handle_connection(self, conn: socket.socket):
        """Handle incoming RPC connection"""
        try:
            # Read header
            header = conn.recv(4)
            if len(header) < 4:
                return
                
            msg_len = struct.unpack('!I', header)[0]
            
            # Read message
            data = b''
            while len(data) < msg_len:
                chunk = conn.recv(min(msg_len - len(data), 4096))
                if not chunk:
                    break
                data += chunk
                
            msg_type, payload = RPCMessage.decode(data)
            
            # Route to handler
            response = self._handle_rpc(msg_type, payload)
            
            # Send response
            response_data = RPCMessage.encode('response', response)
            conn.sendall(response_data)
            
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            conn.close()
            
    def _handle_rpc(self, msg_type: str, payload: dict) -> dict:
        """Handle RPC request"""
        try:
            if msg_type == 'find_successor':
                key_hash = payload['key_hash']
                vnode = self._get_vnode_for_hash(key_hash)
                successor = vnode.find_successor(key_hash)
                
                return {
                    'success': True,
                    'successor': successor.to_dict()
                }
                
            elif msg_type == 'get_predecessor':
                vnode_vid = payload.get('vnode_vid')
                if vnode_vid and vnode_vid in self.virtual_nodes:
                    vnode = self.virtual_nodes[vnode_vid]
                else:
                    vnode = list(self.virtual_nodes.values())[0]
                
                if vnode.predecessor:
                    return {
                        'success': True,
                        'predecessor': vnode.predecessor.to_dict()
                    }
                return {'success': False}
                
            elif msg_type == 'notify':
                vnode_vid = payload.get('vnode_vid')
                potential_pred = FingerEntry.from_dict(payload['predecessor'])
                
                if vnode_vid and vnode_vid in self.virtual_nodes:
                    vnode = self.virtual_nodes[vnode_vid]
                    vnode.notify(potential_pred)
                else:
                    # Notify all virtual nodes
                    for vnode in self.virtual_nodes.values():
                        vnode.notify(potential_pred)
                    
                return {'success': True}
                
            elif msg_type == 'put':
                key = payload['key']
                value = payload['value']
                
                vnode = self._get_vnode_for_hash(key)
                vnode.put(key, value)
                
                return {'success': True}
                
            elif msg_type == 'get':
                key = payload['key']
                vnode = self._get_vnode_for_hash(key)
                value = vnode.get(key)
                
                return {'success': True, 'value': value}
                
            elif msg_type == 'get_range':
                start_key = payload['start_key']
                count = payload['count']
                
                vnode = self._get_vnode_for_hash(start_key)
                results = vnode.get_range(start_key, count)
                
                return {'success': True, 'results': results}
                
            elif msg_type == 'request_keys':
                vnode_vid = payload['vnode_vid']
                new_node_vid = payload['new_node_vid']
                
                if vnode_vid in self.virtual_nodes:
                    vnode = self.virtual_nodes[vnode_vid]
                    pred_vid = vnode.predecessor.vid if vnode.predecessor else 0
                    keys = vnode.transfer_keys(pred_vid, new_node_vid)
                    return {
                        'success': True,
                        'keys': [(kv.key, kv.value) for kv in keys]
                    }
                    
                return {'success': False}
                
            else:
                return {'success': False, 'error': 'Unknown message type'}
                
        except Exception as e:
            logger.error(f"Error handling RPC {msg_type}: {e}")
            return {'success': False, 'error': str(e)}
            
    def _get_vnode_for_hash(self, key_hash: int) -> LEADPeer:
        """Get the virtual node responsible for a hash"""
        # Find closest virtual node by VID
        closest_vnode = None
        min_distance = float('inf')
        
        for vnode in self.virtual_nodes.values():
            dist = distance(key_hash, vnode.vid, self.hash_space_size)
            if dist < min_distance:
                min_distance = dist
                closest_vnode = vnode
                
        return closest_vnode
        
    def _stabilize_loop(self):
        """Periodic stabilization loop"""
        while self.running:
            try:
                for vnode in self.virtual_nodes.values():
                    vnode.stabilize()
                    vnode.update_finger_table()
                    
                time.sleep(self.stabilize_interval)
            except Exception as e:
                logger.error(f"Stabilization error: {e}")
                
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def put(self, key: int, value: Any):
        """
        Store key-value pair in the DHT
        
        Args:
            key: Integer key
            value: Value to store (any serializable object)
        """
        if not self.ready:
            raise NodeNotReadyException("Node is not ready")
            
        # Hash key using learned hash function
        with self.learned_hash_lock:
            hash_value = self.learned_hash.predict(key)
            
        # Find responsible node
        vnode = self._get_vnode_for_hash(hash_value)
        successor = vnode.find_successor(hash_value)
        
        # Store locally or forward
        if successor.vid == vnode.vid:
            vnode.put(key, value)
        else:
            self.rpc_put(successor.ip, successor.port, key, value)
            
    def get(self, key: int) -> Optional[Any]:
        """
        Retrieve value by key
        
        Args:
            key: Integer key
            
        Returns:
            Value associated with key, or None if not found
        """
        if not self.ready:
            raise NodeNotReadyException("Node is not ready")
            
        with self.learned_hash_lock:
            hash_value = self.learned_hash.predict(key)
            
        vnode = self._get_vnode_for_hash(hash_value)
        successor = vnode.find_successor(hash_value)
        
        if successor.vid == vnode.vid:
            return vnode.get(key)
        else:
            return self.rpc_get(successor.ip, successor.port, key)
            
    def range_query(self, start_key: int, count: int) -> List[Tuple[int, Any]]:
        """
        Execute range query for sequential keys
        
        Args:
            start_key: Starting key for range
            count: Number of keys to retrieve
            
        Returns:
            List of (key, value) tuples
        """
        if not self.ready:
            raise NodeNotReadyException("Node is not ready")
            
        with self.learned_hash_lock:
            hash_value = self.learned_hash.predict(start_key)
            
        vnode = self._get_vnode_for_hash(hash_value)
        successor = vnode.find_successor(hash_value)
        
        results = []
        current_node = successor
        remaining = count
        
        # Limit iterations to prevent infinite loops
        max_iterations = self.num_virtual_nodes * 10
        iteration = 0
        
        while remaining > 0 and len(results) < count and iteration < max_iterations:
            iteration += 1
            
            # Get range from current node
            try:
                if current_node.vid == vnode.vid:
                    batch = vnode.get_range(start_key, remaining)
                else:
                    batch = self.rpc_get_range(current_node.ip, current_node.port, 
                                              start_key, remaining)
                    
                if not batch:
                    break
                    
                results.extend(batch)
                remaining -= len(batch)
                
                # Move to successor if we need more keys
                if remaining > 0:
                    if current_node.vid == vnode.vid:
                        if vnode.successor and vnode.successor.vid != vnode.vid:
                            current_node = vnode.successor
                        else:
                            break
                    else:
                        # Get successor of remote node
                        succ = self.rpc_find_successor(
                            current_node.ip, current_node.port,
                            (current_node.vid + 1) % self.hash_space_size
                        )
                        if succ.vid == current_node.vid:
                            break
                        current_node = succ
                        
                    # Update start_key for next batch
                    if results:
                        start_key = results[-1][0] + 1
            except Exception as e:
                logger.error(f"Error in range query: {e}")
                break
                    
        return results
        
    def retrain_model(self, sample_keys: Optional[np.ndarray] = None):
        """
        Retrain the learned hash function
        
        Args:
            sample_keys: Optional array of keys to train on. 
                        If None, collects keys from all virtual nodes
        """
        with self.learned_hash_lock:
            if sample_keys is None:
                # Collect keys from all virtual nodes
                all_keys = []
                for vnode in self.virtual_nodes.values():
                    with vnode.lock:
                        all_keys.extend(vnode.storage.keys())
                        
                if not all_keys:
                    logger.warning("No keys to train model on")
                    return
                    
                sample_keys = np.array(sorted(all_keys))
                
            self.learned_hash.train(sample_keys)
            
            # Reset new key counters
            for vnode in self.virtual_nodes.values():
                with vnode.lock:
                    vnode.new_keys_count = 0
                    vnode.update_ready = False
                    
            logger.info(f"Model retrained with {len(sample_keys)} keys, "
                       f"version {self.learned_hash.version}")
            
    def federated_model_update(self):
        """Perform federated model update across virtual nodes"""
        # Check if majority of virtual nodes are ready for update
        ready_count = sum(1 for vnode in self.virtual_nodes.values() 
                         if vnode.update_ready)
        
        if ready_count < len(self.virtual_nodes) * 0.9:
            logger.debug(f"Not enough nodes ready for update ({ready_count}/{len(self.virtual_nodes)})")
            return
            
        logger.info("Initiating federated model update")
        
        # Collect leaf model updates from local virtual nodes
        leaf_updates = defaultdict(list)
        
        for vnode in self.virtual_nodes.values():
            with vnode.lock:
                if not vnode.storage:
                    continue
                    
                # Get keys and their relative positions
                sorted_keys = sorted(vnode.storage.keys())
                
                for idx, key in enumerate(sorted_keys):
                    # Predict which leaf model this key belongs to
                    normalized_key = key / self.hash_space_size
                    bucket_prediction = self.learned_hash.stage1_model.predict(normalized_key)
                    bucket_id = int(np.clip(bucket_prediction, 0, 
                                           self.learned_hash.branching_factor - 1))
                    
                    # Calculate relative position for training
                    relative_pos = idx / len(sorted_keys)
                    leaf_updates[bucket_id].append((key, relative_pos))
                    
        # Update leaf models with aggregated data
        with self.learned_hash_lock:
            for bucket_id, key_pos_pairs in leaf_updates.items():
                if len(key_pos_pairs) > 2:  # Need minimum data points
                    keys = np.array([kp[0] for kp in key_pos_pairs])
                    positions = np.array([kp[1] for kp in key_pos_pairs])
                    self.learned_hash.update_leaf(bucket_id, keys, positions)
                    
            self.learned_hash.version += 1
            
        # Reset update flags
        for vnode in self.virtual_nodes.values():
            with vnode.lock:
                vnode.new_keys_count = 0
                vnode.update_ready = False
                
        logger.info(f"Federated model update complete, version {self.learned_hash.version}")
        
    def get_stats(self) -> dict:
        """
        Get node statistics
        
        Returns:
            Dictionary with node statistics
        """
        total_keys = 0
        vnode_stats = []
        
        for vnode in self.virtual_nodes.values():
            stats = vnode.get_stats()
            vnode_stats.append(stats)
            total_keys += stats['num_keys']
            
        return {
            'ip': self.ip,
            'base_port': self.base_port,
            'num_virtual_nodes': len(self.virtual_nodes),
            'total_keys': total_keys,
            'avg_keys_per_vnode': total_keys / len(self.virtual_nodes) if self.virtual_nodes else 0,
            'model_version': self.learned_hash.version,
            'running': self.running,
            'ready': self.ready,
            'virtual_nodes': vnode_stats
        }
        
    # ========================================================================
    # RPC CLIENT METHODS
    # ========================================================================
    
    def _send_rpc(self, ip: str, port: int, msg_type: str, payload: dict) -> dict:
        """Send RPC request and get response"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.rpc_timeout)
            sock.connect((ip, port))
            
            # Send request
            request = RPCMessage.encode(msg_type, payload)
            sock.sendall(request)
            
            # Receive response
            header = sock.recv(4)
            if len(header) < 4:
                return {'success': False, 'error': 'Invalid response'}
                
            msg_len = struct.unpack('!I', header)[0]
            
            data = b''
            while len(data) < msg_len:
                chunk = sock.recv(min(msg_len - len(data), 4096))
                if not chunk:
                    break
                data += chunk
                
            _, response = RPCMessage.decode(data)
            sock.close()
            
            return response
            
        except socket.timeout:
            raise RPCException(f"RPC timeout to {ip}:{port}")
        except Exception as e:
            raise RPCException(f"RPC error to {ip}:{port}: {e}")
            
    def rpc_find_successor(self, ip: str, port: int, key_hash: int) -> FingerEntry:
        """RPC: Find successor of a hash value"""
        response = self._send_rpc(ip, port, 'find_successor', 
                                 {'key_hash': key_hash})
                                 
        if response.get('success'):
            return FingerEntry.from_dict(response['successor'])
        else:
            raise RPCException(f"Failed to find successor: {response.get('error')}")
            
    def rpc_get_predecessor(self, ip: str, port: int) -> Optional[FingerEntry]:
        """RPC: Get predecessor of a node"""
        response = self._send_rpc(ip, port, 'get_predecessor', {})
        
        if response.get('success') and 'predecessor' in response:
            return FingerEntry.from_dict(response['predecessor'])
        return None
        
    def rpc_notify(self, ip: str, port: int, potential_pred: FingerEntry):
        """RPC: Notify node of potential predecessor"""
        payload = {'predecessor': potential_pred.to_dict()}
        self._send_rpc(ip, port, 'notify', payload)
        
    def rpc_put(self, ip: str, port: int, key: int, value: Any):
        """RPC: Store key-value pair"""
        response = self._send_rpc(ip, port, 'put', {'key': key, 'value': value})
        if not response.get('success'):
            raise RPCException(f"Failed to put key: {response.get('error')}")
            
    def rpc_get(self, ip: str, port: int, key: int) -> Optional[Any]:
        """RPC: Retrieve value by key"""
        response = self._send_rpc(ip, port, 'get', {'key': key})
        if response.get('success'):
            return response.get('value')
        return None
        
    def rpc_get_range(self, ip: str, port: int, start_key: int, 
                     count: int) -> List[Tuple[int, Any]]:
        """RPC: Get range of keys"""
        response = self._send_rpc(ip, port, 'get_range', 
                                 {'start_key': start_key, 'count': count})
        if response.get('success'):
            return response.get('results', [])
        return []
        
    def _rpc_request_keys(self, ip: str, port: int, vnode_vid: int, new_node_vid: int):
        """RPC: Request key transfer from successor"""
        response = self._send_rpc(ip, port, 'request_keys', 
                                 {'vnode_vid': vnode_vid, 'new_node_vid': new_node_vid})
        if response.get('success'):
            keys = response.get('keys', [])
            # Store transferred keys locally
            for key, value in keys:
                self.put(key, value)