"""
Physical node implementation for LEAD DHT
"""

import socket
import struct
import threading
import time
import logging
import hashlib
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
            hash_space_size=self.hash_space_size,
            stage1_model_type=self.config.stage1_model_type
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
        # All virtual nodes share the same port (base_port) since they're logical entities
        # on the same physical node. VID is generated using a salt to distribute them in hash space.
        for i in range(self.num_virtual_nodes):
            # Generate unique VID for each virtual node using salt
            vid_data = f"{self.ip}:{self.base_port}:vnode{i}".encode()
            vid = int.from_bytes(hashlib.sha1(vid_data).digest(), 'big')

            vnode = LEADPeer(
                vid, self.ip, self.base_port, self,
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

        # Give server time to start listening (especially important in Docker)
        time.sleep(0.5)

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

        # Retry joining if bootstrap peer isn't ready yet
        max_retries = 5
        retry_delay = 2  # seconds

        for vnode in self.virtual_nodes.values():
            for attempt in range(max_retries):
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

                    logger.info(f"Virtual node {vnode.vid} successfully joined network")
                    break  # Success, exit retry loop

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Join attempt {attempt + 1}/{max_retries} failed for vnode {vnode.vid}: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to join network for vnode {vnode.vid} after {max_retries} attempts: {e}")
                
    def _run_server(self):
        """Run RPC server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind to 0.0.0.0 to accept connections from all interfaces (required for Docker)
            bind_ip = '0.0.0.0'
            self.server_socket.bind((bind_ip, self.base_port))
            self.server_socket.listen(100)

            logger.info(f"RPC server listening on {bind_ip}:{self.base_port} (advertised as {self.ip}:{self.base_port})")
            
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
            logger.debug(f"Handling incoming RPC connection")
            # Read header
            header = conn.recv(4)
            logger.debug(f"Received header: {len(header)} bytes")
            if len(header) < 4:
                logger.warning("Incomplete header received")
                return

            msg_len = struct.unpack('!I', header)[0]
            logger.debug(f"Expecting message of {msg_len} bytes")

            # Read message
            data = b''
            while len(data) < msg_len:
                chunk = conn.recv(min(msg_len - len(data), 4096))
                if not chunk:
                    break
                data += chunk
            logger.debug(f"Received {len(data)} bytes of message data")

            msg_type, payload = RPCMessage.decode(data)
            logger.debug(f"Decoded RPC request: {msg_type}")

            # Route to handler
            response = self._handle_rpc(msg_type, payload)
            logger.debug(f"Generated response for {msg_type}")

            # Send response
            response_data = RPCMessage.encode('response', response)
            logger.debug(f"Sending response: {len(response_data)} bytes")
            conn.sendall(response_data)
            logger.debug(f"Response sent successfully for {msg_type}")

        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
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

            elif msg_type == 'get_successor':
                vnode_vid = payload.get('vnode_vid')
                if vnode_vid and vnode_vid in self.virtual_nodes:
                    vnode = self.virtual_nodes[vnode_vid]
                else:
                    vnode = list(self.virtual_nodes.values())[0]

                if vnode.successor:
                    return {
                        'success': True,
                        'successor': vnode.successor.to_dict()
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
        if successor.ip == self.ip and successor.port == self.base_port:
            # Successor is a local vnode
            if successor.vid in self.virtual_nodes:
                self.virtual_nodes[successor.vid].put(key, value)
            else:
                # Shouldn't happen, but fallback to original vnode
                vnode.put(key, value)
        else:
            # Successor is on a different node
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

        logger.debug(f"get({key}): predicted hash={hash_value}")

        vnode = self._get_vnode_for_hash(hash_value)
        successor = vnode.find_successor(hash_value)

        logger.debug(f"get({key}): vnode={vnode.vid}, successor={successor.vid} at {successor.ip}:{successor.port}")

        # Check if successor is a local vnode (same IP and port as this node)
        if successor.ip == self.ip and successor.port == self.base_port:
            logger.debug(f"get({key}): successor is local, checking local vnodes")

            # Try to get from the specific successor vnode first
            if successor.vid in self.virtual_nodes:
                result = self.virtual_nodes[successor.vid].get(key)
                logger.debug(f"get({key}): local vnode {successor.vid} returned {result}")
                if result is not None:
                    return result

            # Fallback: check all local vnodes (handles case where model was retrained)
            logger.debug(f"get({key}): checking all {len(self.virtual_nodes)} local vnodes")
            for vid, local_vnode in self.virtual_nodes.items():
                value = local_vnode.get(key)
                logger.debug(f"get({key}): checking vnode {vid}, got {value}")
                if value is not None:
                    logger.info(f"get({key}): found in local vnode {vid}")
                    return value

            logger.warning(f"get({key}): not found in any local vnode")
            return None
        else:
            # Successor is on a different node - use RPC
            logger.debug(f"get({key}): trying RPC to remote successor {successor.ip}:{successor.port}")
            try:
                result = self.rpc_get(successor.ip, successor.port, key)
                logger.debug(f"get({key}): RPC returned {result}")
                return result
            except (RPCException, Exception) as e:
                logger.error(f"get({key}): RPC failed with {e}")
                return None
            
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
                logger.info(f"Collecting keys from {len(self.virtual_nodes)} virtual nodes")
                
                for vnode_id, vnode in self.virtual_nodes.items():
                    with vnode.lock:
                        storage_keys = list(vnode.storage.keys())
                        logger.info(f"VNode {vnode_id}: found {len(storage_keys)} keys")
                        
                        # Convert keys to numeric format
                        for key in storage_keys:
                            try:
                                numeric_key = float(key)
                                all_keys.append(numeric_key)
                            except (ValueError, TypeError):
                                logger.warning(f"Skipping non-numeric key: {key}")
                                continue
                        
                logger.info(f"Total keys collected: {len(all_keys)}")
                
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
    # def retrain_model(self, sample_keys: Optional[np.ndarray] = None):
    #     """
    #     Retrain the learned hash function
        
    #     Args:
    #         sample_keys: Optional array of keys to train on. 
    #                     If None, collects keys from all virtual nodes
    #     """
    #     with self.learned_hash_lock:
    #         if sample_keys is None:
    #             # Collect keys from all virtual nodes
    #             all_keys = []
    #             for vnode in self.virtual_nodes.values():
    #                 with vnode.lock:
    #                     all_keys.extend(vnode.storage.keys())
                        
    #             if not all_keys:
    #                 logger.warning("No keys to train model on")
    #                 return
                    
    #             sample_keys = np.array(sorted(all_keys))
                
    #         self.learned_hash.train(sample_keys)
            
    #         # Reset new key counters
    #         for vnode in self.virtual_nodes.values():
    #             with vnode.lock:
    #                 vnode.new_keys_count = 0
    #                 vnode.update_ready = False
                    
    #         logger.info(f"Model retrained with {len(sample_keys)} keys, "
    #                    f"version {self.learned_hash.version}")
            
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
            logger.debug(f"Sending RPC {msg_type} to {ip}:{port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.rpc_timeout)

            logger.debug(f"Connecting to {ip}:{port}...")
            sock.connect((ip, port))
            logger.debug(f"Connected to {ip}:{port}")

            # Send request
            request = RPCMessage.encode(msg_type, payload)
            logger.debug(f"Sending request of {len(request)} bytes")
            sock.sendall(request)
            logger.debug(f"Request sent, waiting for response...")

            # Receive response
            header = sock.recv(4)
            logger.debug(f"Received header: {len(header)} bytes")
            if len(header) < 4:
                return {'success': False, 'error': 'Invalid response'}

            msg_len = struct.unpack('!I', header)[0]
            logger.debug(f"Expecting response of {msg_len} bytes")

            data = b''
            while len(data) < msg_len:
                chunk = sock.recv(min(msg_len - len(data), 4096))
                if not chunk:
                    break
                data += chunk
            logger.debug(f"Received {len(data)} bytes of data")

            _, response = RPCMessage.decode(data)
            sock.close()
            logger.debug(f"RPC {msg_type} completed successfully")

            return response

        except socket.timeout:
            logger.error(f"RPC timeout to {ip}:{port} for {msg_type}")
            raise RPCException(f"RPC timeout to {ip}:{port}")
        except Exception as e:
            logger.error(f"RPC error to {ip}:{port} for {msg_type}: {e}")
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

    def rpc_get_successor(self, ip: str, port: int) -> Optional[FingerEntry]:
        """RPC: Get successor of a node"""
        response = self._send_rpc(ip, port, 'get_successor', {})

        if response.get('success') and 'successor' in response:
            return FingerEntry.from_dict(response['successor'])
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