"""
Federated Learning Server

This module provides a server implementation for federated learning.
"""

import os
import sys
import socket
import threading
import json
import time
import numpy as np
from datetime import datetime

class FederatedServer:
    """
    Federated Learning Server implementation.
    
    This server coordinates federated learning across multiple clients,
    aggregating model updates and distributing the global model.
    """
    
    def __init__(self, model, host='0.0.0.0', port=8080, min_clients=2, rounds=5):
        """
        Initialize the Federated Learning Server.
        
        Args:
            model: The initial model (TensorFlow model).
            host: Host to bind the server to.
            port: Port to listen on.
            min_clients: Minimum number of clients required for training.
            rounds: Number of federated learning rounds.
        """
        self.model = model
        self.host = host
        self.port = port
        self.min_clients = min_clients
        self.rounds = rounds
        
        # Server state
        self.running = False
        self.server_socket = None
        self.current_round = 0
        
        # Client management
        self.clients = {}  # Maps client_id to socket connection
        self.client_data_sizes = {}  # Maps client_id to data size
        
        # Round management
        self.round_updates = {}  # Maps client_id to model updates for current round
        self.round_metrics = []  # Metrics for each round
        
        # Locks for thread safety
        self.clients_lock = threading.Lock()
        self.updates_lock = threading.Lock()
    
    def start(self, round_callback=None, final_callback=None):
        """
        Start the federated learning server.
        
        Args:
            round_callback: Callback function called at the end of each round.
            final_callback: Callback function called at the end of training.
        """
        # Set up server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        
        print(f"Server listening on {self.host}:{self.port}")
        print(f"Waiting for at least {self.min_clients} clients to connect")
        
        # Start the server
        self.running = True
        
        # Start a thread to accept client connections
        accept_thread = threading.Thread(target=self._accept_clients)
        accept_thread.daemon = True
        accept_thread.start()
        
        # Wait for minimum number of clients
        while self.running and len(self.clients) < self.min_clients:
            print(f"Waiting for clients: {len(self.clients)}/{self.min_clients} connected")
            time.sleep(5)
        
        print(f"{len(self.clients)} clients connected. Starting federated learning.")
        
        # Get initial model weights
        global_weights = self.model.get_weights()
        
        # Run federated learning rounds
        for round_num in range(1, self.rounds + 1):
            self.current_round = round_num
            print(f"\nStarting round {round_num} of {self.rounds}")
            
            # Reset round updates
            with self.updates_lock:
                self.round_updates = {}
            
            # Broadcast model for training
            self._broadcast_train(global_weights, round_num)
            
            # Wait for updates from clients
            timeout = time.time() + 300  # 5 minutes timeout
            while len(self.round_updates) < len(self.clients) and time.time() < timeout:
                print(f"Waiting for updates: {len(self.round_updates)}/{len(self.clients)} received")
                time.sleep(5)
            
            if not self.round_updates:
                print("No updates received this round, using previous model")
                continue
            
            # Perform federated averaging
            global_weights = self._federated_averaging()
            
            # Update global model
            self.model.set_weights(global_weights)
            
            # Calculate round metrics (average of client metrics)
            metrics = self._calculate_round_metrics()
            self.round_metrics.append(metrics)
            
            print(f"Round {round_num} completed")
            print(f"Metrics: {metrics}")
            
            # Call round callback if provided
            if round_callback:
                round_callback(round_num, self.model, metrics)
        
        # Training complete
        print("\nFederated learning complete!")
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics()
        print(f"Final metrics: {final_metrics}")
        
        # Call final callback if provided
        if final_callback:
            final_callback(self.model, final_metrics)
        
        # Notify clients that training is complete
        self._broadcast_complete()
        
        # Shutdown the server
        self.shutdown()
    
    def _accept_clients(self):
        """Accept client connections in a separate thread."""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                print(f"Client connected from {client_address[0]}:{client_address[1]}")
                
                # Start a thread to handle this client
                client_thread = threading.Thread(target=self._handle_client, args=(client_socket,))
                client_thread.daemon = True
                client_thread.start()
            except:
                break
    
    def _handle_client(self, client_socket):
        """
        Handle communication with a client.
        
        Args:
            client_socket: Socket connection to the client.
        """
        client_id = None
        
        try:
            # Wait for registration message
            message = self._receive_message(client_socket)
            if not message or message.get('type') != 'register':
                print("Expected registration message, closing connection")
                client_socket.close()
                return
            
            # Extract client information
            client_id = message.get('client_id')
            data_size = message.get('data_size', 0)
            
            print(f"Client {client_id} registered with {data_size} samples")
            
            # Register the client
            with self.clients_lock:
                self.clients[client_id] = client_socket
                self.client_data_sizes[client_id] = data_size
            
            # Handle client messages
            while self.running:
                message = self._receive_message(client_socket)
                if not message:
                    break
                
                self._handle_message(client_id, message)
        
        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
        
        finally:
            # Remove client on disconnect
            if client_id and client_id in self.clients:
                with self.clients_lock:
                    del self.clients[client_id]
                    if client_id in self.client_data_sizes:
                        del self.client_data_sizes[client_id]
                
                print(f"Client {client_id} disconnected")
            
            try:
                client_socket.close()
            except:
                pass
    
    def _handle_message(self, client_id, message):
        """
        Handle a message from a client.
        
        Args:
            client_id: ID of the client sending the message.
            message: The message to handle.
        """
        message_type = message.get('type')
        
        if message_type == 'update':
            # Client is sending model updates
            weights = message.get('weights')
            metrics = message.get('metrics', {})
            round_num = message.get('round')
            
            if round_num != self.current_round:
                print(f"Ignoring update from client {client_id} for round {round_num} (current round: {self.current_round})")
                return
            
            print(f"Received model update from client {client_id} for round {round_num}")
            
            # Convert weights from list to numpy arrays
            weights = [np.array(w) for w in weights]
            
            # Store update
            with self.updates_lock:
                self.round_updates[client_id] = {
                    'weights': weights,
                    'metrics': metrics
                }
        
        elif message_type == 'eval_result':
            # Client is sending evaluation results
            metrics = message.get('metrics', {})
            print(f"Received evaluation results from client {client_id}: {metrics}")
    
    def _broadcast_train(self, weights, round_num):
        """
        Broadcast the current model for training.
        
        Args:
            weights: Model weights to send.
            round_num: Current round number.
        """
        with self.clients_lock:
            client_ids = list(self.clients.keys())
        
        for client_id in client_ids:
            try:
                # Convert numpy arrays to lists for JSON serialization
                weights_list = [w.tolist() for w in weights]
                
                # Send training request
                message = {
                    'type': 'train',
                    'weights': weights_list,
                    'round': round_num
                }
                
                self._send_message(client_id, message)
                print(f"Sent model to client {client_id} for round {round_num}")
            
            except Exception as e:
                print(f"Error sending model to client {client_id}: {e}")
    
    def _broadcast_complete(self):
        """Broadcast that training is complete."""
        with self.clients_lock:
            client_ids = list(self.clients.keys())
        
        for client_id in client_ids:
            try:
                message = {'type': 'complete'}
                self._send_message(client_id, message)
                print(f"Sent completion message to client {client_id}")
            except Exception as e:
                print(f"Error sending completion to client {client_id}: {e}")
    
    def _federated_averaging(self):
        """
        Perform federated averaging on client updates.
        
        Returns:
            Updated global model weights.
        """
        with self.updates_lock:
            if not self.round_updates:
                # No updates received, return current model weights
                return self.model.get_weights()
            
            # Get all client updates
            client_weights = {}
            for client_id, update in self.round_updates.items():
                client_weights[client_id] = update['weights']
        
        # Get the total data size for weighted averaging
        total_data_size = sum(self.client_data_sizes.values())
        
        if total_data_size == 0:
            # Equal weighting if no data size information
            weights = list(client_weights.values())
            avg_weights = [np.mean(np.array([w[i] for w in weights]), axis=0) for i in range(len(weights[0]))]
        else:
            # Weighted averaging based on data size
            avg_weights = None
            for client_id, weights in client_weights.items():
                weight = self.client_data_sizes.get(client_id, 0) / total_data_size
                
                if avg_weights is None:
                    avg_weights = [w * weight for w in weights]
                else:
                    for i, w in enumerate(weights):
                        avg_weights[i] += w * weight
        
        return avg_weights
    
    def _calculate_round_metrics(self):
        """
        Calculate metrics for the current round.
        
        Returns:
            dict: Metrics for the current round.
        """
        metrics = {}
        
        with self.updates_lock:
            if not self.round_updates:
                return metrics
            
            # Aggregate metrics from client updates
            for client_id, update in self.round_updates.items():
                client_metrics = update.get('metrics', {})
                
                for key, value in client_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    
                    if isinstance(value, (int, float)):
                        metrics[key].append(value)
        
        # Calculate averages
        for key, values in metrics.items():
            if values:
                metrics[key] = sum(values) / len(values)
        
        return metrics
    
    def _calculate_final_metrics(self):
        """
        Calculate final metrics after all rounds.
        
        Returns:
            dict: Final metrics.
        """
        if not self.round_metrics:
            return {}
        
        # Get the last round metrics
        return self.round_metrics[-1]
    
    def _send_message(self, client_id, message):
        """
        Send a message to a client.
        
        Args:
            client_id: ID of the client to send the message to.
            message: The message to send.
        """
        client_socket = None
        
        with self.clients_lock:
            if client_id in self.clients:
                client_socket = self.clients[client_id]
        
        if not client_socket:
            raise Exception(f"Client {client_id} not found")
        
        # Convert message to JSON and encode as bytes
        message_bytes = json.dumps(message).encode('utf-8')
        
        # Send message length first, then the message
        message_length = len(message_bytes)
        client_socket.sendall(message_length.to_bytes(4, byteorder='big'))
        client_socket.sendall(message_bytes)
    
    def _receive_message(self, client_socket):
        """
        Receive a message from a client socket.
        
        Args:
            client_socket: Socket to receive from.
            
        Returns:
            dict: The received message, or None if there was an error.
        """
        try:
            # Receive message length first (4 bytes)
            length_bytes = client_socket.recv(4)
            if not length_bytes:
                return None
            
            message_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Receive the full message
            message_bytes = b''
            while len(message_bytes) < message_length:
                chunk = client_socket.recv(min(4096, message_length - len(message_bytes)))
                if not chunk:
                    return None
                message_bytes += chunk
            
            # Decode and parse the message
            message = json.loads(message_bytes.decode('utf-8'))
            return message
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None
    
    def shutdown(self):
        """Shutdown the server."""
        self.running = False
        
        # Close all client connections
        with self.clients_lock:
            for client_id, client_socket in self.clients.items():
                try:
                    client_socket.close()
                except:
                    pass
            
            self.clients = {}
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None 