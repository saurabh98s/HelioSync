"""
Federated Learning Server

This module provides a server implementation for federated learning.
"""

import os
import sys
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
    
    def __init__(self, model, host='localhost', port=5000, min_clients=2, rounds=10):
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
        self.current_round = 0
        
        # Client management
        self.clients = set()  # Set of connected client IDs
        self.client_data_sizes = {}  # Maps client_id to data size
        
        # Round management
        self.round_updates = {}  # Maps client_id to model updates for current round
        self.round_metrics = {}  # Metrics for each round
        
        # Client status management
        self.client_statuses = {}  # Maps client_id to client status
        
        # Locks for thread safety
        self.clients_lock = threading.Lock()
        self.updates_lock = threading.Lock()
        self.weights_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.status_lock = threading.Lock()
    
    def add_client(self, client_id, data_size):
        """Add a client to the server."""
        with self.clients_lock:
            print(f"\nAdding client {client_id} with {data_size} samples")
            
            # Remove any existing client with the same ID
            if client_id in self.clients:
                print(f"Client {client_id} already exists, updating registration")
                self.clients.remove(client_id)
                if client_id in self.client_data_sizes:
                    del self.client_data_sizes[client_id]
                if client_id in self.client_statuses:
                    del self.client_statuses[client_id]
            
            # Add the client
            self.clients.add(client_id)
            self.client_data_sizes[client_id] = data_size
            self.client_statuses[client_id] = 'ready'  # Initialize client status as ready
            
            print(f"Current clients ({len(self.clients)}):")
            for cid in self.clients:
                print(f"- {cid} (status: {self.client_statuses.get(cid, 'unknown')})")
    
    def remove_client(self, client_id):
        """Remove a client from the server."""
        with self.clients_lock:
            print(f"\nRemoving client {client_id}")
            if client_id in self.clients:
                self.clients.remove(client_id)
                if client_id in self.client_data_sizes:
                    del self.client_data_sizes[client_id]
                if client_id in self.client_statuses:
                    del self.client_statuses[client_id]
                print(f"Client {client_id} removed")
                print(f"Current clients: {self.clients}")
            else:
                print(f"Client {client_id} not found")
    
    def get_model_weights(self):
        """Get the current model weights."""
        return self.model.get_weights()
    
    def get_client_status(self, client_id):
        """Get current status of a client."""
        with self.status_lock:
            status = self.client_statuses.get(client_id, 'wait')
            print(f"Getting status for client {client_id}: {status}")
            return status
    
    def update_client_status(self, client_id, status):
        """Update status of a client."""
        with self.status_lock:
            print(f"Updating status for client {client_id} to {status}")
            self.client_statuses[client_id] = status
            print(f"Current client statuses: {self.client_statuses}")
    
    def update_model(self, client_id, weights, metrics):
        """Update model with client's weights and metrics."""
        if not isinstance(metrics, dict):
            metrics = {}  # Use empty dict if metrics is None or invalid
        
        with self.updates_lock:
            # Store the update
            self.round_updates[client_id] = {
                'weights': weights,
                'metrics': metrics
            }
            
            # Update round metrics
            if self.current_round not in self.round_metrics:
                self.round_metrics[self.current_round] = []
            
            # Check if we already have metrics for this client in this round
            existing_metric = None
            for i, m in enumerate(self.round_metrics[self.current_round]):
                if isinstance(m, dict) and m.get('client_id') == client_id:
                    existing_metric = i
                    break
            
            # Create metrics entry with default values if fields are missing
            metric_entry = {
                'client_id': client_id,
                'round': self.current_round,
                'epoch': metrics.get('epoch', 0),
                'total_epochs': metrics.get('total_epochs', 0),
                'loss': float(metrics.get('loss', 0.0)),
                'accuracy': float(metrics.get('accuracy', 0.0)),
                'val_loss': float(metrics.get('val_loss', 0.0)),
                'val_accuracy': float(metrics.get('val_accuracy', 0.0)),
                'samples': int(metrics.get('samples', 0)),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Update or append the metrics
            if existing_metric is not None:
                self.round_metrics[self.current_round][existing_metric] = metric_entry
            else:
                self.round_metrics[self.current_round].append(metric_entry)
            
            print(f"\nUpdated metrics for client {client_id} in round {self.current_round}")
            print(f"Current metrics: {metric_entry}")
            
            # Check if we have enough updates to proceed to next round
            if len(self.round_updates) >= len(self.clients):
                self.train_round()
    
    def aggregate_updates(self):
        """Aggregate updates from all clients using FedAvg."""
        if not self.round_updates:
            return
        
        # Get the first client's weights shape
        first_client = next(iter(self.round_updates))
        weights_shape = [w.shape for w in self.round_updates[first_client]['weights']]
        
        # Initialize aggregated weights
        aggregated_weights = [np.zeros(shape) for shape in weights_shape]
        total_samples = 0
        
        # Sum up the weighted updates
        for client_id, update in self.round_updates.items():
            client_samples = self.client_data_sizes.get(client_id, 1)
            total_samples += client_samples
            
            for i, w in enumerate(update['weights']):
                aggregated_weights[i] += w * client_samples
        
        # Average the weights
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] /= total_samples
        
        # Update the model
        self.model.set_weights(aggregated_weights)
        
        # Record metrics
        round_metrics = {
            'round': self.current_round,
            'clients': len(self.round_updates),
            'metrics': {
                client_id: update['metrics']
                for client_id, update in self.round_updates.items()
            }
        }
        self.round_metrics[self.current_round].append(round_metrics)
        
        # Clear round updates
        self.round_updates = {}
    
    def get_metrics(self):
        """Get current training metrics."""
        with self.metrics_lock:
            print("\nGetting current metrics")
            print(f"Connected clients: {len(self.clients)}")
            print(f"Client statuses: {self.client_statuses}")
            
            # Calculate aggregated metrics per round
            aggregated_metrics = {}
            for round_num, round_metrics in self.round_metrics.items():
                if not round_metrics:
                    continue
                
                # Initialize metrics for this round
                round_agg = {
                    'loss': 0.0,
                    'accuracy': 0.0,
                    'val_loss': 0.0,
                    'val_accuracy': 0.0,
                    'total_samples': 0,
                    'num_clients': len([m for m in round_metrics if isinstance(m, dict) and 'client_id' in m])
                }
                
                # Sum up metrics weighted by sample size
                total_samples = 0
                for metric in round_metrics:
                    if not isinstance(metric, dict) or 'samples' not in metric:
                        continue
                        
                    samples = metric.get('samples', 0)
                    if samples > 0:
                        total_samples += samples
                        round_agg['loss'] += metric.get('loss', 0.0) * samples
                        round_agg['accuracy'] += metric.get('accuracy', 0.0) * samples
                        round_agg['val_loss'] += metric.get('val_loss', 0.0) * samples
                        round_agg['val_accuracy'] += metric.get('val_accuracy', 0.0) * samples
                
                # Calculate weighted averages
                if total_samples > 0:
                    for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                        round_agg[key] /= total_samples
                    
                    round_agg['total_samples'] = total_samples
                    aggregated_metrics[round_num] = round_agg

            metrics = {
                'current_round': self.current_round,
                'total_rounds': self.rounds,
                'connected_clients': len(self.clients),
                'min_clients': self.min_clients,
                'round_metrics': self.round_metrics,
                'client_statuses': self.client_statuses,
                'aggregated_metrics': aggregated_metrics,
                'clients': list(self.clients)  # Add list of client IDs
            }
            
            print(f"Aggregated metrics: {aggregated_metrics}")
            return metrics
    
    def start(self, round_callback=None, final_callback=None):
        """
        Start the federated learning server.
        
        Args:
            round_callback: Callback function called at the end of each round.
            final_callback: Callback function called at the end of training.
        """
        print(f"Server starting. Waiting for at least {self.min_clients} clients to connect")
        
        # Start the server
        self.running = True
        
        # Initialize round metrics
        self.round_metrics = {}
        self.current_round = 0
        
        print(f"Server ready to accept clients")
        return True
    
    def shutdown(self):
        """Shutdown the server."""
        self.running = False
        print("Server shutting down")
        
        # Clear all client data
        with self.clients_lock:
            self.clients.clear()
            self.client_data_sizes.clear()
            self.client_statuses.clear()
        
        # Clear all round data
        with self.updates_lock:
            self.round_updates.clear()
            self.round_metrics.clear()
        
        print("Server shutdown complete")
    
    def train_round(self):
        """Train one round of federated learning."""
        if not self.running:
            return False
        
        print(f"\nStarting round {self.current_round + 1}/{self.rounds}")
        
        # Check if we have minimum number of clients
        if len(self.clients) < self.min_clients:
            print(f"Not enough clients connected. Need {self.min_clients}, have {len(self.clients)}")
            return False
        
        # Calculate minimum updates needed to proceed
        min_updates_needed = max(int(len(self.clients) * 0.5), self.min_clients)  # At least 50% of clients or min_clients
        print(f"Need at least {min_updates_needed} client updates to proceed with round")
        
        # Update all client statuses to 'training'
        with self.status_lock:
            for client_id in self.clients:
                self.client_statuses[client_id] = 'training'
                print(f"Set client {client_id} status to training")
        
        # Wait for updates from enough clients
        start_time = time.time()
        while len(self.round_updates) < min_updates_needed:
            # If we've waited a long time but have some updates, proceed anyway
            if time.time() - start_time > 300 and len(self.round_updates) > 0:  # 5 minutes timeout
                print(f"Round timeout - proceeding with {len(self.round_updates)} updates (wanted {min_updates_needed})")
                break
                
            # If we've waited even longer and still have no updates, abort round
            if time.time() - start_time > 600:  # 10 minutes timeout
                print("Round complete timeout with no updates - aborting round")
                return False
                
            time.sleep(1)
        
        # Aggregate updates if we have any
        if self.round_updates:
            print(f"Aggregating updates from {len(self.round_updates)} clients")
            self.aggregate_updates()
            self.current_round += 1
            
            # Save aggregated model
            try:
                # Get the weights of the aggregated model
                weights = self.model.get_weights()
                
                # Save the aggregated model to disk
                model_save_path = f"models/federated_round_{self.current_round}.h5"
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                self.model.save(model_save_path)
                print(f"Saved aggregated model to {model_save_path}")
            except Exception as e:
                print(f"Failed to save aggregated model: {e}")
            
            # Update client statuses to 'ready'
            with self.status_lock:
                for client_id in self.clients:
                    self.client_statuses[client_id] = 'ready'
                    print(f"Set client {client_id} status to ready")
            
            # Check if we've reached the final round
            if self.current_round >= self.rounds:
                print(f"Completed all {self.rounds} rounds of training")
                # Save final model
                try:
                    final_model_path = "models/federated_final.h5"
                    self.model.save(final_model_path)
                    print(f"Saved final model to {final_model_path}")
                except Exception as e:
                    print(f"Failed to save final model: {e}")
            
            return True
        
        print("No updates received this round")
        return False 