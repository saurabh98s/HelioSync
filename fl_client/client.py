"""
Federated Learning Client

This module provides a client implementation for federated learning.
"""

import os
import sys
import socket
import threading
import json
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import requests
import platform

class FederatedClient:
    """
    Federated Learning Client implementation.
    
    This client communicates with a Federated Learning server and performs
    local training on its own data.
    """
    
    def __init__(self, client_id, model, x_train, y_train, x_test=None, y_test=None, 
                 batch_size=32, epochs=5, api_key=None, server_url="http://localhost:5000"):
        """
        Initialize the Federated Learning Client.
        
        Args:
            client_id: Unique identifier for this client.
            model: TensorFlow model for training.
            x_train: Training data features.
            y_train: Training data labels.
            x_test: Test data features (optional).
            y_test: Test data labels (optional).
            batch_size: Batch size for local training.
            epochs: Number of local epochs.
            api_key: API key for authentication.
            server_url: URL of the federated learning server.
        """
        self.client_id = client_id
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.api_key = api_key
        self.server_url = server_url
        self.is_training = False
        self.current_round = 0
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': None,
            'test_accuracy': None,
            'training_time': 0
        }
        
        # Client state
        self.running = False
        self.connected = False
        self.client_thread = None
        
        # Compile model if not already compiled
        if not self.model.optimizer:
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def start(self):
        """Start the federated learning client."""
        self.is_training = True
        self._register_client()
        self._training_loop()
    
    def stop(self):
        """Stop the federated learning client."""
        self.is_training = False
    
    def _register_client(self):
        """Register the client with the server."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"\nAttempting to register client {self.client_id} (attempt {attempt + 1}/{max_retries})")
                
                # Prepare registration data with more details
                registration_data = {
                    'client_id': self.client_id,
                    'name': f'Client-{self.client_id}',
                    'data_size': len(self.x_train),
                    'device_info': f'Python Client - TensorFlow {tf.__version__}',
                    'platform': platform.system(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'python_version': platform.python_version(),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                print(f"Sending registration data: {registration_data}")
                
                # Send registration request
                response = requests.post(
                    f"{self.server_url}/api/register_client",
                    json=registration_data,
                    headers={
                        'X-API-Key': self.api_key,
                        'Content-Type': 'application/json'
                    }
                )
                
                # Check response
                response.raise_for_status()
                response_data = response.json()
                
                if response_data.get('status') == 'success':
                    print(f"Client {self.client_id} registered successfully")
                    print(f"Server response: {response_data}")
                    
                    # Start heartbeat thread
                    self._start_heartbeat()
                    return True
                else:
                    print(f"Registration failed: {response_data.get('error', 'Unknown error')}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Network error during registration: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Registration failed.")
                    self.stop()
                    return False
                    
            except Exception as e:
                print(f"Unexpected error during registration: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Registration failed.")
                    self.stop()
                    return False
        
        return False
    
    def _start_heartbeat(self):
        """Start sending heartbeat to server."""
        def heartbeat():
            while self.is_training:
                try:
                    response = requests.post(
                        f"{self.server_url}/api/clients/{self.client_id}/heartbeat",
                        headers={'X-API-Key': self.api_key}
                    )
                    response.raise_for_status()
                except:
                    pass
                time.sleep(30)  # Send heartbeat every 30 seconds
        
        self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self.heartbeat_thread.start()
    
    def _training_loop(self):
        """Main training loop."""
        while self.is_training:
            try:
                # Get current model weights from server
                response = requests.get(
                    f"{self.server_url}/api/clients/{self.client_id}/tasks",
                    headers={'X-API-Key': self.api_key}
                )
                response.raise_for_status()
                data = response.json()

                # Handle different status responses from server
                status = data.get('status')
                
                if status == 'error':
                    print(f"Error from server: {data.get('message', 'Unknown error')}")
                    if 'details' in data and isinstance(data['details'], dict):
                        print(f"Details: {data['details']}")
                    time.sleep(5)
                    continue
                    
                elif status == 'waiting':
                    print(f"Waiting: {data.get('message', 'No tasks available')}")
                    if 'details' in data and isinstance(data['details'], dict):
                        print(f"Details: {data['details']}")
                    time.sleep(5)
                    continue
                    
                elif status == 'training':
                    print(f"Training task received: {data.get('message', '')}")
                    details = data.get('details', {})
                    
                    # Extract training details
                    project_id = details.get('project_id')
                    project_name = details.get('project_name')
                    current_round = details.get('round', 0)
                    total_rounds = details.get('total_rounds', 0)
                    weights = details.get('weights', [])
                    
                    print(f"Training for project: {project_name} (ID: {project_id})")
                    print(f"Round {current_round}/{total_rounds}")
                    
                    # Update model with server weights
                    self.model.set_weights([np.array(w) for w in weights])
                    self.current_round = current_round
                    
                    # Train locally
                    print(f"\nStarting local training for round {self.current_round}")
                    history = self.model.fit(
                        self.x_train,
                        self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[MetricCallback(self)]
                    )

                    # Evaluate model
                    val_loss, val_accuracy = self.model.evaluate(
                        self.x_test if self.x_test is not None else self.x_train,
                        self.y_test if self.y_test is not None else self.y_train,
                        verbose=0
                    )

                    # Send final update to server
                    final_metrics = {
                        'round': self.current_round,
                        'epoch': self.epochs,
                        'total_epochs': self.epochs,
                        'loss': float(history.history['loss'][-1]),
                        'accuracy': float(history.history['accuracy'][-1]),
                        'val_loss': float(val_loss),
                        'val_accuracy': float(val_accuracy),
                        'samples': len(self.x_train),
                        'project_id': project_id
                    }

                    response = requests.post(
                        f"{self.server_url}/api/clients/{self.client_id}/model_update",
                        json={
                            'weights': [w.tolist() for w in self.model.get_weights()],
                            'metrics': final_metrics
                        },
                        headers={'X-API-Key': self.api_key}
                    )
                    response.raise_for_status()
                    update_response = response.json()
                    if update_response.get('status') == 'success':
                        print(f"Round {self.current_round} completed successfully")
                    else:
                        print(f"Update response: {update_response}")
                
                else:
                    print(f"Unknown status from server: {status}")
                    print(f"Response data: {data}")
                
                # Wait before checking for the next task
                time.sleep(5)

            except requests.exceptions.RequestException as e:
                print(f"Network error during training: {e}")
                time.sleep(5)
            except Exception as e:
                print(f"Error during training: {e}")
                time.sleep(5)

class MetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, client):
        super().__init__()
        self.client = client

    def on_epoch_end(self, epoch, logs=None):
        """Send metrics after each epoch."""
        logs = logs or {}
        try:
            # Evaluate on test set if available
            if self.client.x_test is not None and self.client.y_test is not None:
                test_results = self.model.evaluate(
                    self.client.x_test,
                    self.client.y_test,
                    verbose=0
                )
                test_loss = test_results[0]
                test_accuracy = test_results[1]
            else:
                test_loss = logs.get('val_loss', 0.0)
                test_accuracy = logs.get('val_accuracy', 0.0)

            # Ensure all metrics are valid numbers
            metrics = {
                'round': self.client.current_round,
                'epoch': epoch + 1,
                'total_epochs': self.client.epochs,
                'loss': float(logs.get('loss', 0.0)),
                'accuracy': float(logs.get('accuracy', 0.0)),
                'val_loss': float(test_loss),
                'val_accuracy': float(test_accuracy),
                'samples': len(self.client.x_train),
                'client_id': self.client.client_id,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Validate metrics before sending
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if not np.isfinite(value):  # Check for inf/nan
                        metrics[key] = 0.0
                        print(f"Warning: Invalid {key} value detected, using 0.0")

            print(f"\nSending metrics to server for round {metrics['round']}, epoch {metrics['epoch']}:")
            print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            print(f"Val Loss: {metrics['val_loss']:.4f}, Val Accuracy: {metrics['val_accuracy']:.4f}")

            response = requests.post(
                f"{self.client.server_url}/api/clients/{self.client.client_id}/model_update",
                json={
                    'weights': [w.tolist() for w in self.model.get_weights()],
                    'metrics': metrics
                },
                headers={'X-API-Key': self.client.api_key}
            )
            response.raise_for_status()
            print("Metrics sent successfully")
            
        except Exception as e:
            print(f"Error sending epoch metrics: {e}")
            # Continue training even if metrics sending fails
            pass 