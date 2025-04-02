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
        self.current_project_id = None  # Track the current project ID
        
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
        while True:
            try:
                # Check for tasks from server
                response = requests.get(
                    f"{self.server_url}/api/clients/{self.client_id}/tasks",
                    headers={'X-API-Key': self.api_key}
                )
                response.raise_for_status()
                data = response.json()
                
                status = data.get('status')
                
                if status == 'error':
                    print(f"Error from server: {data.get('message', 'Unknown error')}")
                    time.sleep(5)
                    continue
                    
                elif status == 'waiting':
                    details = data.get('details', {})
                    print(f"Waiting: {details.get('message', 'No active projects')}")
                    print(f"Details: {details}")
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
                    dataset_name = details.get('dataset', 'mnist')
                    
                    # Store project ID and current round
                    self.current_project_id = project_id
                    self.current_round = current_round
                    
                    print(f"Training for project: {project_name} (ID: {project_id})")
                    print(f"Round {current_round}/{total_rounds}")
                    
                    # Check if we need to recreate the model to match server weights
                    server_weights = [np.array(w) for w in weights]
                    model_weights_count = len(self.model.get_weights())
                    server_weights_count = len(server_weights)
                    
                    if model_weights_count != server_weights_count:
                        print(f"Model architecture mismatch: client has {model_weights_count} weights, "
                              f"server sent {server_weights_count} weights")
                        print("Recreating model to match server architecture...")
                        
                        # Create a new model that exactly matches the server's architecture
                        # Using the exact architecture from the server's _initialize_tensorflow_model
                        input_shape = (28, 28, 1)
                        
                        # Create sequential model to match server exactly
                        self.model = tf.keras.Sequential([
                            # First Conv Block
                            # Conv1 (28x28x1 -> 28x28x32)
                            tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                            tf.keras.layers.BatchNormalization(),
                            # Conv2 (28x28x32 -> 28x28x32)
                            tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
                            # MaxPooling only after first block (28x28x32 -> 14x14x32)
                            tf.keras.layers.MaxPooling2D((2, 2)),
                            
                            # Second Conv Block
                            # Conv3 (14x14x32 -> 14x14x64)
                            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                            tf.keras.layers.BatchNormalization(),
                            # Conv4 (14x14x64 -> 14x14x64)
                            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                            # No MaxPooling here - keep 14x14 dimensions
                            
                            # Flatten layer - 14*14*64 = 12544 neurons
                            tf.keras.layers.Flatten(),
                            
                            # Dense layers
                            # Dense1 (12544 -> 512)
                            tf.keras.layers.Dense(512),
                            tf.keras.layers.BatchNormalization(),
                            # Output (512 -> 10)
                            tf.keras.layers.Dense(10, activation='softmax')
                        ])
                        
                        # Compile model
                        self.model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        print(f"Created new model with {len(self.model.get_weights())} weight tensors")
                        print("Model summary:")
                        self.model.summary()
                    
                    # Update model with server weights
                    try:
                        # Convert weights to float32 to ensure compatibility with TensorFlow
                        server_weights_float32 = [w.astype(np.float32) for w in server_weights]
                        
                        # Print weight shapes for debugging
                        print("\nServer weights shapes:")
                        for i, w in enumerate(server_weights_float32):
                            print(f"Weight {i}: {w.shape}")
                        
                        print("\nModel weights shapes:")
                        for i, w in enumerate(self.model.get_weights()):
                            print(f"Weight {i}: {w.shape}")
                        
                        # Apply weights
                        self.model.set_weights(server_weights_float32)
                        print("Successfully applied server weights to model")
                    except ValueError as e:
                        print(f"Error applying weights: {e}")
                        print("Detailed weight information:")
                        for i, w in enumerate(server_weights):
                            print(f"Server weight {i}: shape {w.shape}, type {w.dtype}")
                        for i, w in enumerate(self.model.get_weights()):
                            print(f"Model weight {i}: shape {w.shape}, type {w.dtype}")
                        continue
                    
                    # Train the model
                    history = self.model.fit(
                        self.x_train,
                        self.y_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_data=(self.x_test, self.y_test),
                        callbacks=[MetricCallback(self)],
                        verbose=1
                    )
                    
                    # Send final update to server
                    final_weights = [w.tolist() for w in self.model.get_weights()]
                    final_metrics = {
                        'loss': float(history.history['loss'][-1]),
                        'accuracy': float(history.history['accuracy'][-1]),
                        'val_loss': float(history.history['val_loss'][-1]),
                        'val_accuracy': float(history.history['val_accuracy'][-1]),
                        'round': current_round,
                        'project_id': project_id  # Ensure project_id is included
                    }
                    
                    # Double-check that project_id is set
                    if 'project_id' not in final_metrics or not final_metrics['project_id']:
                        print("Warning: project_id not set in final metrics")
                        if self.current_project_id:
                            final_metrics['project_id'] = self.current_project_id
                            print(f"Using stored project_id: {self.current_project_id}")
                        else:
                            print("Error: Cannot send update without project_id")
                            continue
                    
                    response = requests.post(
                        f"{self.server_url}/api/clients/{self.client_id}/model_update",
                        json={
                            'weights': final_weights,
                            'metrics': final_metrics
                        },
                        headers={'X-API-Key': self.api_key}
                    )
                    
                    # Check if the response is successful (even for completed projects)
                    if response.status_code == 200:
                        response_data = response.json()
                        if response_data.get('status') == 'success':
                            # Check if project is completed
                            if 'project_status' in response_data.get('details', {}) and response_data['details']['project_status'] == 'completed':
                                print(f"Project is already completed: {response_data['message']}")
                            else:
                                print("Training completed and update sent to server")
                        else:
                            print(f"Server response: {response_data.get('message', 'Unknown response')}")
                    else:
                        response.raise_for_status()
                        print("Training completed and update sent to server")
                
                time.sleep(1)  # Small delay between iterations
                
            except requests.exceptions.RequestException as e:
                print(f"Network error: {e}")
                time.sleep(5)
            except Exception as e:
                print(f"Error in training loop: {e}")
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
            
            # Add project_id to metrics if available
            if self.client.current_project_id:
                metrics['project_id'] = self.client.current_project_id
            
            # If project_id not available from client, get it from server
            if 'project_id' not in metrics:
                try:
                    project_response = requests.get(
                        f"{self.client.server_url}/api/clients/{self.client.client_id}/tasks",
                        headers={'X-API-Key': self.client.api_key}
                    )
                    if project_response.status_code == 200:
                        project_data = project_response.json()
                        if project_data.get('status') == 'training':
                            # Add project_id to metrics
                            project_id = project_data.get('details', {}).get('project_id')
                            if project_id:
                                metrics['project_id'] = project_id
                                # Also update the client's current_project_id
                                self.client.current_project_id = project_id
                except Exception as e:
                    print(f"Error getting project_id: {e}")
            
            # If project_id is still not available, don't send metrics
            if 'project_id' not in metrics:
                print("Warning: project_id not available, skipping metrics update")
                return

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
            
            # Check if the response is successful (even for completed projects)
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') == 'success':
                    # Check if project is completed
                    if 'project_status' in response_data.get('details', {}) and response_data['details']['project_status'] == 'completed':
                        print(f"Project is already completed: {response_data['message']}")
                    else:
                        print("Metrics sent successfully")
                else:
                    print(f"Server response: {response_data.get('message', 'Unknown response')}")
            else:
                response.raise_for_status()
                print("Metrics sent successfully")
            
        except Exception as e:
            print(f"Error sending epoch metrics: {e}")
            # Continue training even if metrics sending fails
            pass 