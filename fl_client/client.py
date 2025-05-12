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
import random

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
        self.final_update_sent = False  # Flag to track if final update was sent
        self.max_retries = 5            # Maximum number of retries for network operations
        
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
        if self._register_client():
            self._training_loop()
        else:
            print("Failed to register client. Stopping.")
            self.stop()
    
    def stop(self):
        """Stop the federated learning client."""
        self.is_training = False
        print("Client stopped.")
    
    def _register_client(self):
        """Register the client with the server."""
        max_retries = self.max_retries
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
                    },
                    timeout=30  # Add timeout to prevent hanging
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
                    # Use exponential backoff with jitter
                    delay = min(retry_delay * (2 ** attempt) + random.uniform(0, 1), 60)
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Registration failed.")
                    return False
                    
            except Exception as e:
                print(f"Unexpected error during registration: {e}")
                if attempt < max_retries - 1:
                    # Use exponential backoff with jitter
                    delay = min(retry_delay * (2 ** attempt) + random.uniform(0, 1), 60)
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Registration failed.")
                    return False
        
        return False
    
    def _start_heartbeat(self):
        """Start sending heartbeat to server."""
        def heartbeat():
            while self.is_training:
                try:
                    response = requests.post(
                        f"{self.server_url}/api/clients/{self.client_id}/heartbeat",
                        headers={'X-API-Key': self.api_key},
                        timeout=10  # Add timeout to prevent hanging
                    )
                    response.raise_for_status()
                except Exception as e:
                    # Don't log every heartbeat error to avoid console spam
                    pass
                time.sleep(30)  # Send heartbeat every 30 seconds
        
        self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self.heartbeat_thread.start()
    
    def _retry_request(self, request_fn, max_retries=None, is_final=False):
        """Generic retry function for network requests with exponential backoff.
        
        Args:
            request_fn: Function that makes the actual request
            max_retries: Maximum number of retries (default: self.max_retries)
            is_final: Whether this is a final update request (more aggressive retries)
            
        Returns:
            Response object or None if all retries failed
        """
        if max_retries is None:
            max_retries = self.max_retries if not is_final else self.max_retries * 2
            
        retry_delay = 5  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = request_fn()
                return response
            except requests.exceptions.RequestException as e:
                print(f"Network error (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Use exponential backoff with jitter for more robust retries
                    jitter = random.uniform(0, 1)
                    delay = min(retry_delay * (2 ** attempt) + jitter, 120)  # Cap at 2 minutes
                    
                    if is_final:
                        print(f"This is a FINAL update, will retry more aggressively.")
                        # For final updates, use a more aggressive retry strategy
                        delay = max(delay / 2, 5)  # Retry quicker for final updates
                        
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Max retries ({max_retries}) reached.")
                    return None
            except Exception as e:
                print(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return None
                
        return None
    
    def _training_loop(self):
        """Main training loop."""
        while self.is_training:
            try:
                # Define the request function
                def get_tasks():
                    return requests.get(
                        f"{self.server_url}/api/clients/{self.client_id}/tasks",
                        headers={'X-API-Key': self.api_key},
                        timeout=30
                    )
                
                # Use retry mechanism
                response = self._retry_request(get_tasks)
                
                if response and response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success' or data.get('status') == 'training':
                        # Extract project details
                        details = data.get('details', {})
                        project_id = details.get('project_id')
                        current_round = details.get('round', 0)
                        total_rounds = details.get('total_rounds', 0)
                        weights = details.get('weights', [])
                        dataset_name = details.get('dataset', 'mnist')
                        
                        # Store the project_id and current round
                        self.current_project_id = project_id
                        self.current_round = current_round
                        
                        # Check if project is completed
                        if details.get('project_status') == 'completed':
                            print(f"Project is already completed: {data['message']}")
                            
                            # If we haven't sent our final update yet, continue once more to ensure it's sent
                            if not self.final_update_sent and current_round >= total_rounds - 1:
                                print("Haven't sent final update yet. Proceeding with training one more time.")
                            else:
                                # We've already sent our final update or aren't in the final round, so we can stop
                                break
                        
                        # Get model weights and update local model
                        if weights:
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
                                if dataset_name.lower() == 'mnist':
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
                                elif dataset_name.lower() == 'cifar10':
                                    input_shape = (32, 32, 3)
                                    self.model = tf.keras.Sequential([
                                        # Conv layers
                                        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                                        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                                        tf.keras.layers.MaxPooling2D((2, 2)),
                                        
                                        # Flatten layer
                                        tf.keras.layers.Flatten(),
                                        
                                        # Dense layers
                                        tf.keras.layers.Dense(128, activation='relu'),
                                        tf.keras.layers.Dense(10, activation='softmax')
                                    ])
                                else:
                                    # Generic model for other datasets
                                    self.model = tf.keras.Sequential([
                                        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
                                        tf.keras.layers.Dense(10, activation='softmax')
                                    ])
                                
                                # Compile model
                                self.model.compile(
                                    optimizer='adam',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                
                                print(f"Created new model with {len(self.model.get_weights())} weight tensors")
                            
                            try:
                                # Set the weights to the model
                                self.model.set_weights(server_weights)
                                print("Successfully applied server weights to model")
                                
                                # Train the model
                                history = self.model.fit(
                                    self.x_train, self.y_train,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    validation_data=(self.x_test, self.y_test),
                                    callbacks=[MetricCallback(self)]
                                )
                                
                                # Check if this is the final round
                                is_final_round = current_round >= total_rounds - 1
                                
                                # Get the final metrics
                                final_metrics = {
                                    'accuracy': float(history.history['accuracy'][-1]),
                                    'loss': float(history.history['loss'][-1]),
                                    'val_accuracy': float(history.history['val_accuracy'][-1]),
                                    'val_loss': float(history.history['val_loss'][-1]),
                                    'epoch': self.epochs,
                                    'total_epochs': self.epochs,
                                    'round': current_round,
                                    'samples': len(self.x_train),
                                    'project_id': project_id,
                                    'is_final': is_final_round  # Set final flag based on round number
                                }
                                
                                if is_final_round:
                                    print(f"\n*** SENDING FINAL UPDATE FOR PROJECT {project_id} ***")
                                    print(f"*** CURRENT ROUND: {current_round}, TOTAL ROUNDS: {total_rounds} ***")
                                    print(f"*** IS_FINAL FLAG SET TO: {final_metrics['is_final']} ***\n")
                                
                                # Send the updated weights and metrics back to the server
                                def send_update():
                                    return requests.post(
                                        f"{self.server_url}/api/clients/{self.client_id}/model_update",
                                        json={
                                            'weights': [w.tolist() for w in self.model.get_weights()],
                                            'metrics': final_metrics
                                        },
                                        headers={'X-API-Key': self.api_key},
                                        timeout=60 if is_final_round else 30  # Longer timeout for final updates
                                    )
                                
                                # Use retry mechanism with more aggressive settings for final updates
                                response = self._retry_request(send_update, is_final=is_final_round)
                                
                                # Check if the response is successful
                                if response and response.status_code == 200:
                                    response_data = response.json()
                                    if response_data.get('status') == 'success':
                                        # If this was the final update, mark it as sent
                                        if is_final_round:
                                            self.final_update_sent = True
                                            print("Final update successfully sent and processed by server!")
                                            
                                        # Check if project is completed
                                        if 'project_status' in response_data.get('details', {}) and response_data['details']['project_status'] == 'completed':
                                            print(f"Project is now marked as completed on the server.")
                                            
                                            # If we successfully sent the final update, we can break
                                            if self.final_update_sent:
                                                print("Training cycle complete. Exiting training loop.")
                                                break
                                        else:
                                            print("Training completed and update sent to server")
                                    else:
                                        print(f"Server response: {response_data.get('message', 'Unknown response')}")
                                else:
                                    print(f"Failed to send update after multiple retries.")
                                    
                                    # If this was the final update and all retries failed,
                                    # try the emergency completion as last resort
                                    if is_final_round and not self.final_update_sent:
                                        print("Attempting emergency completion...")
                                        
                                        # Create minimal metrics for emergency completion
                                        emergency_metrics = {
                                            'project_id': project_id,
                                            'is_final': True,
                                            'accuracy': final_metrics['accuracy'],
                                            'loss': final_metrics['loss'],
                                            'val_accuracy': final_metrics['val_accuracy'],
                                            'val_loss': final_metrics['val_loss'],
                                            'emergency': True
                                        }
                                        
                                        # Define emergency request function (no weights, smaller payload)
                                        def send_emergency():
                                            return requests.post(
                                                f"{self.server_url}/api/clients/{self.client_id}/model_update",
                                                json={'metrics': emergency_metrics},
                                                headers={'X-API-Key': self.api_key},
                                                timeout=30
                                            )
                                        
                                        # Try the emergency completion with max retries
                                        emergency_response = self._retry_request(send_emergency, max_retries=8, is_final=True)
                                        
                                        if emergency_response and emergency_response.status_code == 200:
                                            print("Emergency completion successful!")
                                            self.final_update_sent = True
                                            # We can now exit the training loop
                                            break
                                        else:
                                            print("Emergency completion also failed. Will continue normally.")
                                
                            except ValueError as e:
                                print(f"Error applying weights: {e}")
                                print("Weight mismatch. This might be due to incompatible model architectures.")
                    elif data.get('status') == 'waiting':
                        print(f"Waiting: {data.get('message', 'No active projects')}")
                        # Sleep longer when waiting for projects
                        time.sleep(10)
                    else:
                        print(f"Server response: {data.get('message', 'Unknown response')}")
                else:
                    if response:
                        print(f"Error getting tasks: {response.status_code}")
                        if hasattr(response, 'text'):
                            print(f"Response text: {response.text}")
                    else:
                        print("Failed to get tasks after multiple retries.")
            
            except Exception as e:
                print(f"Error in training loop: {e}")
                import traceback
                print(traceback.format_exc())
                time.sleep(5)
            
            # Small delay between iterations
            time.sleep(2)
        
        print("Training loop exited.")

class MetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.last_completed_epoch = -1
        self.project_completed = False

    def on_epoch_end(self, epoch, logs=None):
        """Send metrics after each epoch."""
        logs = logs or {}
        try:
            # Skip if project is already marked as completed
            if self.project_completed:
                print("Project already marked as completed. Skipping metrics update.")
                return
                
            # Skip if we've already processed this epoch (avoid duplicate updates)
            if epoch <= self.last_completed_epoch:
                print(f"Epoch {epoch+1} already processed. Skipping duplicate update.")
                return
                
            # Update the last completed epoch
            self.last_completed_epoch = epoch
            
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
                    def get_tasks():
                        return requests.get(
                            f"{self.client.server_url}/api/clients/{self.client.client_id}/tasks",
                            headers={'X-API-Key': self.client.api_key},
                            timeout=10
                        )
                    
                    # Use retry mechanism with fewer retries for status updates
                    project_response = self.client._retry_request(get_tasks, max_retries=2)
                    
                    if project_response and project_response.status_code == 200:
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
            
            # Define the metrics update request
            def send_metrics():
                return requests.post(
                    f"{self.client.server_url}/api/clients/{self.client.client_id}/model_update",
                    json={
                        'weights': [w.tolist() for w in self.model.get_weights()],
                        'metrics': metrics
                    },
                    headers={'X-API-Key': self.client.api_key},
                    timeout=20  # Shorter timeout for epoch metrics
                )
            
            # Use retry mechanism with fewer retries for epoch metrics
            response = self.client._retry_request(send_metrics, max_retries=2)
            
            # Check if the response is successful
            if response and response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') == 'success':
                    # Check if project is completed
                    if 'project_status' in response_data.get('details', {}) and response_data['details']['project_status'] == 'completed':
                        print(f"Project is already completed: {response_data['message']}")
                        self.project_completed = True
                    else:
                        print("Metrics sent successfully")
                else:
                    print(f"Server response: {response_data.get('message', 'Unknown response')}")
            elif response and response.status_code == 503:
                print(f"Server unavailable (503): The server might be temporarily down. Will continue training locally.")
            elif response:
                print(f"Error sending metrics: {response.status_code}")
                if hasattr(response, 'text'):
                    print(f"Response text: {response.text}")
            else:
                print("Failed to send metrics after retries. Continuing with training.")
            
        except Exception as e:
            print(f"Error sending epoch metrics: {e}")
            # Continue training even if metrics sending fails 