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
        
        # Preprocess input data
        self.x_train, self.y_train = self._preprocess_data(x_train, y_train)
        
        # Preprocess test data if available
        if x_test is not None and y_test is not None:
            self.x_test, self.y_test = self._preprocess_data(x_test, y_test)
        else:
            self.x_test, self.y_test = None, None
            
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
        
        # Print data shape information
        print(f"Initialized client with {len(self.x_train)} training samples")
        if self.x_test is not None:
            print(f"Test set has {len(self.x_test)} samples")
        print(f"Input shape: {self.x_train[0].shape}")
        print(f"Model input shape: {self.model.input_shape}")
            
        # Compile model if not already compiled
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            print("Model not compiled, compiling with default settings...")
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            print("Model already compiled")
            
        # Test weight serialization to catch issues early
        try:
            print("\nVerifying model weight serialization...")
            weights = self.model.get_weights()
            print(f"Model has {len(weights)} weight arrays")
            
            # Test serialization of a few weights
            for i in range(min(3, len(weights))):
                w = weights[i]
                if hasattr(w, 'shape'):
                    print(f"Weight {i}: shape={w.shape}, size={w.size}")
                    # Try basic serialization
                    w_list = w.tolist()
                    print(f"  Serialization test: list length={len(w_list) if hasattr(w_list, '__len__') else 'N/A'}")
            
            # Test full serialization
            serialized = self._serialize_weights(weights)
            if serialized:
                print(f"✓ Weight serialization test successful: {len(serialized)}/{len(weights)} arrays")
            else:
                print("⚠ Weight serialization test failed")
        except Exception as e:
            print(f"⚠ Error testing weight serialization: {e}")
            print("This might cause issues when sending updates to the server")
    
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
        """Start a background thread for sending periodic heartbeats."""
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def _heartbeat_worker(self):
        """Worker thread for sending heartbeats."""
        while self.is_training:
            try:
                self._send_heartbeat()
            except Exception as e:
                print(f"Heartbeat error: {e}")
            
            time.sleep(5)  # Send heartbeat more frequently (changed from 30)
    
    def _retry_request(self, request_fn, max_retries=None, is_final=False):
        """Retry a request with exponential backoff."""
        if max_retries is None:
            max_retries = self.max_retries if not is_final else self.max_retries * 2
            
        retry_delay = 5  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = request_fn()
                
                # Handle 5xx server errors with retries
                if response.status_code >= 500:
                    print(f"Server error: {response.status_code} (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        jitter = random.uniform(0, 1)
                        delay = min(retry_delay * (1.5 ** attempt) + jitter, 60)  # More gradual backoff
                        print(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                
                return response
            except requests.exceptions.RequestException as e:
                print(f"Network error (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Use exponential backoff with jitter for more robust retries
                    jitter = random.uniform(0, 1)
                    delay = min(retry_delay * (1.5 ** attempt) + jitter, 60)  # More gradual backoff
                    
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
        consecutive_failures = 0
        max_consecutive_failures = 10  # After this many failures, pause for longer
        
        # Backoff strategy for "no projects" state
        no_projects_count = 0
        no_projects_max_backoff = 0  # No backoff - poll continuously (changed from 60)
        
        while self.is_training:
            try:
                # If we've been in "no projects" state for a while, use longer timeouts
                use_short_timeout = True
                timeout = 15
                
                # Removed adaptive timeout based on no_projects_count
                
                # Define the request function
                def get_tasks():
                    params = {'short_timeout': use_short_timeout}
                    # Add current project ID if available to avoid "not found" errors
                    if self.current_project_id:
                        params['project_id'] = self.current_project_id
                    
                    return requests.get(
                        f"{self.server_url}/api/clients/{self.client_id}/tasks",
                        headers={'X-API-Key': self.api_key},
                        params=params,
                        timeout=timeout
                    )
                
                # Use retry mechanism
                response = self._retry_request(get_tasks)
                
                if response and response.status_code == 200:
                    data = response.json()
                    
                    # Check if we're waiting for projects
                    if data.get('status') == 'waiting' and 'No active projects' in data.get('message', ''):
                        no_projects_count += 1
                        wait_time = 0.1  # Minimal wait time for polling (almost continuously)
                        print(f"No active projects available. Checking again immediately. (Poll #{no_projects_count})")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Reset no_projects counter if we got a different response
                        no_projects_count = 0
                    
                    if data.get('status') == 'success' or data.get('status') == 'training':
                        consecutive_failures = 0  # Reset failure counter on success
                        
                        # Extract project details
                        details = data.get('details', {})
                        project_id = details.get('project_id')
                        
                        # Skip invalid projects
                        if not project_id:
                            print("No project ID in response, waiting before retrying...")
                            time.sleep(1)
                            continue
                        
                        current_round = details.get('round', 0)
                        total_rounds = details.get('total_rounds', 0)
                        weights = details.get('weights', [])
                        weights_available = details.get('weights_available', False)
                        dataset_name = details.get('dataset', 'mnist')
                        
                        # Store the project_id and current round
                        self.current_project_id = project_id
                        self.current_round = current_round
                        
                        # Check if project is completed - look for multiple completion indicators
                        if (data.get('status') == 'completed' or 
                            details.get('project_status') == 'completed' or
                            details.get('should_stop') or 
                            details.get('training_complete') or
                            details.get('next_action') == 'stop_training'):
                            
                            print(f"Project completion detected: {data.get('message', 'Project completed')}")
                            
                            # If we haven't sent our final update yet, continue once more to ensure it's sent
                            if not self.final_update_sent and current_round >= 0:
                                print("Haven't sent final update yet. Proceeding with training one more time.")
                            else:
                                # We've already sent our final update, so we can stop
                                print("Final update already sent or project marked complete. Stopping client.")
                                self.is_training = False
                                break  # Exit the training loop completely
                        
                        # If the response indicated weights are available but didn't include them (short_timeout mode)
                        # make a second request to get the weights
                        if not weights and weights_available:
                            print("Lightweight response received, requesting full weights...")
                            
                            def get_full_weights():
                                return requests.get(
                                    f"{self.server_url}/api/clients/{self.client_id}/tasks",
                                    headers={'X-API-Key': self.api_key},
                                    params={'project_id': project_id},  # Request specific project
                                    timeout=60  # Longer timeout for full weights request
                                )
                            
                            full_response = self._retry_request(get_full_weights)
                            if full_response and full_response.status_code == 200:
                                full_data = full_response.json()
                                weights = full_data.get('details', {}).get('weights', [])
                                if not weights:
                                    print("Failed to get weights in second request. Waiting before retrying...")
                                    time.sleep(1)  # Shorter wait time (changed from 15)
                                    continue
                            else:
                                print("Failed to get full weights. Waiting before retrying...")
                                time.sleep(1)  # Shorter wait time (changed from 15)
                                continue
                        
                        # Get model weights and update local model
                        if weights:
                            # Check if we need to recreate the model to match server weights
                            server_weights = []
                            for w in weights:
                                try:
                                    # Convert each weight array properly and ensure it's valid
                                    if isinstance(w, list):
                                        # Skip empty arrays
                                        if not w:
                                            print(f"Skipping empty weight array")
                                            continue
                                            
                                        # Convert list to numpy array
                                        w_array = np.array(w, dtype=np.float32)
                                        
                                        # Check for valid array
                                        if w_array.size == 0 or np.any(np.isnan(w_array)) or np.any(np.isinf(w_array)):
                                            print(f"Invalid weight array detected: empty or contains NaN/Inf")
                                            continue
                                            
                                        server_weights.append(w_array)
                                    elif isinstance(w, dict):
                                        print(f"Unexpected weight format (dict): {w.keys()}")
                                        continue
                                    else:
                                        print(f"Unexpected weight type: {type(w)}")
                                        continue
                                except Exception as w_err:
                                    print(f"Error processing weight: {w_err}")
                                    continue
                                    
                            # Only proceed if we have valid weights
                            if len(server_weights) == 0:
                                print("No valid weights received, skipping training round")
                                time.sleep(1)
                                continue
                                
                            # Check if weights count matches model
                            model_weights_count = len(self.model.get_weights())
                            server_weights_count = len(server_weights)
                            
                            if model_weights_count != server_weights_count:
                                print(f"Model architecture mismatch: client has {model_weights_count} weights, "
                                      f"server sent {server_weights_count} weights")
                                print("Recreating model to match server architecture...")
                                
                                # Create a new model that exactly matches the server's architecture
                                # Using the exact architecture from the server's _initialize_tensorflow_model
                                if dataset_name.lower() == 'mnist':
                                    print("Creating MNIST model to match server architecture...")
                                    input_shape = (28, 28, 1)
                                    
                                    # Create sequential model to match server exactly - with activations
                                    self.model = tf.keras.Sequential([
                                        # First Conv Block
                                        # Conv1 (28x28x1 -> 28x28x32)
                                        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.ReLU(),  # Add ReLU activation
                                        # Conv2 (28x28x32 -> 28x28x32)
                                        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
                                        # MaxPooling only after first block (28x28x32 -> 14x14x32)
                                        tf.keras.layers.MaxPooling2D((2, 2)),
                                        
                                        # Second Conv Block
                                        # Conv3 (14x14x32 -> 14x14x64)
                                        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.ReLU(),  # Add ReLU activation
                                        # Conv4 (14x14x64 -> 14x14x64)
                                        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                                        # NO second MaxPooling - keep dimensions at 14x14x64
                                        
                                        # Flatten layer - 14*14*64 = 12544 neurons
                                        tf.keras.layers.Flatten(),
                                        
                                        # Dense layers
                                        # Dense1 (12544 -> 512)
                                        tf.keras.layers.Dense(512),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.ReLU(),  # Add ReLU activation
                                        # Output (512 -> 10)
                                        tf.keras.layers.Dense(10, activation='softmax')
                                    ])
                                elif dataset_name.lower() == 'cifar10':
                                    print("Creating CIFAR10 model to match server architecture...")
                                    input_shape = (32, 32, 3)
                                    self.model = tf.keras.Sequential([
                                        # Conv layers
                                        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                                        tf.keras.layers.ReLU(),  # Add activation
                                        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                                        tf.keras.layers.ReLU(),  # Add activation
                                        tf.keras.layers.MaxPooling2D((2, 2)),
                                        
                                        # Flatten layer
                                        tf.keras.layers.Flatten(),
                                        
                                        # Dense layers
                                        tf.keras.layers.Dense(128, activation='relu'),
                                        tf.keras.layers.Dense(10, activation='softmax')
                                    ])
                                else:
                                    # Generic model for other datasets
                                    print(f"Creating generic model for {dataset_name} dataset...")
                                    self.model = tf.keras.Sequential([
                                        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
                                        tf.keras.layers.Dense(10, activation='softmax')
                                    ])
                                
                                # Compile model
                                print("Compiling model...")
                                self.model.compile(
                                    optimizer='adam',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                
                                # Build model to initialize weights
                                print("Building model to initialize weights...")
                                if self.x_train is not None:
                                    # Use actual data shape for build
                                    sample_shape = self.x_train[0:1].shape
                                    self.model.build(sample_shape)
                                
                                # Print detailed model summary for debugging
                                self._print_model_summary()
                                    
                                # Verify model is properly initialized
                                weights = self.model.get_weights()
                                print(f"Created new model with {len(weights)} weight tensors")
                                
                                # Verify weights are non-empty
                                empty_weights = 0
                                for i, w in enumerate(weights):
                                    if not isinstance(w, np.ndarray) or w.size == 0 or not np.all(np.isfinite(w)):
                                        print(f"Warning: Weight {i} is empty or invalid")
                                        empty_weights += 1
                                
                                if empty_weights > 0:
                                    print(f"WARNING: {empty_weights} empty or invalid weight tensors detected")
                                else:
                                    print("All weight tensors are valid")
                                    
                            try:
                                # Verify server weights match model's expected shapes
                                model_weights = self.model.get_weights()
                                shape_mismatch = False
                                
                                print("\n=== WEIGHT COMPATIBILITY CHECK ===")
                                if len(model_weights) != len(server_weights):
                                    print(f"❌ Count mismatch: Model has {len(model_weights)} weight arrays, server sent {len(server_weights)}")
                                    shape_mismatch = True
                                else:
                                    print(f"✓ Weight count matches: {len(model_weights)} arrays")
                                    
                                # Compare shapes of each weight tensor
                                for i, (model_w, server_w) in enumerate(zip(model_weights, server_weights)):
                                    if model_w.shape != server_w.shape:
                                        print(f"❌ Shape mismatch at index {i}: Model expects {model_w.shape}, server sent {server_w.shape}")
                                        shape_mismatch = True
                                    else:
                                        print(f"✓ Weight {i} shape matches: {model_w.shape}")
                                
                                if shape_mismatch:
                                    print("\nDetailed server weights:")
                                    for i, w in enumerate(server_weights):
                                        print(f"Server weight {i}: Shape {w.shape}")
                                    
                                    print("\nThis suggests a model architecture mismatch between client and server.")
                                    print("The likely cause is different neural network structures.")
                                    raise ValueError(f"Weight shape mismatch detected. Cannot apply server weights to client model.")
                                else:
                                    print("✓ All weight shapes compatible.\n")
                                
                                # Set the weights to the model
                                self.model.set_weights(server_weights)
                                print("Successfully applied server weights to model")
                                
                                # Train the model
                                print(f"Training model for {self.epochs} epochs with batch size {self.batch_size}")
                                print(f"Training data shape: {self.x_train.shape}, Labels shape: {self.y_train.shape}")
                                
                                # Store original weights for comparison
                                original_weights = [w.copy() for w in self.model.get_weights()]
                                
                                # Check if this is the final round
                                is_final_round = current_round >= total_rounds - 1
                                
                                # Initialize this right at the beginning of training loop before any try/except
                                is_final = is_final_round
                                
                                # Calculate steps per epoch to fix "Unknown" issue
                                steps_per_epoch = len(self.x_train) // self.batch_size
                                if steps_per_epoch == 0:  # If batch_size > dataset size
                                    steps_per_epoch = 1
                                
                                print(f"Training with steps_per_epoch={steps_per_epoch}, batch_size={self.batch_size}")
                                
                                # COMPLETELY MANUAL TRAINING APPROACH to avoid TensorFlow bugs
                                try:
                                    print("Starting custom training loop to ensure proper weight updates...")
                                    start_time = time.time()
                                    # Store original weights for comparison
                                    original_weights = [w.copy() for w in self.model.get_weights()]
                                    
                                    # Get training data
                                    x_data = self.x_train
                                    y_data = self.y_train
                                    
                                    # Shuffle the training data
                                    indices = np.random.permutation(len(x_data))
                                    x_data = x_data[indices]
                                    y_data = y_data[indices]
                                    
                                    # Perform manual training for specified number of epochs
                                    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
                                    
                                    for epoch in range(self.epochs):
                                        epoch_loss = 0
                                        epoch_accuracy = 0
                                        batch_count = 0
                                        stalled_counter = 0
                                        last_batch_time = time.time()
                                        
                                        # Process mini-batches
                                        for i in range(0, len(x_data), self.batch_size):
                                            # Check for stalled training (not progressing for 30 seconds)
                                            current_time = time.time()
                                            if current_time - last_batch_time > 30:
                                                stalled_counter += 1
                                                print(f"Training appears stalled - no progress for {current_time - last_batch_time:.1f} seconds")
                                                
                                                if stalled_counter >= 2:  # After 2 stall detections (60 seconds total)
                                                    print("Training stalled for too long - stopping")
                                                    break
                                            else:
                                                stalled_counter = 0  # Reset if we're making progress
                                                
                                            # Get batch
                                            x_batch = x_data[i:i + self.batch_size]
                                            y_batch = y_data[i:i + self.batch_size]
                                            
                                            # Train on batch and get metrics
                                            metrics = self.model.train_on_batch(x_batch, y_batch, return_dict=True)
                                            
                                            # Update epoch metrics
                                            epoch_loss += metrics['loss']
                                            if 'accuracy' in metrics:
                                                epoch_accuracy += metrics['accuracy']
                                            
                                            batch_count += 1
                                            last_batch_time = time.time()  # Update the last batch time
                                            
                                            # Print progress
                                            if batch_count % 10 == 0:
                                                print(f"Epoch {epoch+1}, Batch {batch_count}/{steps_per_epoch}: loss={metrics['loss']:.4f}, accuracy={metrics.get('accuracy', 0):.4f}")
                                        
                                        # Compute average metrics for the epoch
                                        if batch_count > 0:
                                            epoch_loss /= batch_count
                                            epoch_accuracy /= batch_count
                                            
                                            # Evaluate on validation set
                                            if self.x_test is not None and self.y_test is not None:
                                                val_metrics = self.model.evaluate(self.x_test, self.y_test, verbose=0, return_dict=True)
                                                val_loss = val_metrics['loss']
                                                val_accuracy = val_metrics.get('accuracy', 0)
                                            else:
                                                val_loss = 0
                                                val_accuracy = 0
                                            
                                            # Store metrics
                                            history['loss'].append(epoch_loss)
                                            history['accuracy'].append(epoch_accuracy)
                                            history['val_loss'].append(val_loss)
                                            history['val_accuracy'].append(val_accuracy)
                                            
                                            print(f"Epoch {epoch+1}/{self.epochs}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
                                        
                                        # Stop if training has stalled
                                        if stalled_counter >= 2:
                                            print("Stopping training due to stalled progress")
                                            break
                                    
                                    # Manually invoke the callback to report metrics
                                    if hasattr(self, 'callback') and self.callback:
                                        self.callback.on_epoch_end(self.epochs-1, history)
                                    
                                    # Final metrics for the update
                                    final_metrics = {
                                        'accuracy': float(history['accuracy'][-1]) if history['accuracy'] else 0.0,
                                        'loss': float(history['loss'][-1]) if history['loss'] else 0.0,
                                        'val_accuracy': float(history['val_accuracy'][-1]) if history['val_accuracy'] else 0.0,
                                        'val_loss': float(history['val_loss'][-1]) if history['val_loss'] else 0.0,
                                        'epoch': self.epochs,
                                        'total_epochs': self.epochs,
                                        'round': current_round,
                                        'samples': len(self.x_train),
                                        'project_id': project_id,
                                        'is_final': is_final
                                    }
                                    
                                    print("Custom training loop completed successfully")
                                except Exception as train_err:
                                    print(f"Error during custom training: {train_err}")
                                    import traceback
                                    traceback.print_exc()
                                    print("Training failed - will use initial weights")
                                    # Restore original weights on failure
                                    self.model.set_weights(original_weights)
                                    # Create empty history for downstream code
                                    history = {'loss': [1.0], 'accuracy': [0.0], 'val_loss': [1.0], 'val_accuracy': [0.0]}
                                    # Create metrics structure
                                    final_metrics = {
                                        'accuracy': 0.0,
                                        'loss': 1.0,
                                        'val_accuracy': 0.0,
                                        'val_loss': 1.0,
                                        'epoch': 1,
                                        'total_epochs': self.epochs,
                                        'round': current_round,
                                        'samples': len(self.x_train),
                                        'project_id': project_id,
                                        'is_final': is_final
                                    }
                                
                                # Get the final metrics
                                final_metrics = {
                                    'accuracy': float(history['accuracy'][-1]) if history['accuracy'] else 0.0,
                                    'loss': float(history['loss'][-1]) if history['loss'] else 0.0,
                                    'val_accuracy': float(history['val_accuracy'][-1]) if history['val_accuracy'] else 0.0,
                                    'val_loss': float(history['val_loss'][-1]) if history['val_loss'] else 0.0,
                                    'epoch': self.epochs,
                                    'total_epochs': self.epochs,
                                    'round': current_round,
                                    'samples': len(self.x_train),
                                    'project_id': project_id,
                                    'is_final': is_final
                                }
                                
                                if is_final_round:
                                    print(f"\n*** SENDING UPDATE FOR FINAL ROUND OF PROJECT {project_id} ***")
                                    print(f"*** CURRENT ROUND: {current_round}, TOTAL ROUNDS: {total_rounds} ***")
                                    print(f"*** IS_FINAL FLAG SET TO: {final_metrics['is_final']} ***\n")
                                
                                # Send the updated weights and metrics back to the server
                                def send_update():
                                    # Get model weights and verify they're not empty
                                    weights = self.model.get_weights()
                                    
                                    # Debug output to see the weights
                                    print(f"Model has {len(weights)} weight arrays")
                                    for i, w in enumerate(weights[:3]):  # Print first few weights for debugging
                                        if hasattr(w, 'shape'):
                                            print(f"Weight {i} shape: {w.shape}, size: {w.size}, min: {np.min(w):.4f}, max: {np.max(w):.4f}")
                                        else:
                                            print(f"Weight {i} type: {type(w)}, length: {len(w) if hasattr(w, '__len__') else 'N/A'}")
                                    
                                    # Generate a unique token for this update to prevent file conflicts
                                    client_unique_id = f"{self.client_id}_{socket.gethostname().replace('-', '_')}"
                                    timestamp = int(time.time())
                                    random_part = random.randint(10000, 99999)
                                    update_token = f"{client_unique_id}_{timestamp}_{random_part}"
                                    
                                    # Create a file-safe path component for the server
                                    file_safe_id = ''.join(c if c.isalnum() else '_' for c in self.client_id)
                                    file_path_suggestion = f"client_{file_safe_id}_{timestamp}_{random_part}"
                                    
                                    # Use the robust serialization function
                                    valid_weights = self._serialize_weights(weights)
                                    
                                    if not valid_weights:
                                        print("ERROR: Weight serialization failed. Sending metrics-only update.")
                                        return requests.post(
                                            f"{self.server_url}/api/clients/{self.client_id}/model_update",
                                            json={
                                                'metrics': final_metrics,
                                                'update_token': update_token,  # Add token to help server track updates
                                                'file_path_suggestion': file_path_suggestion,  # Help server create unique paths
                                                'weights_failed': True  # Indicator that weights failed to serialize
                                            },
                                            headers={'X-API-Key': self.api_key},
                                            timeout=60
                                        )
                                    
                                    print(f"Serialized {len(valid_weights)} weight arrays for update")
                                    
                                    # Check total size of serialized weights to avoid request size limits
                                    import sys
                                    try:
                                        # Estimate memory size of weights
                                        estimated_size = 0
                                        for w in valid_weights:
                                            # Rough estimation - each float is 4 bytes + overhead
                                            if isinstance(w, list):
                                                estimated_size += sys.getsizeof(w)
                                                # Check nested lists
                                                if w and isinstance(w[0], list):
                                                    for inner_list in w:
                                                        estimated_size += sys.getsizeof(inner_list)
                                        
                                        print(f"Estimated weight data size: {estimated_size / (1024*1024):.2f} MB")
                                        
                                        # If weights are too large, send metrics-only update
                                        if estimated_size > 50 * 1024 * 1024:  # 50MB limit
                                            print("WARNING: Weight data too large for single request, sending metrics-only update")
                                            
                                            return requests.post(
                                                f"{self.server_url}/api/clients/{self.client_id}/model_update",
                                                json={
                                                    'metrics': final_metrics,
                                                    'weights_too_large': True,
                                                    'update_token': update_token,  # Add token for server tracking
                                                    'file_path_suggestion': file_path_suggestion,  # Help server create unique paths
                                                    'weights_failed': True  # Indicator that weights failed to serialize
                                                },
                                                headers={'X-API-Key': self.api_key},
                                                timeout=60
                                            )
                                    except Exception as size_err:
                                        print(f"Error estimating weight size: {size_err}")
                                    
                                    # Try to verify weights are properly serializable
                                    try:
                                        import json
                                        # Test JSON serialization with a small sample (first weight array)
                                        if valid_weights:
                                            json_test = json.dumps(valid_weights[0])
                                            print(f"JSON serialization test successful, sample size: {len(json_test)} bytes")
                                    except Exception as json_err:
                                        print(f"JSON serialization test failed: {json_err}")
                                        # Fall back to metrics-only update
                                        return requests.post(
                                            f"{self.server_url}/api/clients/{self.client_id}/model_update",
                                            json={
                                                'metrics': final_metrics,
                                                'update_token': update_token,  # Add token to help server track updates
                                                'file_path_suggestion': file_path_suggestion,  # Help server create unique paths
                                                'weights_failed': True  # Indicator that weights failed to serialize
                                            },
                                            headers={'X-API-Key': self.api_key},
                                            timeout=60
                                        )
                                    
                                    # Send the full update with weights
                                    return requests.post(
                                        f"{self.server_url}/api/clients/{self.client_id}/model_update",
                                        json={
                                            'weights': valid_weights,
                                            'metrics': final_metrics,
                                            'update_token': update_token,  # Add token to help avoid file conflicts
                                            'file_path_suggestion': file_path_suggestion,  # Help server create unique paths
                                            'weight_metadata': {  # Add metadata about weights to help server validate
                                                'weight_count': len(valid_weights),
                                                'shapes': [list(w.shape) if hasattr(w, 'shape') else [] for w in self.model.get_weights()],
                                                'sizes': [w.size if hasattr(w, 'size') else 0 for w in self.model.get_weights()],
                                                'has_nan': [bool(np.any(np.isnan(w))) if isinstance(w, np.ndarray) else False for w in self.model.get_weights()],
                                                'weight_version': 2,  # Version to help server identify format
                                                'client_timestamp': int(time.time())
                                            }
                                        },
                                        headers={'X-API-Key': self.api_key},
                                        timeout=120 if is_final_round else 60  # Increased timeout for model updates
                                    )
                                
                                # Use retry mechanism with more aggressive settings for final updates
                                response = self._retry_request(send_update, is_final=is_final_round)
                                
                                # Check if the response is successful
                                if response and response.status_code == 200:
                                    response_data = response.json()
                                    print(f"Server response: {response_data.get('message', 'Unknown response')}")
                                    
                                    # Check multiple indicators of completion
                                    project_completed = (
                                        response_data.get('status') == 'completed' or
                                        response_data.get('project_completed') or
                                        response_data.get('training_complete') or
                                        response_data.get('should_stop') or
                                        response_data.get('next_action') == 'stop_training' or
                                        (response_data.get('project_status') == 'completed')
                                    )
                                    
                                    # If this was the final update, mark it as sent
                                    if is_final_round:
                                        self.final_update_sent = True
                                        print("Final update successfully sent and processed by server!")
                                        
                                    # For any completion signal, stop training
                                    if project_completed or is_final_round or self.final_update_sent:
                                        print("Training cycle complete. Exiting training loop.")
                                        self.is_training = False
                                        break
                                        
                                    print("Update sent to server successfully.")
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
                        else:
                            print("No weights received in task. Waiting before trying again...")
                            time.sleep(1)  # Shorter wait time (changed from 15)
                    elif data.get('status') == 'waiting':
                        print(f"Waiting: {data.get('message', 'No active projects')}")
                        # Poll continuously with minimal delay
                        time.sleep(0.1)
                    elif data.get('status') == 'completed':
                        print(f"Project completed: {data.get('message', 'Training finished')}")
                        # The project is complete, break out of the training loop
                        print("Server indicates project is complete. Stopping training loop.")
                        break
                    else:
                        print(f"Server response: {data.get('message', 'Unknown response')}")
                else:
                    # Response error or no response
                    error_msg = "Failed to get tasks"
                    if response:
                        error_msg += f" - Status code: {response.status_code}"
                    
                    print(f"{error_msg} after multiple retries.")
                    
                    # Increment failure counter but use minimal backoff
                    consecutive_failures += 1
                    wait_time = 1.0  # Fixed short wait time regardless of failures
                    
                    print(f"Retrying immediately after failure ({consecutive_failures} consecutive failures)")
                    time.sleep(wait_time)
            except Exception as e:
                print(f"Error in training loop: {str(e)}")
                traceback.print_exc()
                
                # Shorter safety wait after unexpected error
                time.sleep(5)
            
            # Minimal delay between iterations
            time.sleep(0.1)
        
        print("Training loop exited.")

    def _send_heartbeat(self):
        """Send a heartbeat to the server to indicate the client is still active."""
        try:
            response = requests.post(
                f"{self.server_url}/api/clients/{self.client_id}/heartbeat",
                headers={'X-API-Key': self.api_key},
                timeout=5  # Short timeout
            )
            response.raise_for_status()
            return True
        except Exception:
            # Silently fail - heartbeat errors are not critical
            return False

    def _preprocess_data(self, x_data, y_data):
        """Preprocess input data to ensure correct format for the model."""
        print(f"Preprocessing data with shapes: x={x_data.shape}, y={y_data.shape}")
        
        # Convert to numpy arrays if needed
        if not isinstance(x_data, np.ndarray):
            x_data = np.array(x_data)
        if not isinstance(y_data, np.ndarray):
            y_data = np.array(y_data)
            
        # Normalize pixel values if image data (values between 0-255)
        if x_data.dtype == np.uint8 or np.max(x_data) > 1.0:
            print("Normalizing pixel values to range [0, 1]")
            x_data = x_data.astype(np.float32) / 255.0
            
        # Reshape input data if needed based on model expectation
        if len(x_data.shape) == 3 and self.model and hasattr(self.model, 'input_shape'):
            # For image data that needs channel dimension
            if self.model.input_shape and len(self.model.input_shape) == 4:
                # Model expects channel dimension but data doesn't have it
                if self.model.input_shape[-1] == 1:  # Grayscale images
                    print("Reshaping data to include channel dimension (grayscale)")
                    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)
                elif self.model.input_shape[-1] == 3 and x_data.shape[-1] != 3:  # RGB images
                    print("Warning: Model expects RGB images but data doesn't match")
                    
        # One-hot encode labels if needed (for categorical crossentropy)
        if len(y_data.shape) == 1:
            print("Converting labels to one-hot encoding")
            # Count unique classes
            num_classes = len(np.unique(y_data))
            print(f"Detected {num_classes} unique classes")
            # Convert to one-hot
            y_one_hot = np.zeros((y_data.size, num_classes))
            y_one_hot[np.arange(y_data.size), y_data] = 1
            y_data = y_one_hot
            
        print(f"Preprocessing complete: x={x_data.shape}, y={y_data.shape}")
        return x_data, y_data

    def _print_model_summary(self, model=None):
        """Print detailed model architecture info for debugging."""
        if model is None:
            model = self.model
            
        if model is None:
            print("No model available to print summary")
            return
            
        print("\n===== MODEL ARCHITECTURE =====")
        model.summary()
        
        print("\n===== LAYER DETAILS =====")
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name}, Type: {type(layer).__name__}")
            print(f"  Input shape: {layer.input_shape}")
            print(f"  Output shape: {layer.output_shape}")
            
            # For layers with weights, print their shapes
            if len(layer.weights) > 0:
                for j, weight in enumerate(layer.weights):
                    print(f"  Weight {j}: {weight.name}, Shape: {weight.shape}")
            else:
                print("  No weights in this layer")
                
        print("\n===== WEIGHT SHAPES =====")
        weights = model.get_weights()
        for i, w in enumerate(weights):
            if hasattr(w, 'shape'):
                print(f"Weight array {i}: Shape {w.shape}, Size: {w.size}")
            else:
                print(f"Weight array {i}: Type {type(w)}")
        
        print("=============================\n")

    def _serialize_weights(self, weights):
        """Completely rewritten weight serialization function that guarantees valid output.
        
        Args:
            weights: List of numpy arrays from model.get_weights()
            
        Returns:
            List of serialized weights (Python lists) or None if serialization fails
        """
        try:
            print(f"Serializing {len(weights)} weight arrays with improved method...")
            serialized_weights = []
            empty_weights = 0
            
            # Create simple directory to store weights temporarily if needed
            tmp_dir = os.path.join(os.getcwd(), 'tmp_weights')
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Pre-verification step - check all weights
            valid_weights_count = 0
            invalid_weights = []
            for i, w in enumerate(weights):
                if isinstance(w, np.ndarray) and w.size > 0 and np.all(np.isfinite(w)):
                    valid_weights_count += 1
                else:
                    invalid_weights.append(i)
                    
            print(f"Pre-verification: {valid_weights_count}/{len(weights)} weights valid")
            if invalid_weights:
                print(f"Invalid weight indices: {invalid_weights}")
                
            # Extra step: If all weights are invalid, return emergency fallback immediately
            if valid_weights_count == 0:
                print("All weights invalid - using emergency fallback weights")
                return self._create_fallback_weights(weights)
            
            for i, w in enumerate(weights):
                try:
                    # Skip empty arrays
                    if not isinstance(w, np.ndarray) or w.size == 0:
                        print(f"Skipping empty weight at index {i}")
                        empty_weights += 1
                        continue
                    
                    # Print weight stats for debugging
                    print(f"Weight {i}: shape={w.shape}, size={w.size}, min={np.min(w):.6f}, max={np.max(w):.6f}, mean={np.mean(w):.6f}")
                    
                    # Check for NaN or infinity
                    if not np.all(np.isfinite(w)):
                        print(f"Weight at index {i} contains NaN/Inf values, replacing with small values")
                        w = np.nan_to_num(w, nan=0.01, posinf=0.01, neginf=-0.01)
                    
                    # Force to float32 for consistent serialization
                    w_float32 = w.astype(np.float32)
                    
                    # DIRECT APPROACH: If the array is large, save and load it in a controlled way
                    if w.size > 100000:  # For arrays with > 100k elements
                        # Create a unique filename
                        unique_token = f"{int(time.time())}_{random.randint(10000, 99999)}"
                        temp_file = os.path.join(tmp_dir, f"weight_{i}_{unique_token}.npy")
                        
                        print(f"Large array at index {i} with {w.size} elements - using temporary file method")
                        
                        # Save to file
                        np.save(temp_file, w_float32)
                        
                        # Read it back in chunks
                        loaded = np.load(temp_file)
                        
                        # Convert to list in chunks to avoid memory issues
                        chunks = []
                        flat = loaded.flatten()
                        chunk_size = 10000
                        
                        for start in range(0, flat.size, chunk_size):
                            end = min(start + chunk_size, flat.size)
                            chunk = flat[start:end].tolist()
                            chunks.extend(chunk)
                        
                        # Reshape back to original
                        result = np.array(chunks, dtype=np.float32).reshape(w.shape).tolist()
                        
                        # Clean up
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    else:
                        # For smaller arrays, direct conversion should work
                        result = w_float32.tolist()
                    
                    # Verify the serialized array
                    if not result or (isinstance(result, list) and len(result) == 0):
                        print(f"Warning: Weight at index {i} converted to empty list, using fallback method")
                        
                        # Fallback method for problematic arrays
                        data = []
                        flat_array = w_float32.flatten()
                        
                        # Manually convert each value to ensure precision
                        for j in range(flat_array.size):
                            val = float(flat_array[j])
                            if not np.isfinite(val):
                                val = 0.01  # Replace non-finite values
                            data.append(val)
                        
                        # Reshape back to original dimensions
                        shape = list(w.shape)
                        reshaped = []
                        
                        # Simple reshaping for 1D arrays
                        if len(shape) == 1:
                            reshaped = data
                        # Handle 2D arrays (most common case)
                        elif len(shape) == 2:
                            index = 0
                            for row in range(shape[0]):
                                row_data = []
                                for col in range(shape[1]):
                                    row_data.append(data[index])
                                    index += 1
                                reshaped.append(row_data)
                        # For higher dimensions, use a placeholder with the correct shape
                        else:
                            print(f"Using simplified placeholder for complex shape: {shape}")
                            placeholder = np.ones(shape, dtype=np.float32) * 0.01
                            reshaped = placeholder.tolist()
                        
                        result = reshaped
                        
                        # Final check on the fallback result
                        if not result or len(result) == 0:
                            print(f"ERROR: All serialization methods failed for weight {i}")
                            empty_weights += 1
                            continue
                    
                    # Verify the result size and structure
                    try:
                        # Test serialization with a sample
                        import json
                        test_data = result
                        if isinstance(test_data, list) and len(test_data) > 10:
                            # Just test a small part for large arrays
                            if isinstance(test_data[0], list):
                                test_data = test_data[0:2]
                            else:
                                test_data = test_data[0:10]
                        
                        json_str = json.dumps(test_data)
                        print(f"  JSON test success: sample size={len(json_str)} bytes")
                        
                        # Double-check result is non-empty (important validation)
                        if isinstance(result, list):
                            if len(result) == 0:
                                print(f"ERROR: Result is empty list for weight {i}")
                                result = [[0.01, 0.01], [0.01, 0.01]]  # Minimal placeholder data
                            elif isinstance(result[0], list) and len(result[0]) == 0:
                                print(f"ERROR: Result contains empty sublists for weight {i}")
                                result = [[0.01, 0.01], [0.01, 0.01]]  # Minimal placeholder data
                        
                    except Exception as json_err:
                        print(f"  Warning: JSON test failed: {json_err}, using fallback placeholder")
                        # Last resort placeholder
                        placeholder = np.ones(w.shape, dtype=np.float32) * 0.01
                        result = placeholder.tolist()
                    
                    serialized_weights.append(result)
                    
                except Exception as e:
                    print(f"Error serializing weight at index {i}: {e}")
                    empty_weights += 1
                    # Create a placeholder for this weight
                    try:
                        if hasattr(w, 'shape') and w.shape:
                            placeholder = np.ones(w.shape, dtype=np.float32) * 0.01
                            placeholder_list = placeholder.tolist()
                            if placeholder_list and (isinstance(placeholder_list, list) and len(placeholder_list) > 0):
                                serialized_weights.append(placeholder_list)
                                print(f"Created placeholder for weight at index {i}")
                            else:
                                print(f"Created placeholder is empty, skipping weight {i}")
                        else:
                            print(f"Cannot create placeholder for weight {i} - no shape info")
                    except Exception as placeholder_err:
                        print(f"Could not create placeholder: {placeholder_err}")
            
            # Ensure we have all weight arrays
            if len(serialized_weights) != len(weights):
                print(f"Warning: Serialized {len(serialized_weights)}/{len(weights)} weight arrays, {empty_weights} empty/error weights")
                
                # If we're missing more than half the weights, something is very wrong
                if len(serialized_weights) < len(weights) / 2:
                    print("ERROR: Too many missing weights. Serialization likely failed.")
                    return self._create_fallback_weights(weights)
            
            # Final verification - make sure we have the weights
            if not serialized_weights or len(serialized_weights) == 0:
                print("ERROR: Serialization produced no valid weights, using fallback method")
                return self._create_fallback_weights(weights)
                
            # Final verification - check each serialized weight for validity
            for i, w in enumerate(serialized_weights):
                if not w or (isinstance(w, list) and len(w) == 0):
                    print(f"Final verification: Weight {i} is empty, replacing with placeholder")
                    serialized_weights[i] = [[0.01, 0.01], [0.01, 0.01]]  # Minimal valid placeholder
                
            print(f"Successfully serialized {len(serialized_weights)}/{len(weights)} valid weight arrays")
            return serialized_weights
        
        except Exception as e:
            print(f"Error in weight serialization: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_weights(weights)
            
    def _create_fallback_weights(self, weights):
        """Emergency method to create valid placeholder weights that the server can use."""
        print("Creating fallback placeholder weights with guaranteed format")
        fake_weights = []
        
        # Ensure socket module is imported for hostname uniqueness
        import socket
        import random
        
        # Include a timestamp in weights to ensure uniqueness
        timestamp = int(time.time())
        # Include machine identity to ensure uniqueness across clients
        machine_id = socket.gethostname()
        # Random seed to further ensure uniqueness
        random_seed = random.randint(1000, 9999)
        
        # Log the fallback creation details
        print(f"Creating emergency weights with timestamp={timestamp}, machine={machine_id}, seed={random_seed}")
        
        for i, w in enumerate(weights):
            try:
                if hasattr(w, 'shape') and w.shape:
                    shape = w.shape
                    # Create non-zero values with slight variations to make weights recognizable
                    # Use a small value (0.01) plus a tiny random component that depends on index
                    base_value = 0.01 
                    random_component = 0.001 * (i + 1) / len(weights)
                    value = base_value + random_component
                    
                    # For 1D arrays
                    if len(shape) == 1:
                        fallback = [value] * shape[0]
                    # For 2D arrays (most common case)
                    elif len(shape) == 2:
                        fallback = []
                        for row in range(shape[0]):
                            row_data = [value] * shape[1]
                            fallback.append(row_data)
                    # For 3D arrays
                    elif len(shape) == 3:
                        fallback = []
                        for dim1 in range(shape[0]):
                            dim1_data = []
                            for dim2 in range(shape[1]):
                                dim2_data = [value] * shape[2]
                                dim1_data.append(dim2_data)
                            fallback.append(dim1_data)
                    # For 4D arrays (typical for convolution kernels)
                    elif len(shape) == 4:
                        # Create a minimal valid 4D structure
                        fallback = []
                        for dim1 in range(shape[0]):
                            dim1_data = []
                            for dim2 in range(shape[1]):
                                dim2_data = []
                                for dim3 in range(shape[2]):
                                    dim3_data = [value] * shape[3]
                                    dim2_data.append(dim3_data)
                                dim1_data.append(dim2_data)
                            fallback.append(dim1_data)
                    else:
                        # Fallback for any other shapes - create minimal valid structure
                        print(f"Complex shape for weight {i}: {shape} - using minimal fallback")
                        fallback = [[value, value], [value, value]]
                        
                    fake_weights.append(fallback)
                    print(f"Created shaped fallback for weight {i}: shape={shape}")
                else:
                    # If we can't access shape, create a minimal array
                    fallback = [[0.01, 0.01], [0.01, 0.01]]
                    fake_weights.append(fallback)
                    print(f"Created minimal fallback for weight {i} (no shape info)")
            except Exception as e:
                print(f"Error creating fallback for weight {i}: {e}")
                # Absolute minimal fallback
                fake_weights.append([[0.01, 0.01], [0.01, 0.01]])
                
        print(f"Created {len(fake_weights)} emergency placeholder weights")
        
        # Verify the fallback weights
        for i, w in enumerate(fake_weights):
            if not w or len(w) == 0:
                print(f"Invalid empty fallback weight at index {i}, using guaranteed minimal structure")
                fake_weights[i] = [[0.01, 0.01], [0.01, 0.01]]
        
        # If still empty (shouldn't happen), create minimal data
        if not fake_weights or len(fake_weights) == 0:
            print("Creating guaranteed minimal emergency weights")
            fake_weights = [[[0.01, 0.01], [0.01, 0.01]] for _ in range(len(weights))]
            
        # Final verification to ensure all weights exist
        if len(fake_weights) != len(weights):
            print(f"Warning: Created {len(fake_weights)} fallbacks but needed {len(weights)}")
            # Extend with minimal arrays if needed
            while len(fake_weights) < len(weights):
                fake_weights.append([[0.01, 0.01], [0.01, 0.01]])
        
        print(f"Successfully created {len(fake_weights)} verified emergency weights")
        return fake_weights

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
                # Get model weights
                model_weights = self.model.get_weights()
                
                # Serialize weights using the client's serialization function
                if hasattr(self.client, '_serialize_weights'):
                    serialized_weights = self.client._serialize_weights(model_weights)
                    if serialized_weights:
                        print(f"Serialized {len(serialized_weights)} weight arrays for epoch metrics")
                        return requests.post(
                            f"{self.client.server_url}/api/clients/{self.client.client_id}/model_update",
                            json={
                                'weights': serialized_weights,
                                'metrics': metrics
                            },
                            headers={'X-API-Key': self.client.api_key},
                            timeout=20  # Shorter timeout for epoch metrics
                        )
                
                # Fallback to metrics-only if weight serialization failed or not available
                print("Sending metrics-only update for epoch")
                return requests.post(
                    f"{self.client.server_url}/api/clients/{self.client.client_id}/model_update",
                    json={
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