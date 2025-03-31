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

class FederatedClient:
    """
    Federated Learning Client implementation.
    
    This client communicates with a Federated Learning server and performs
    local training on its own data.
    """
    
    def __init__(self, client_id, model, x_train, y_train, x_test=None, y_test=None, 
                 batch_size=32, epochs=5):
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
        """
        self.client_id = client_id
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize socket and connection
        self.sock = None
        self.connected = False
        self.running = False
        self.current_weights = None
        
        # Track training metrics
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': None,
            'test_accuracy': None,
            'training_time': 0
        }
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def start(self, server_host, server_port):
        """
        Connect to the server and start the client.
        
        Args:
            server_host: Server hostname or IP.
            server_port: Server port.
        """
        # Create a socket connection to the server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((server_host, server_port))
            self.connected = True
            print(f"Connected to server at {server_host}:{server_port}")
            
            # Start the client thread
            self.running = True
            self.client_thread = threading.Thread(target=self._run)
            self.client_thread.daemon = True
            self.client_thread.start()
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            self.sock = None
    
    def _run(self):
        """Main client thread that handles communication with the server."""
        try:
            # Send initial registration message
            self._send_message({
                'type': 'register',
                'client_id': self.client_id,
                'data_size': len(self.x_train)
            })
            
            # Main communication loop
            while self.running:
                message = self._receive_message()
                if not message:
                    print("Connection closed by server.")
                    break
                
                self._handle_message(message)
                
        except Exception as e:
            print(f"Error in client thread: {e}")
        finally:
            self.connected = False
            if self.sock:
                self.sock.close()
                self.sock = None
    
    def _handle_message(self, message):
        """
        Handle a message received from the server.
        
        Args:
            message: The received message.
        """
        msg_type = message.get('type')
        
        if msg_type == 'train':
            # Server is requesting training
            weights = message.get('weights')
            round_num = message.get('round')
            
            print(f"Received training request for round {round_num}")
            
            # Convert weights from list to numpy arrays
            model_weights = [np.array(w) for w in weights]
            
            # Set model weights
            self.model.set_weights(model_weights)
            
            # Train the model
            training_result = self._train()
            
            # Send the updated weights back to the server
            self._send_message({
                'type': 'update',
                'client_id': self.client_id,
                'weights': [w.tolist() for w in self.model.get_weights()],
                'metrics': training_result,
                'round': round_num
            })
            
        elif msg_type == 'evaluate':
            # Server is requesting evaluation
            weights = message.get('weights')
            
            # Convert weights from list to numpy arrays
            model_weights = [np.array(w) for w in weights]
            
            # Set model weights
            self.model.set_weights(model_weights)
            
            # Evaluate the model
            eval_result = self._evaluate()
            
            # Send the evaluation results back to the server
            self._send_message({
                'type': 'eval_result',
                'client_id': self.client_id,
                'metrics': eval_result
            })
        
        elif msg_type == 'complete':
            # Training is complete
            print("Federated learning process completed.")
            
        elif msg_type == 'error':
            # Server reported an error
            print(f"Server error: {message.get('message')}")
    
    def _train(self):
        """
        Train the model on local data.
        
        Returns:
            dict: Training metrics.
        """
        print(f"Training with {len(self.x_train)} samples for {self.epochs} epochs")
        
        # Setup training
        start_time = time.time()
        
        # Compile the model if it hasn't been compiled yet
        if not self.model.optimizer:
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Train the model
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )
        
        # Record training time
        training_time = time.time() - start_time
        
        # Record metrics
        train_loss = float(history.history['loss'][-1])
        train_accuracy = float(history.history['accuracy'][-1])
        
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_accuracy'].append(train_accuracy)
        self.metrics['training_time'] = training_time
        
        # Evaluate on test data if available
        if self.x_test is not None and self.y_test is not None:
            test_loss, test_accuracy = self.model.evaluate(
                self.x_test, self.y_test,
                verbose=0
            )
            self.metrics['test_loss'] = float(test_loss)
            self.metrics['test_accuracy'] = float(test_accuracy)
        
        print(f"Training completed in {training_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        return {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': self.metrics['test_loss'],
            'test_accuracy': self.metrics['test_accuracy'],
            'training_time': training_time,
            'samples': len(self.x_train)
        }
    
    def _evaluate(self):
        """
        Evaluate the model on local test data.
        
        Returns:
            dict: Evaluation metrics.
        """
        if self.x_test is None or self.y_test is None:
            return {'error': 'No test data available'}
        
        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(
            self.x_test, self.y_test,
            verbose=0
        )
        
        # Update metrics
        self.metrics['test_loss'] = float(test_loss)
        self.metrics['test_accuracy'] = float(test_accuracy)
        
        print(f"Evaluation - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'samples': len(self.x_test)
        }
    
    def _send_message(self, message):
        """
        Send a message to the server.
        
        Args:
            message: The message to send.
        """
        if not self.connected or not self.sock:
            print("Not connected to server.")
            return False
        
        try:
            # Convert message to JSON and encode as bytes
            message_bytes = json.dumps(message).encode('utf-8')
            
            # Send message length first, then the message
            message_length = len(message_bytes)
            self.sock.sendall(message_length.to_bytes(4, byteorder='big'))
            self.sock.sendall(message_bytes)
            
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            self.connected = False
            return False
    
    def _receive_message(self):
        """
        Receive a message from the server.
        
        Returns:
            dict: The received message, or None if there was an error.
        """
        if not self.connected or not self.sock:
            print("Not connected to server.")
            return None
        
        try:
            # Receive message length first (4 bytes)
            length_bytes = self.sock.recv(4)
            if not length_bytes:
                return None
            
            message_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Receive the full message
            message_bytes = b''
            while len(message_bytes) < message_length:
                chunk = self.sock.recv(min(4096, message_length - len(message_bytes)))
                if not chunk:
                    return None
                message_bytes += chunk
            
            # Decode and parse the message
            message = json.loads(message_bytes.decode('utf-8'))
            return message
        except Exception as e:
            print(f"Error receiving message: {e}")
            self.connected = False
            return None
    
    def wait(self):
        """Wait for the client thread to finish."""
        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.join()
    
    def shutdown(self):
        """Shutdown the client."""
        self.running = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False 