#!/usr/bin/env python3
"""
Federated Learning - MNIST Example Client

This script runs a federated learning client for the MNIST dataset.
"""

import os
import sys
import json
import time
import argparse
import socket
import requests
import numpy as np
from datetime import datetime
import tensorflow as tf

# Add parent directory to path so we can import from fl_client
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fl_client.client import FederatedClient
from fl_server.models.mnist import create_mnist_model

def main():
    """Run the MNIST federated learning client."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run MNIST federated learning client')
    parser.add_argument('--server', type=str, default='localhost:8080', help='Server address')
    parser.add_argument('--data_dir', type=str, default='data/mnist', help='Path to save/load data')
    parser.add_argument('--client_id', type=str, default=None, help='Client ID (default: hostname)')
    parser.add_argument('--api_key', type=str, required=True, help='API key for the federated learning platform')
    parser.add_argument('--api_url', type=str, default='http://localhost:5000/api', help='API URL for the federated learning platform')
    parser.add_argument('--epochs', type=int, default=5, help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    
    # Set client ID
    client_id = args.client_id or socket.gethostname()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Load MNIST dataset (for this example, we'll use the full dataset and shard it)
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess data
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Shard the data (in a real scenario, each client would have its own data)
    # For this example, we'll randomly sample the dataset
    np.random.seed(int.from_bytes(client_id.encode(), 'little') % 2**32)
    indices = np.random.choice(len(x_train), 5000, replace=False)
    x_train_shard = x_train[indices]
    y_train_shard = y_train[indices]
    
    # Create model
    model = create_mnist_model()
    
    # Register with the platform
    print(f"Registering client with ID: {client_id}")
    
    headers = {'X-API-Key': args.api_key}
    client_specs = {
        'platform': sys.platform,
        'python_version': sys.version,
        'tensorflow_version': tf.__version__,
        'num_samples': len(x_train_shard)
    }
    
    register_url = f"{args.api_url}/register_client"
    try:
        response = requests.post(
            register_url,
            headers=headers,
            json={
                'client_id': client_id,
                'name': f"MNIST-Client-{client_id}",
                'specs': client_specs
            }
        )
        response.raise_for_status()
        print("Client registered successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Failed to register client: {e}")
        return
    
    # Start the client
    client = FederatedClient(
        client_id=client_id,
        model=model,
        x_train=x_train_shard,
        y_train=y_train_shard,
        x_test=x_test,
        y_test=y_test,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Connect to the server
    server_host, server_port = args.server.split(':')
    print(f"Connecting to server at {server_host}:{server_port}")
    
    # Start the client (this will listen for training requests)
    client.start(server_host, int(server_port))
    
    # Heartbeat thread to update the platform
    def heartbeat():
        """Send heartbeat to the platform."""
        while True:
            try:
                heartbeat_url = f"{args.api_url}/clients/{client_id}/heartbeat"
                response = requests.post(
                    heartbeat_url,
                    headers=headers,
                    json={'specs': client_specs}
                )
                response.raise_for_status()
                print("Heartbeat sent successfully!")
            except requests.exceptions.RequestException as e:
                print(f"Failed to send heartbeat: {e}")
            
            # Sleep for 30 seconds
            time.sleep(30)
    
    # Start the heartbeat thread
    import threading
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    
    # Wait for the client to finish
    try:
        client.wait()
    except KeyboardInterrupt:
        print("Client interrupted, shutting down...")
        client.shutdown()

if __name__ == '__main__':
    main() 