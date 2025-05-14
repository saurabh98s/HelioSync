#!/usr/bin/env python3
"""
Client Runner Script

This script provides a command-line interface for running federated learning clients.
"""

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from fl_client.client import FederatedClient

def load_mnist_data():
    """Load MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_mnist_model():
    """Create a simple CNN model for MNIST that matches the server model architecture."""
    model = tf.keras.Sequential([
        # First Conv Block
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),  # First MaxPooling: 28x28 -> 14x14
        
        # Second Conv Block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        # NO second MaxPooling to match server model architecture!
        
        # Flatten layer - 14*14*64 = 12544 neurons
        tf.keras.layers.Flatten(),
        
        # Dense layers
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main entry point for the client runner."""
    parser = argparse.ArgumentParser(description='Run a federated learning client')
    parser.add_argument('--client_id', type=str, required=True,
                        help='Unique identifier for this client')
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key for authentication')
    parser.add_argument('--server_url', type=str, default='http://localhost:5000',
                        help='URL of the federated learning server')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs for local training')
    parser.add_argument('--data_split', type=float, default=0.1,
                        help='Fraction of data to use for training')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Split data for this client
    split_idx = int(len(x_train) * args.data_split)
    
    # Use a random subset of data to ensure variety between clients
    indices = np.random.permutation(len(x_train))[:split_idx]
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    # Use smaller test set
    test_indices = np.random.permutation(len(x_test))[:1000]  # Use only 1000 test samples
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]
    
    # Print dataset size information
    print(f"Training on {len(x_train)} samples (reduced dataset size)")
    print(f"Test set has {len(x_test)} samples")
    
    # Create model
    print("Creating model...")
    model = create_mnist_model()
    
    # Create and start client
    print(f"Starting client {args.client_id}...")
    client = FederatedClient(
        client_id=args.client_id,
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        api_key=args.api_key,
        server_url=args.server_url
    )
    
    try:
        client.start()
        print("Client started successfully. Press Ctrl+C to stop.")
        
        # Keep the main thread alive
        while True:
            pass
            
    except KeyboardInterrupt:
        print("\nStopping client...")
        client.stop()
        print("Client stopped.")

if __name__ == "__main__":
    main() 