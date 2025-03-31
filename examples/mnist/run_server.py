#!/usr/bin/env python3
"""
Federated Learning - MNIST Example Server

This script runs a federated learning server for the MNIST dataset.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
import tensorflow as tf

# Add parent directory to path so we can import from fl_server
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fl_server.server import FederatedServer
from fl_server.models.mnist import create_mnist_model

def main():
    """Run the MNIST federated learning server."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run MNIST federated learning server')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to listen on')
    parser.add_argument('--min_clients', type=int, default=2, help='Minimum number of clients')
    parser.add_argument('--rounds', type=int, default=5, help='Number of training rounds')
    parser.add_argument('--model_path', type=str, default='models/mnist', help='Path to save model')
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_path, exist_ok=True)
    
    # Load MNIST dataset for testing
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Create the model
    model = create_mnist_model()
    
    # Create server
    server = FederatedServer(
        model=model,
        host=args.host,
        port=args.port,
        min_clients=args.min_clients,
        rounds=args.rounds
    )
    
    # Define callback for end of round
    def round_callback(round_num, global_model, metrics):
        """Callback for end of round."""
        print(f"Finished round {round_num}")
        print(f"Metrics: {metrics}")
        
        # Evaluate on test data
        results = global_model.evaluate(x_test, y_test, verbose=0)
        test_loss, test_accuracy = results
        
        # Save metrics
        round_metrics = {
            'round': round_num,
            'loss': float(test_loss),
            'accuracy': float(test_accuracy),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to metrics file
        metrics_path = os.path.join(args.model_path, 'metrics.json')
        metrics_data = []
        
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
            except json.JSONDecodeError:
                metrics_data = []
        
        metrics_data.append(round_metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f)
        
        # Save model
        model_path = os.path.join(args.model_path, f'model_round_{round_num}.h5')
        global_model.save(model_path)
        
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
        print(f"Model saved to {model_path}")
        
        # Update the central server (this could be calling the web API to update the project)
        # For now, just print a message
        print(f"Updating central server with round {round_num} metrics")
    
    # Define callback for end of training
    def final_callback(global_model, metrics):
        """Callback for end of training."""
        print("FL finished")
        
        # Evaluate on test data
        results = global_model.evaluate(x_test, y_test, verbose=0)
        test_loss, test_accuracy = results
        
        # Save final metrics
        final_metrics = {
            'loss': float(test_loss),
            'accuracy': float(test_accuracy),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save metrics to file
        with open(os.path.join(args.model_path, 'metrics.json'), 'w') as f:
            json.dump(final_metrics, f)
        
        # Save model
        model_path = os.path.join(args.model_path, 'final_model.h5')
        global_model.save(model_path)
        
        print(f"Final test loss: {test_loss:.4f}, Final test accuracy: {test_accuracy:.4f}")
        print(f"Final model saved to {model_path}")
    
    # Start server
    server.start(round_callback=round_callback, final_callback=final_callback)

if __name__ == '__main__':
    main() 