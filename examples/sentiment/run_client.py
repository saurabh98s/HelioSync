#!/usr/bin/env python3
"""
Federated Learning - Sentiment Analysis Example (Client)

This script starts a federated learning client for the sentiment analysis task.
"""

import sys
import os

# Add project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from client.client import FederatedClient
from client.data_loader import load_dataset
import argparse
import tensorflow as tf
import torch

def main():
    """Main function to start a federated learning client for sentiment analysis."""
    parser = argparse.ArgumentParser(description="Sentiment Analysis Federated Learning Client")
    parser.add_argument("--server_address", type=str, default="localhost:8080",
                       help="Server address in the format host:port")
    parser.add_argument("--client_id", type=str, required=True,
                       help="Unique identifier for this client")
    parser.add_argument("--framework", type=str, default="tensorflow",
                       choices=["tensorflow", "pytorch"],
                       help="Framework to use (tensorflow or pytorch)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--local_epochs", type=int, default=3,
                       help="Number of local epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                       help="Learning rate for optimization")
    parser.add_argument("--non_iid", action="store_true",
                       help="Use non-IID data partitioning")
    parser.add_argument("--num_clients", type=int, default=10,
                       help="Total number of clients in the system")
    
    args = parser.parse_args()
    
    print(f"Starting Sentiment Analysis Federated Learning Client (ID: {args.client_id})...")
    print(f"Framework: {args.framework}, IID: {not args.non_iid}")
    
    # Load dataset
    train_data, test_data = load_dataset(
        dataset_name="sentiment",
        client_id=args.client_id,
        num_clients=args.num_clients,
        iid=not args.non_iid
    )
    
    # Load model based on framework
    if args.framework.lower() == "tensorflow":
        from client.models.tf_models import create_model
        model = create_model("sentiment")
    else:  # pytorch
        from client.models.torch_models import create_model
        model = create_model("sentiment")
    
    # Create and start client
    client = FederatedClient(
        client_id=args.client_id,
        model=model,
        train_data=train_data,
        test_data=test_data,
        framework=args.framework,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Start Flower client
    import flwr as fl
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main() 