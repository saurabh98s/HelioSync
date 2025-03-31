#!/usr/bin/env python3
"""
Federated Learning - Sentiment Analysis Example (Server)

This script starts a federated learning server for the sentiment analysis task.
"""

import sys
import os

# Add project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from server.server import FederatedServer
import argparse

def main():
    """Main function to start the federated learning server for sentiment analysis."""
    parser = argparse.ArgumentParser(description="Sentiment Analysis Federated Learning Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--min_clients", type=int, default=2, help="Minimum number of clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated learning rounds")
    parser.add_argument("--model_path", type=str, default="models/sentiment_federated", 
                       help="Path to save/load model")
    parser.add_argument("--fraction_fit", type=float, default=1.0, 
                       help="Fraction of clients to sample in each round")
    
    args = parser.parse_args()
    
    print("Starting Sentiment Analysis Federated Learning Server...")
    print(f"Settings: min_clients={args.min_clients}, rounds={args.rounds}")
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Start the federated learning server
    server = FederatedServer(
        min_clients=args.min_clients,
        rounds=args.rounds,
        model_path=args.model_path,
        fraction_fit=args.fraction_fit
    )
    
    server.start(port=args.port)

if __name__ == "__main__":
    main() 