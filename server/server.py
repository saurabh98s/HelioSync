#!/usr/bin/env python3
"""
Federated Learning - Server Implementation

This module implements the server-side logic for federated learning.
The server coordinates the training process across multiple clients.
"""

import argparse
import flwr as fl
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy

# Import local modules
from server.aggregation import FedAvg
from server.utils import save_model, load_model

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from clients using weighted averaging."""
    # Check if metrics list is empty
    if len(metrics) == 0:
        return {}

    # Calculate weighted average for each metric
    weighted_metrics = {}
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    for metric_name in metrics[0][1].keys():
        weighted_metrics[metric_name] = sum(
            [value[metric_name] * num_examples for num_examples, value in metrics]
        ) / total_examples
    
    return weighted_metrics

class FederatedServer:
    """Server for federated learning implementation."""
    
    def __init__(self, min_clients: int = 2, rounds: int = 3, 
                 model_path: str = "model", fraction_fit: float = 1.0):
        """Initialize the federated server.
        
        Args:
            min_clients: Minimum number of clients required to start training.
            rounds: Number of federated learning rounds.
            model_path: Path to save/load model.
            fraction_fit: Fraction of clients to sample in each round.
        """
        self.min_clients = min_clients
        self.rounds = rounds
        self.model_path = model_path
        self.fraction_fit = fraction_fit
        
    def start(self, port: int = 8080):
        """Start the federated learning server.
        
        Args:
            port: The port to run the server on.
        """
        # Define strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=self.fraction_fit,
            min_fit_clients=self.min_clients,
            min_available_clients=self.min_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        
        # Start server
        fl.server.start_server(
            server_address=f"0.0.0.0:{port}",
            config=fl.server.ServerConfig(num_rounds=self.rounds),
            strategy=strategy,
        )

def main():
    """Main function to start the federated learning server."""
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--min_clients", type=int, default=2, help="Minimum number of clients")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--model_path", type=str, default="model", help="Path to save/load model")
    parser.add_argument("--fraction_fit", type=float, default=1.0, 
                       help="Fraction of clients to sample in each round")
    
    args = parser.parse_args()
    
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