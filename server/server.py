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
from server.metrics_collector import MetricsCollector

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
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(save_dir=os.path.join(model_path, "metrics"))
        
        # Create directories
        os.makedirs(model_path, exist_ok=True)
    
    def start(self, port: int = 8080):
        """Start the federated learning server.
        
        Args:
            port: The port to run the server on.
        """
        # Define strategy with callbacks
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=self.fraction_fit,
            min_fit_clients=self.min_clients,
            min_available_clients=self.min_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config
        )
        
        # Add callbacks for metrics collection
        self.add_metrics_callbacks(strategy)
        
        # Start server
        fl.server.start_server(
            server_address=f"0.0.0.0:{port}",
            config=fl.server.ServerConfig(num_rounds=self.rounds),
            strategy=strategy
        )
    
    def fit_config(self, server_round: int) -> Dict[str, fl.common.Scalar]:
        """Return training configuration for clients.
        
        Args:
            server_round: The current round number.
            
        Returns:
            Configuration dictionary for client training.
        """
        return {
            "epoch": 1,  # Local epochs
            "batch_size": 32,
            "round": server_round,
            "learning_rate": 0.001
        }
    
    def evaluate_config(self, server_round: int) -> Dict[str, fl.common.Scalar]:
        """Return evaluation configuration for clients.
        
        Args:
            server_round: The current round number.
            
        Returns:
            Configuration dictionary for client evaluation.
        """
        return {
            "round": server_round,
            "batch_size": 32
        }
    
    def add_metrics_callbacks(self, strategy: fl.server.strategy.Strategy):
        """Add callbacks to the strategy for metrics collection.
        
        Args:
            strategy: The federated learning strategy.
        """
        # Store original methods
        original_aggregate_fit = strategy.aggregate_fit
        original_aggregate_evaluate = strategy.aggregate_evaluate
        
        def aggregate_fit_with_metrics(
            server_round: int,
            results: List[Tuple[ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
        ) -> Optional[fl.common.Parameters]:
            """Aggregate training results with metrics collection."""
            aggregated = original_aggregate_fit(server_round, results, failures)
            
            if aggregated is not None:
                # Collect metrics from this round
                metrics = {}
                for _, fit_res in results:
                    metrics.update(fit_res.metrics)
                
                self.metrics_collector.add_round_metrics(
                    round_num=server_round,
                    metrics=metrics,
                    num_clients=len(results),
                    total_clients=len(results) + len(failures)
                )
                
                # Save model if needed
                if server_round % 5 == 0:  # Save every 5 rounds
                    save_model(aggregated, os.path.join(
                        self.model_path, f"model_round_{server_round}.h5"
                    ))
            
            return aggregated
        
        def aggregate_evaluate_with_metrics(
            server_round: int,
            results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
            failures: List[BaseException],
        ) -> Optional[float]:
            """Aggregate evaluation results with metrics collection."""
            aggregated = original_aggregate_evaluate(server_round, results, failures)
            
            if aggregated is not None:
                # Collect evaluation metrics
                for client_proxy, eval_res in results:
                    self.metrics_collector.add_client_metrics(
                        client_id=client_proxy.cid,
                        round_num=server_round,
                        metrics=eval_res.metrics
                    )
            
            return aggregated
        
        # Replace strategy methods with our wrapped versions
        strategy.aggregate_fit = aggregate_fit_with_metrics
        strategy.aggregate_evaluate = aggregate_evaluate_with_metrics

def main():
    """Main function to start the federated learning server."""
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port to run the server on")
    parser.add_argument("--min_clients", type=int, default=2,
                       help="Minimum number of clients required to start training")
    parser.add_argument("--rounds", type=int, default=5,
                       help="Number of federated learning rounds")
    parser.add_argument("--model_path", type=str, default="model",
                       help="Path to save/load model")
    parser.add_argument("--fraction_fit", type=float, default=1.0,
                       help="Fraction of clients to sample in each round")
    
    args = parser.parse_args()
    
    # Create and start server
    server = FederatedServer(
        min_clients=args.min_clients,
        rounds=args.rounds,
        model_path=args.model_path,
        fraction_fit=args.fraction_fit
    )
    
    server.start(port=args.port)

if __name__ == "__main__":
    main() 