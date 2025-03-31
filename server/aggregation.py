#!/usr/bin/env python3
"""
Federated Learning - Aggregation Strategies

This module implements various aggregation strategies for federated learning.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as FlowerFedAvg

class FedAvg(FlowerFedAvg):
    """Federated Averaging (FedAvg) strategy.
    
    This implementation extends the Flower FedAvg strategy with additional
    functionality for model saving and loading.
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        *args,
        **kwargs,
    ):
        """Initialize FedAvg strategy.
        
        Args:
            fraction_fit: Fraction of clients to use for training.
            fraction_evaluate: Fraction of clients to use for evaluation.
            min_fit_clients: Minimum number of clients for training.
            min_evaluate_clients: Minimum number of clients for evaluation.
            min_available_clients: Minimum number of available clients.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            *args,
            **kwargs,
        )
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients.
        
        Args:
            server_round: Current round of federated learning.
            results: List of tuples (client, fit_res) containing client updates.
            failures: List of failures that occurred during the fitting process.
            
        Returns:
            parameters_aggregated: Aggregated model parameters.
            metrics: Dict containing aggregated metrics.
        """
        # Call aggregate_fit from parent class (FedAvg)
        parameters_aggregated, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if parameters_aggregated is not None:
            # Add custom metrics if needed
            metrics["num_clients"] = len(results)
            
        return parameters_aggregated, metrics

class FedProx(FedAvg):
    """FedProx strategy (FedAvg with proximal term).
    
    This strategy extends FedAvg with a proximal term to improve convergence
    when client data is heterogeneous.
    """
    
    def __init__(
        self,
        mu: float = 0.01,  # Proximal term parameter
        *args,
        **kwargs,
    ):
        """Initialize FedProx strategy.
        
        Args:
            mu: Proximal term parameter (higher values enforce
                more similarity to the global model).
        """
        super().__init__(*args, **kwargs)
        self.mu = mu
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training.
        
        Args:
            server_round: Current round of federated learning.
            parameters: Parameters to be used for the next training round.
            client_manager: Client manager that selects clients for training.
            
        Returns:
            A list of tuples (client, fit_configuration).
        """
        # Get clients and their configurations from parent class
        client_configs = super().configure_fit(server_round, parameters, client_manager)
        
        # Add proximal term parameter to configurations
        for client, config in client_configs:
            if config["config"] is None:
                config["config"] = {}
            config["config"]["mu"] = self.mu
            
        return client_configs 