#!/usr/bin/env python3
"""
Federated Learning - Client Implementation

This module implements the client-side logic for federated learning.
Each client trains on local data and shares model updates with the server.
"""

import argparse
import flwr as fl
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union
import time

# TensorFlow and PyTorch imports
import tensorflow as tf
import torch
from torch.utils.data import DataLoader

# Import local modules
from client.data_loader import load_dataset

class FederatedClient(fl.client.NumPyClient):
    """Client for federated learning implementation."""
    
    def __init__(
        self,
        client_id: str,
        model: Union[tf.keras.Model, torch.nn.Module],
        train_data,
        test_data,
        framework: str = "tensorflow",
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
    ):
        """Initialize the federated client.
        
        Args:
            client_id: Unique identifier for the client.
            model: Machine learning model to train.
            train_data: Training data.
            test_data: Testing data.
            framework: The framework of the model ('tensorflow' or 'pytorch').
            local_epochs: Number of epochs to train locally.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimization.
        """
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.framework = framework.lower()
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        if self.framework == "tensorflow":
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        elif self.framework == "pytorch":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate
            )
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def get_parameters(self, config):
        """Get model parameters.
        
        Args:
            config: Configuration from the server.
            
        Returns:
            Model parameters as a list of numpy arrays.
        """
        if self.framework == "tensorflow":
            return [np.array(w) for w in self.model.get_weights()]
        else:  # PyTorch
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters.
        
        Args:
            parameters: Model parameters as a list of numpy arrays.
        """
        if self.framework == "tensorflow":
            self.model.set_weights(parameters)
        else:  # PyTorch
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.Tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model on local data.
        
        Args:
            parameters: Initial model parameters.
            config: Configuration from the server.
            
        Returns:
            Updated model parameters and metrics.
        """
        print(f"Client {self.client_id}: Starting local training...")
        start_time = time.time()
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Adjust local parameters based on server config
        epochs = config.get("epochs", self.local_epochs)
        batch_size = config.get("batch_size", self.batch_size)
        
        # Handle FedProx mu parameter if present
        mu = config.get("mu", 0.0)
        
        if self.framework == "tensorflow":
            # Train on TensorFlow
            history = self.model.fit(
                self.train_data[0],
                self.train_data[1],
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )
            metrics = {
                "loss": history.history["loss"][-1],
                "accuracy": history.history["accuracy"][-1],
            }
        else:  # PyTorch
            # Train on PyTorch
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            # Create DataLoader
            if isinstance(self.train_data, DataLoader):
                train_loader = self.train_data
            else:
                # Assuming train_data is a tuple of (inputs, labels)
                train_dataset = torch.utils.data.TensorDataset(
                    torch.Tensor(self.train_data[0]), 
                    torch.LongTensor(self.train_data[1])
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
            
            # Training loop
            for epoch in range(epochs):
                running_loss = 0.0
                for inputs, labels in train_loader:
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    # Add proximal term if using FedProx
                    if mu > 0:
                        proximal_term = 0.0
                        global_weights = parameters
                        local_weights = self.get_parameters(config={})
                        
                        for l_w, g_w in zip(local_weights, global_weights):
                            proximal_term += np.linalg.norm(l_w - g_w) ** 2
                        
                        loss += (mu / 2) * proximal_term
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            # Calculate metrics
            metrics = {
                "loss": running_loss / len(train_loader),
                "accuracy": correct / total,
            }
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config={})
        
        # Calculate training time
        train_time = time.time() - start_time
        print(f"Client {self.client_id}: Finished training in {train_time:.2f} seconds")
        print(f"Client {self.client_id}: Training metrics: {metrics}")
        
        return updated_parameters, len(self.train_data[0]), metrics
    
    def evaluate(self, parameters, config):
        """Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters to evaluate.
            config: Configuration from the server.
            
        Returns:
            Evaluation loss, number of samples, and metrics.
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        if self.framework == "tensorflow":
            # Evaluate on TensorFlow
            loss, accuracy = self.model.evaluate(
                self.test_data[0], self.test_data[1], verbose=0
            )
            metrics = {"loss": loss, "accuracy": accuracy}
        else:  # PyTorch
            # Evaluate on PyTorch
            self.model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            # Create DataLoader
            if isinstance(self.test_data, DataLoader):
                test_loader = self.test_data
            else:
                # Assuming test_data is a tuple of (inputs, labels)
                test_dataset = torch.utils.data.TensorDataset(
                    torch.Tensor(self.test_data[0]), 
                    torch.LongTensor(self.test_data[1])
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=self.batch_size, shuffle=False
                )
            
            # Evaluation loop
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            # Calculate metrics
            metrics = {
                "loss": test_loss / len(test_loader),
                "accuracy": correct / total,
            }
        
        return metrics["loss"], len(self.test_data[0]), metrics
        
def main():
    """Main function to start a federated learning client."""
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--server_address", type=str, default="localhost:8080",
                       help="Server address in the format host:port")
    parser.add_argument("--client_id", type=str, required=True,
                       help="Unique identifier for this client")
    parser.add_argument("--framework", type=str, default="tensorflow",
                       choices=["tensorflow", "pytorch"],
                       help="Framework to use (tensorflow or pytorch)")
    parser.add_argument("--dataset", type=str, default="mnist",
                       help="Dataset to use for training")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--local_epochs", type=int, default=5,
                       help="Number of local epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                       help="Learning rate for optimization")
    
    args = parser.parse_args()
    
    # Load dataset
    train_data, test_data = load_dataset(args.dataset, client_id=args.client_id)
    
    # Load model based on framework
    if args.framework.lower() == "tensorflow":
        from client.models.tf_models import create_model
        model = create_model(args.dataset)
    else:  # pytorch
        from client.models.torch_models import create_model
        model = create_model(args.dataset)
    
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
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main() 