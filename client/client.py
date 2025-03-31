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
from .data_loader import load_dataset
from .models.tf_models import create_mnist_model

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
            parameters: Model parameters from the server.
            config: Configuration from the server.
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics).
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Train the model
        if self.framework == "tensorflow":
            history = self.model.fit(
                self.train_data[0],
                self.train_data[1],
                epochs=self.local_epochs,
                batch_size=self.batch_size,
                validation_data=self.test_data,
                verbose=0
            )
            
            # Get metrics
            metrics = {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
                "val_accuracy": float(history.history["val_accuracy"][-1])
            }
        else:  # PyTorch
            self.model.train()
            for epoch in range(self.local_epochs):
                for batch_idx, (data, target) in enumerate(self.train_data):
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
            
            # Evaluate
            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_data:
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.argmax(dim=1)).sum().item()
            
            test_loss /= len(self.test_data)
            accuracy = 100. * correct / len(self.test_data.dataset)
            
            metrics = {
                "loss": test_loss,
                "accuracy": accuracy
            }
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config)
        
        return updated_parameters, len(self.train_data[0]), metrics
    
    def evaluate(self, parameters, config):
        """Evaluate the model on local data.
        
        Args:
            parameters: Model parameters from the server.
            config: Configuration from the server.
            
        Returns:
            Tuple of (loss, num_examples, metrics).
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        if self.framework == "tensorflow":
            loss, accuracy = self.model.evaluate(
                self.test_data[0],
                self.test_data[1],
                batch_size=self.batch_size,
                verbose=0
            )
            metrics = {"accuracy": float(accuracy)}
        else:  # PyTorch
            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_data:
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.argmax(dim=1)).sum().item()
            
            test_loss /= len(self.test_data)
            accuracy = 100. * correct / len(self.test_data.dataset)
            
            loss = test_loss
            metrics = {"accuracy": accuracy}
        
        return loss, len(self.test_data[0]), metrics
        
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
        from .models.tf_models import create_model
        model = create_model(args.dataset)
    else:  # pytorch
        from .models.torch_models import create_model
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