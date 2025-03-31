#!/usr/bin/env python3
"""
Federated Learning - PyTorch Models

This module defines PyTorch models for different datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable

def create_model(dataset_name: str) -> nn.Module:
    """Create a PyTorch model for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', etc.).
        
    Returns:
        A PyTorch model.
    """
    if dataset_name.lower() == "mnist":
        return MNISTModel()
    elif dataset_name.lower() == "cifar10":
        return CIFAR10Model()
    elif dataset_name.lower() == "sentiment":
        return SentimentModel()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

class MNISTModel(nn.Module):
    """PyTorch model for the MNIST dataset."""
    
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Make sure input has the right shape
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR10Model(nn.Module):
    """PyTorch model for the CIFAR-10 dataset."""
    
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # Make sure input has the right shape
        if len(x.shape) == 3 and x.shape[0] == 3:  # If the channel dimension is first
            x = x.permute(1, 0, 2).unsqueeze(0)  # (C, H, W) -> (1, H, C, W)
        elif len(x.shape) == 3:  # If channel dimension is last
            x = x.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class SentimentModel(nn.Module):
    """PyTorch model for sentiment analysis."""
    
    def __init__(self, input_dim: int = 100):
        super(SentimentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

class CustomModel(nn.Module):
    """Custom PyTorch model with configurable architecture."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        hidden_layers: List[int] = [128, 64],
    ):
        super(CustomModel, self).__init__()
        
        # Calculate flattened input size
        input_size = 1
        for dim in input_shape:
            input_size *= dim
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for units in hidden_layers:
            layers.append(nn.Linear(prev_size, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = units
        
        # Add output layer
        if num_classes == 1 or num_classes == 2:
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.model(x) 