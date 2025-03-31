#!/usr/bin/env python3
"""
Federated Learning - Data Loader

This module handles loading and partitioning datasets for federated learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import os
import json
import random

# TensorFlow and PyTorch imports
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from tensorflow.keras.datasets import mnist

def load_dataset():
    """Load MNIST dataset.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    return x_train, y_train, x_test, y_test

def load_cifar10(
    client_idx: int,
    data_dir: str,
    num_clients: int,
    iid: bool,
    alpha: float,
):
    """Load and partition the CIFAR-10 dataset for federated learning.
    
    Args:
        client_idx: Index of the client.
        data_dir: Directory to store/load the data.
        num_clients: Total number of clients.
        iid: Whether to partition data in an IID manner.
        alpha: Parameter for Dirichlet distribution.
        
    Returns:
        A tuple of (train_data, test_data) for the specified client.
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize data
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Partition data
    if iid:
        # IID partitioning
        num_samples_per_client = len(x_train) // num_clients
        client_train_data = (
            x_train[client_idx * num_samples_per_client:(client_idx + 1) * num_samples_per_client],
            y_train[client_idx * num_samples_per_client:(client_idx + 1) * num_samples_per_client],
        )
    else:
        # Non-IID partitioning using Dirichlet distribution
        client_train_data = create_non_iid_partition(
            x_train, y_train, client_idx, num_clients, alpha
        )
    
    # Use a subset of test data for each client
    num_test_samples_per_client = len(x_test) // num_clients
    client_test_data = (
        x_test[client_idx * num_test_samples_per_client:(client_idx + 1) * num_test_samples_per_client],
        y_test[client_idx * num_test_samples_per_client:(client_idx + 1) * num_test_samples_per_client],
    )
    
    return client_train_data, client_test_data

def load_sentiment(
    client_idx: int,
    data_dir: str,
    num_clients: int,
    iid: bool,
):
    """Load and partition a sentiment analysis dataset for federated learning.
    
    This is a placeholder for a text classification dataset. In a real implementation,
    you would load and preprocess an actual text dataset like IMDB.
    
    Args:
        client_idx: Index of the client.
        data_dir: Directory to store/load the data.
        num_clients: Total number of clients.
        iid: Whether to partition data in an IID manner.
        
    Returns:
        A tuple of (train_data, test_data) for the specified client.
    """
    # This is a placeholder implementation
    # In a real implementation, you would load an actual text dataset
    
    # Simulate embeddings and labels
    num_samples = 1000
    embedding_dim = 100
    
    # Generate random embeddings and binary labels
    x_data = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    y_data = np.random.randint(0, 2, size=num_samples).astype(np.int32)
    
    # Split into train and test
    train_split = int(0.8 * num_samples)
    x_train, x_test = x_data[:train_split], x_data[train_split:]
    y_train, y_test = y_data[:train_split], y_data[train_split:]
    
    # Partition training data
    num_train_samples_per_client = len(x_train) // num_clients
    client_train_data = (
        x_train[client_idx * num_train_samples_per_client:(client_idx + 1) * num_train_samples_per_client],
        y_train[client_idx * num_train_samples_per_client:(client_idx + 1) * num_train_samples_per_client],
    )
    
    # Partition test data
    num_test_samples_per_client = len(x_test) // num_clients
    client_test_data = (
        x_test[client_idx * num_test_samples_per_client:(client_idx + 1) * num_test_samples_per_client],
        y_test[client_idx * num_test_samples_per_client:(client_idx + 1) * num_test_samples_per_client],
    )
    
    return client_train_data, client_test_data

def create_non_iid_partition(
    x_data: np.ndarray,
    y_data: np.ndarray,
    client_idx: int,
    num_clients: int,
    alpha: float,
):
    """Create a non-IID partition using Dirichlet distribution.
    
    Args:
        x_data: Input data.
        y_data: Labels.
        client_idx: Index of the client.
        num_clients: Total number of clients.
        alpha: Parameter for Dirichlet distribution.
        
    Returns:
        A tuple of (x_subset, y_subset) for the specified client.
    """
    # Get number of classes
    num_classes = len(np.unique(y_data))
    
    # Create distribution for each client
    label_distribution = np.random.dirichlet(alpha * np.ones(num_classes), num_clients)
    
    # Get indices for each class
    class_indices = [np.where(y_data == i)[0] for i in range(num_classes)]
    
    # Determine number of samples for this client
    num_samples_per_client = len(x_data) // num_clients
    
    # Create partition for this client
    client_indices = []
    client_dist = label_distribution[client_idx]
    
    # For each class, select samples based on the distribution
    for class_idx, class_prop in enumerate(client_dist):
        num_samples_class = int(class_prop * num_samples_per_client)
        # Randomly select indices for this class
        if len(class_indices[class_idx]) >= num_samples_class:
            selected_indices = np.random.choice(
                class_indices[class_idx], num_samples_class, replace=False
            )
            client_indices.extend(selected_indices)
    
    # If we didn't get enough samples, add more from random classes
    while len(client_indices) < num_samples_per_client:
        class_idx = np.random.randint(0, num_classes)
        if len(class_indices[class_idx]) > 0:
            idx = np.random.choice(class_indices[class_idx], 1, replace=False)[0]
            client_indices.append(idx)
            # Remove the selected index to avoid duplicates
            class_indices[class_idx] = np.setdiff1d(class_indices[class_idx], [idx])
    
    # Return the partition for this client
    return x_data[client_indices], y_data[client_indices] 