#!/usr/bin/env python3
"""
Federated Learning - Server Utilities

This module contains utility functions for the federated learning server.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import tensorflow as tf
import torch

def save_model(model: Any, path: str, framework: str = "tensorflow"):
    """Save a machine learning model to disk.
    
    Args:
        model: The model to save.
        path: Path where the model should be saved.
        framework: The framework of the model ('tensorflow' or 'pytorch').
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if framework.lower() == "tensorflow":
        model.save(path)
    elif framework.lower() == "pytorch":
        torch.save(model.state_dict(), path)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    print(f"Model saved to {path}")

def load_model(path: str, model_class: Any = None, framework: str = "tensorflow"):
    """Load a machine learning model from disk.
    
    Args:
        path: Path where the model is saved.
        model_class: For PyTorch, the class of the model to load.
        framework: The framework of the model ('tensorflow' or 'pytorch').
        
    Returns:
        The loaded model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    
    if framework.lower() == "tensorflow":
        model = tf.keras.models.load_model(path)
    elif framework.lower() == "pytorch":
        if model_class is None:
            raise ValueError("model_class must be provided for PyTorch models")
        model = model_class()
        model.load_state_dict(torch.load(path))
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    print(f"Model loaded from {path}")
    return model

def parameters_to_weights(parameters):
    """Convert serialized model parameters to numpy arrays.
    
    Args:
        parameters: Serialized model parameters.
        
    Returns:
        List of numpy arrays representing model weights.
    """
    if parameters is None:
        return None
    
    weights = []
    shapes = parameters.tensor_type
    
    for tensor_bytes in parameters.tensors:
        weight = np.frombuffer(tensor_bytes, dtype=np.float32)
        weights.append(weight)
    
    return weights

def weights_to_parameters(weights):
    """Convert numpy arrays to serialized model parameters.
    
    Args:
        weights: List of numpy arrays representing model weights.
        
    Returns:
        Serialized model parameters.
    """
    import flwr as fl
    
    tensors = [w.tobytes() for w in weights]
    tensor_type = [fl.common.ndarray_to_bytes(w) for w in weights]
    
    return fl.common.Parameters(tensors=tensors, tensor_type=tensor_type)

def log_metrics(metrics: Dict[str, float], round_num: int, prefix: str = ""):
    """Log metrics to console.
    
    Args:
        metrics: Dictionary of metrics to log.
        round_num: Current round number.
        prefix: Prefix for the metrics (e.g., 'train' or 'val').
    """
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(f"Round {round_num} {prefix} metrics: {metrics_str}") 