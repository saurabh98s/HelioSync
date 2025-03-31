"""
Client package for federated learning.
"""

from .models.tf_models import create_mnist_model
from .data_loader import load_dataset

__all__ = ['create_mnist_model', 'load_dataset'] 