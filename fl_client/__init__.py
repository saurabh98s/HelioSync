"""
Federated Learning Client Package

This package provides client-side functionality for federated learning.
"""

from .client import FederatedClient
from .run_client import main as run_client_main

__all__ = ['FederatedClient', 'run_client_main']

# Make run_client available as a module
from . import run_client 