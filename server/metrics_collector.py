"""
Metrics Collector for Federated Learning

This module provides functionality to collect and track metrics during federated learning.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

class MetricsCollector:
    """Collects and manages metrics during federated learning."""
    
    def __init__(self, save_dir: str = "metrics"):
        """Initialize the metrics collector.
        
        Args:
            save_dir: Directory to save metrics and plots.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.round_metrics = []
        self.client_metrics = {}
        self.global_metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'client_participation': []
        }
    
    def add_round_metrics(self, round_num: int, metrics: Dict[str, Any], 
                         num_clients: int, total_clients: int):
        """Add metrics for a training round.
        
        Args:
            round_num: The current round number.
            metrics: Dictionary of metrics from the round.
            num_clients: Number of clients that participated.
            total_clients: Total number of available clients.
        """
        # Store round metrics
        round_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'num_clients': num_clients,
            'total_clients': total_clients,
            'participation_rate': num_clients / total_clients
        }
        self.round_metrics.append(round_data)
        
        # Update global metrics
        self.global_metrics['train_loss'].append(metrics.get('train_loss', 0))
        self.global_metrics['train_accuracy'].append(metrics.get('train_accuracy', 0))
        self.global_metrics['test_loss'].append(metrics.get('test_loss', 0))
        self.global_metrics['test_accuracy'].append(metrics.get('test_accuracy', 0))
        self.global_metrics['client_participation'].append(round_data['participation_rate'])
        
        # Save metrics after each round
        self.save_metrics()
    
    def add_client_metrics(self, client_id: str, round_num: int, metrics: Dict[str, Any]):
        """Add metrics for a specific client.
        
        Args:
            client_id: The client's identifier.
            round_num: The current round number.
            metrics: Dictionary of metrics from the client.
        """
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []
        
        self.client_metrics[client_id].append({
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
    
    def save_metrics(self):
        """Save all metrics to disk."""
        metrics_data = {
            'rounds': self.round_metrics,
            'clients': self.client_metrics,
            'global': self.global_metrics
        }
        
        # Save JSON file
        metrics_file = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Generate and save plots
        self._generate_plots()
    
    def _generate_plots(self):
        """Generate plots of training metrics."""
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        rounds = list(range(1, len(self.round_metrics) + 1))
        
        # Plot training metrics
        ax1.plot(rounds, self.global_metrics['train_loss'], 'b-', label='Training Loss')
        ax1.plot(rounds, self.global_metrics['test_loss'], 'r--', label='Test Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(rounds, self.global_metrics['train_accuracy'], 'b-', label='Training Accuracy')
        ax2.plot(rounds, self.global_metrics['test_accuracy'], 'r--', label='Test Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot client participation
        ax3.plot(rounds, self.global_metrics['client_participation'], 'g-')
        ax3.set_title('Client Participation Rate')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Participation Rate')
        ax3.grid(True)
        
        # Plot client performance distribution (box plot)
        if self.client_metrics:
            client_accuracies = []
            for round_data in self.round_metrics:
                round_accuracies = []
                for client_data in self.client_metrics.values():
                    for data in client_data:
                        if data['round'] == round_data['round']:
                            acc = data['metrics'].get('accuracy', 0)
                            round_accuracies.append(acc)
                if round_accuracies:
                    client_accuracies.append(round_accuracies)
            
            if client_accuracies:
                ax4.boxplot(client_accuracies, labels=rounds)
                ax4.set_title('Client Accuracy Distribution')
                ax4.set_xlabel('Round')
                ax4.set_ylabel('Accuracy')
                ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
        plt.close()
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics.
        
        Returns:
            Dictionary containing the latest metrics.
        """
        if not self.round_metrics:
            return {}
        
        latest = self.round_metrics[-1]
        return {
            'round': latest['round'],
            'train_loss': latest['metrics'].get('train_loss', 0),
            'train_accuracy': latest['metrics'].get('train_accuracy', 0),
            'test_loss': latest['metrics'].get('test_loss', 0),
            'test_accuracy': latest['metrics'].get('test_accuracy', 0),
            'num_clients': latest['num_clients'],
            'total_clients': latest['total_clients'],
            'participation_rate': latest['participation_rate']
        } 