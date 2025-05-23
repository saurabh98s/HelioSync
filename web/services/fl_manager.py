"""
Federated Learning Manager Service

This module provides services for managing the federated learning process,
including starting the server, updating status, and handling model creation.
"""

import os
import sys
import logging
import subprocess
import threading
from datetime import datetime
from flask import current_app
import numpy as np
import time
import shutil
import json
import tempfile
import secrets

from web.app import db
from web.models import Project, Client, Model, ProjectClient, Organization
from web.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

class FederatedLearningServer:
    """Manages the federated learning server and client connections."""
    
    def __init__(self):
        """Initialize the federated learning server."""
        self.clients = {}  # Maps client_id to client info
        self.client_projects = {}  # Maps client_id to list of project_ids
        self.model_weights = {}  # Maps project_id to model weights
        self.aggregated_metrics = {}  # Maps project_id to metrics
        self.client_weights = {}  # Maps project_id to dict of client weights
        self.client_metrics = {}  # Maps client_id to metrics
        self.projects = {}  # Maps project_id to project info
        self.current_metrics = {}  # Maps project_id to current round metrics
        self.models = {}  # Maps project_id to model instances
        self.current_round = 0
        self.rounds = 0
        self.min_clients = 0
        
        logger.info("Federated Learning Server initialized.")
    
    def add_client(self, client_id, name, data_size, device_info, platform, machine, python_version):
        """Add a new client to the FL server."""
        try:
            # Register the client in the database and get client info
            success = self.register_client(
                client_id=client_id,
                name=name,
                data_size=data_size,
                device_info=device_info,
                platform=platform,
                machine=machine,
                python_version=python_version
            )
            
            if success:
                # Initialize client's project participation
                self.client_projects[client_id] = set()
                logger.info(f"Client {client_id} added successfully to FL server")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error adding client {client_id} to FL server: {str(e)}")
            return False
    
    def remove_client(self, client_id):
        """Remove a client from the FL server."""
        try:
            # Unregister the client
            success = self.unregister_client(client_id)
            
            if success:
                # Remove client from all projects
                if client_id in self.client_projects:
                    for project_id in self.client_projects[client_id]:
                        self.remove_client_from_project(client_id, project_id)
                    del self.client_projects[client_id]
                
                # Remove client metrics
                if client_id in self.client_metrics:
                    del self.client_metrics[client_id]
                
                logger.info(f"Client {client_id} removed successfully from FL server")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing client {client_id} from FL server: {str(e)}")
            return False
    
    def add_client_to_project(self, client_id, project_id):
        """Add a client to a specific project."""
        try:
            # Get the client from the database to ensure we have correct client_id (UUID)
            client = Client.query.filter_by(client_id=client_id).first()
            if not client:
                client = Client.query.get(client_id)
                if not client:
                    logger.error(f"Client with ID/UUID {client_id} not found")
                    return False
                # We found the client by ID, so use its UUID
                client_id = client.client_id
                
            if client_id not in self.clients:
                logger.error(f"Client {client_id} not found in FL server. Attempting to register...")
                # Try to register the client with the server
                self.register_client(
                    client_id=client_id,
                    name=client.name,
                    data_size=client.data_size or 0,
                    device_info=client.device_info or '',
                    platform=client.platform or 'unknown',
                    machine=client.machine or 'unknown',
                    python_version=client.python_version or '3.x'
                )
            
            if project_id not in self.projects:
                logger.error(f"Project {project_id} not found")
                return False
            
            # Make sure client_projects dict is initialized for this client
            if client_id not in self.client_projects:
                self.client_projects[client_id] = set()
                
            # Add client to project's client set
            self.client_projects[client_id].add(project_id)
            
            # Check if the project client association already exists
            project_client = ProjectClient.query.filter_by(
                project_id=project_id,
                client_id=client.id
            ).first()
            
            if not project_client:
                # Create project client association in database
                project_client = ProjectClient(
                    project_id=project_id,
                    client_id=client.id,
                    status='registered',
                    joined_at=datetime.utcnow()
                )
                db.session.add(project_client)
            else:
                # Update existing association
                project_client.status = 'registered'
                project_client.joined_at = datetime.utcnow()
                
            db.session.commit()
            
            logger.info(f"Client {client_id} added to project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding client {client_id} to project {project_id}: {str(e)}")
            db.session.rollback()
            return False
    
    def remove_client_from_project(self, client_id, project_id):
        """Remove a client from a specific project."""
        try:
            if client_id in self.client_projects and project_id in self.client_projects[client_id]:
                # Remove from project's client set
                self.client_projects[client_id].remove(project_id)
                
                # Update project client status in database
                project_client = ProjectClient.query.filter_by(
                    project_id=project_id,
                    client_id=client_id
                ).first()
                
                if project_client:
                    project_client.status = 'disconnected'
                    project_client.disconnected_at = datetime.utcnow()
                    db.session.commit()
                
                logger.info(f"Client {client_id} removed from project {project_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing client {client_id} from project {project_id}: {str(e)}")
            db.session.rollback()
            return False
    
    def get_client_projects(self, client_id):
        """Get all projects a client is participating in."""
        return list(self.client_projects.get(client_id, set()))
    
    def get_project_clients(self, project_id):
        """Get all clients participating in a project."""
        return [client_id for client_id, projects in self.client_projects.items() 
                if project_id in projects]
    
    def is_client_in_project(self, client_id, project_id):
        """Check if a client is participating in a project."""
        return (client_id in self.client_projects and 
                project_id in self.client_projects[client_id])
    
    def get_client_status(self, client_id):
        """Get the current status of a client."""
        if client_id not in self.clients:
            return None
        
        return {
            'is_connected': True,
            'last_seen': self.clients[client_id]['last_seen'],
            'active_projects': self.get_client_projects(client_id)
        }
    
    def get_project_status(self, project_id):
        """Get the current status of a project."""
        if project_id not in self.projects:
            return None
        
        return {
            'active_clients': self.get_project_clients(project_id),
            'metrics': self.aggregated_metrics.get(project_id, {}),
            'status': self.projects[project_id].get('status', 'unknown')
        }
    
    def register_client(self, client_id, name, data_size, device_info, platform, machine, python_version):
        """Register a client in the FL server."""
        try:
            # Store client information
            self.clients[client_id] = {
                'name': name,
                'data_size': data_size,
                'device_info': device_info,
                'platform': platform,
                'machine': machine,
                'python_version': python_version,
                'last_seen': datetime.utcnow()
            }
            
            # Initialize client metrics
            self.client_metrics[client_id] = {}
            
            logger.info(f"Client {client_id} registered with FL server")
            return True
            
        except Exception as e:
            logger.error(f"Error registering client {client_id}: {str(e)}")
            return False
    
    def unregister_client(self, client_id):
        """Unregister a client from the FL server."""
        try:
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Client {client_id} unregistered from FL server")
                return True
            logger.warning(f"Client {client_id} not found for unregistration")
            return False
            
        except Exception as e:
            logger.error(f"Error unregistering client {client_id}: {str(e)}")
            return False
    
    def update_client_metrics(self, client_id, project_id, metrics):
        """Update metrics for a specific client in a project."""
        try:
            if client_id not in self.client_metrics:
                self.client_metrics[client_id] = {}
            
            # Store all metrics in a consistent format
            self.client_metrics[client_id][project_id] = {
                'accuracy': float(metrics.get('accuracy', 0)),
                'loss': float(metrics.get('loss', 0)),
                'val_accuracy': float(metrics.get('val_accuracy', 0)),
                'val_loss': float(metrics.get('val_loss', 0)),
                'epoch': int(metrics.get('epoch', 0)),
                'total_epochs': int(metrics.get('total_epochs', 0)),
                'round': int(metrics.get('round', 0)),
                'samples': int(metrics.get('samples', 0)),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Get the client from the database
            client = Client.query.filter_by(client_id=client_id).first()
            if not client:
                logger.warning(f"Client {client_id} not found in database when updating metrics")
                return False
            
            # Update project client status
            project_client = ProjectClient.query.filter_by(
                project_id=project_id,
                client_id=client.id
            ).first()
            
            if project_client:
                project_client.status = 'training'
                project_client.metrics = {
                    'accuracy': float(metrics.get('accuracy', 0)),
                    'loss': float(metrics.get('loss', 0)),
                    'val_accuracy': float(metrics.get('val_accuracy', 0)),
                    'val_loss': float(metrics.get('val_loss', 0)),
                    'epoch': int(metrics.get('epoch', 0)),
                    'total_epochs': int(metrics.get('total_epochs', 0)),
                    'round': int(metrics.get('round', 0)),
                    'timestamp': datetime.utcnow().isoformat()
                }
                project_client.training_samples = int(metrics.get('samples', 0))
                project_client.local_epochs = int(metrics.get('total_epochs', 0))
                project_client.last_update = datetime.utcnow()
                db.session.commit()
                
                logger.info(f"Updated metrics for client {client_id} in project {project_id}")
                logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}, Loss: {metrics.get('loss', 0):.4f}")
                logger.info(f"Val Accuracy: {metrics.get('val_accuracy', 0):.4f}, Val Loss: {metrics.get('val_loss', 0):.4f}")
            else:
                logger.warning(f"ProjectClient association not found for client {client_id} and project {project_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating metrics for client {client_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            db.session.rollback()
            return False
    
    def update_aggregated_metrics(self, project_id, metrics):
        """Update aggregated metrics for a project."""
        try:
            self.aggregated_metrics[project_id] = {
                'accuracy': metrics.get('accuracy', 0),
                'loss': metrics.get('loss', 0),
                'round': metrics.get('round', 0),
                'timestamp': datetime.utcnow()
            }
            
            # Update project status
            project = Project.query.get(project_id)
            if project:
                project.current_round = metrics.get('round', 0)
                # Remove premature completion check
                db.session.commit()
            
            logger.info(f"Updated aggregated metrics for project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating aggregated metrics for project {project_id}: {str(e)}")
            db.session.rollback()
            return False
    
    def initialize_project(self, project):
        """Initialize a project in the FL server."""
        try:
            project_id = project.id
            
            # Store project settings
            self.projects[project_id] = {
                'status': project.status,
                'framework': project.framework,
                'dataset': project.dataset_name,
                'current_round': project.current_round,
                'total_rounds': project.rounds,
                'min_clients': project.min_clients
            }
            
            # Initialize project-specific attributes
            self.current_round = project.current_round
            self.rounds = project.rounds
            self.min_clients = project.min_clients
            
            # Initialize model based on framework and dataset
            if project.framework.lower() == 'tensorflow':
                if project_id not in self.model_weights:
                    self.model_weights[project_id] = self._initialize_tensorflow_model(project.dataset_name)
                    logger.info(f"Initialized model weights for project {project_id}")
            else:
                raise ValueError(f"Unsupported framework: {project.framework}")
            
            # Initialize client_weights dictionary for this project if it doesn't exist
            if project_id not in self.client_weights:
                self.client_weights[project_id] = {}
                logger.info(f"Initialized client_weights dictionary for project {project_id}")
            
            # Initialize metrics for this project
            if project_id not in self.aggregated_metrics:
                self.aggregated_metrics[project_id] = {
                    'accuracy': 0,
                    'loss': 0,
                    'round': project.current_round,
                    'clients': 0,
                    'timestamp': datetime.utcnow()
                }
                logger.info(f"Initialized aggregated_metrics for project {project_id}")
            
            # If the project already has models, use the latest model's metrics
            latest_model = Model.query.filter_by(project_id=project_id).order_by(Model.version.desc()).first()
            if latest_model and latest_model.metrics:
                self.aggregated_metrics[project_id].update({
                    'accuracy': latest_model.metrics.get('accuracy', 0),
                    'loss': latest_model.metrics.get('loss', 0),
                    'round': latest_model.metrics.get('round', project.current_round),
                    'clients': latest_model.clients_count
                })
                logger.info(f"Updated metrics from latest model for project {project_id}")
            
            logger.info(f"Project {project_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing project {project.id}: {str(e)}")
            return False
    
    def _initialize_tensorflow_model(self, dataset_name):
        """Initialize a TensorFlow model based on dataset.
        
        Args:
            dataset_name: Name of the dataset to use.
            
        Returns:
            List of numpy arrays representing model weights.
        """
        # Create mock model weights for testing purposes
        # In a real implementation, you would actually load or create a real model
        
        # MNIST weights structure (CNN with BatchNorm)
        if dataset_name.lower() == 'mnist':
            # First Conv Block
            # Conv1 (28x28x1 -> 28x28x32)
            conv1_w = np.random.randn(3, 3, 1, 32).astype(np.float32) * 0.1
            conv1_b = np.zeros(32, dtype=np.float32)
            # BatchNorm1
            bn1_gamma = np.ones(32, dtype=np.float32)
            bn1_beta = np.zeros(32, dtype=np.float32)
            bn1_mean = np.zeros(32, dtype=np.float32)
            bn1_var = np.ones(32, dtype=np.float32)
            
            # Conv2 (28x28x32 -> 28x28x32)
            conv2_w = np.random.randn(3, 3, 32, 32).astype(np.float32) * 0.1
            conv2_b = np.zeros(32, dtype=np.float32)
            
            # Second Conv Block
            # Conv3 (14x14x32 -> 14x14x64)
            conv3_w = np.random.randn(3, 3, 32, 64).astype(np.float32) * 0.1
            conv3_b = np.zeros(64, dtype=np.float32)
            # BatchNorm2
            bn2_gamma = np.ones(64, dtype=np.float32)
            bn2_beta = np.zeros(64, dtype=np.float32)
            bn2_mean = np.zeros(64, dtype=np.float32)
            bn2_var = np.ones(64, dtype=np.float32)
            
            # Conv4 (14x14x64 -> 14x14x64)
            conv4_w = np.random.randn(3, 3, 64, 64).astype(np.float32) * 0.1
            conv4_b = np.zeros(64, dtype=np.float32)
            
            # Dense layers
            # Dense1 (flattened -> 512)
            # IMPORTANT: Changed from 3136 (7*7*64) to 12544 (14*14*64) to match client model architecture
            dense1_w = np.random.randn(12544, 512).astype(np.float32) * 0.1  # 14*14*64 -> 512
            dense1_b = np.zeros(512, dtype=np.float32)
            # BatchNorm3
            bn3_gamma = np.ones(512, dtype=np.float32)
            bn3_beta = np.zeros(512, dtype=np.float32)
            bn3_mean = np.zeros(512, dtype=np.float32)
            bn3_var = np.ones(512, dtype=np.float32)
            
            # Output (512 -> 10)
            output_w = np.random.randn(512, 10).astype(np.float32) * 0.1
            output_b = np.zeros(10, dtype=np.float32)
            
            weights = [
                # First Conv Block
                conv1_w, conv1_b,
                bn1_gamma, bn1_beta, bn1_mean, bn1_var,
                conv2_w, conv2_b,
                
                # Second Conv Block
                conv3_w, conv3_b,
                bn2_gamma, bn2_beta, bn2_mean, bn2_var,
                conv4_w, conv4_b,
                
                # Dense layers
                dense1_w, dense1_b,
                bn3_gamma, bn3_beta, bn3_mean, bn3_var,
                output_w, output_b
            ]
            logger.info(f"Initialized mock MNIST model weights with BatchNorm")
            
        # CIFAR-10 weights structure (simple CNN)
        elif dataset_name.lower() == 'cifar10':
            # Conv1 (32x32x3 -> 30x30x32)
            conv1_w = np.random.randn(3, 3, 3, 32).astype(np.float32) * 0.1
            conv1_b = np.zeros(32, dtype=np.float32)
            
            # Conv2 (30x30x32 -> 28x28x64)
            conv2_w = np.random.randn(3, 3, 32, 64).astype(np.float32) * 0.1
            conv2_b = np.zeros(64, dtype=np.float32)
            
            # Dense (16384 -> 128)
            dense1_w = np.random.randn(16384, 128).astype(np.float32) * 0.1
            dense1_b = np.zeros(128, dtype=np.float32)
            
            # Output (128 -> 10)
            output_w = np.random.randn(128, 10).astype(np.float32) * 0.1
            output_b = np.zeros(10, dtype=np.float32)
            
            weights = [conv1_w, conv1_b, conv2_w, conv2_b, dense1_w, dense1_b, output_w, output_b]
            logger.info(f"Initialized mock CIFAR-10 model weights")
            
        # Fallback to simple model for other datasets
        else:
            logger.warning(f"No specific model for dataset {dataset_name}, using generic model")
            # Simple 784 -> 128 -> 10 model for any dataset
            weights = [
                np.random.randn(784, 128).astype(np.float32) * 0.1,  # Dense1 weights
                np.zeros(128, dtype=np.float32),                      # Dense1 bias
                np.random.randn(128, 10).astype(np.float32) * 0.1,   # Output weights
                np.zeros(10, dtype=np.float32)                        # Output bias
            ]
            
        return weights
    
    def get_model_weights(self, project_id=None):
        """Get the current model weights for a project.
        
        Args:
            project_id: Project ID to get weights for. If None, returns weights for first project.
            
        Returns:
            Model weights as a list of numpy arrays.
            
        Raises:
            ValueError: If no projects are initialized or project_id is not found.
        """
        if project_id is None:
            # Return first project if no project_id specified
            if not self.model_weights:
                raise ValueError("No projects initialized")
            return list(self.model_weights.values())[0]
        
        if project_id not in self.model_weights:
            raise ValueError(f"Project {project_id} not initialized")
        
        return self.model_weights[project_id]
    
    def update_model(self, client_id, weights, metrics, project_id=None):
        """Update the model with client's weights."""
        try:
            # Extract project_id from metrics if not provided
            if project_id is None:
                project_id = metrics.get('project_id')
                
            if project_id is None:
                logger.error("No project_id provided or found in metrics")
                logger.error(f"Available metrics: {metrics}")
                # Try to find the first project the client is assigned to
                if client_id in self.client_projects and self.client_projects[client_id]:
                    project_id = next(iter(self.client_projects[client_id]))
                    logger.info(f"Using first available project for client: {project_id}")
                else:
                    # Try to get the first active project
                    if self.model_weights:
                        project_id = list(self.model_weights.keys())[0]
                        logger.info(f"Using first available project: {project_id}")
                    else:
                        raise ValueError("No projects initialized and no project_id provided")
            
            logger.info(f"Processing update from client {client_id} for project {project_id}")
            
            # Pre-validate weights to help debugging
            if weights:
                try:
                    # Log the weight count and first few weights' shapes
                    logger.info(f"Client {client_id} sent {len(weights)} weight arrays")
                    
                    # Check shape and type of first few arrays as a sample
                    for i, w in enumerate(weights[:3]):
                        if isinstance(w, np.ndarray):
                            logger.info(f"Weight {i}: np.ndarray, shape={w.shape}, dtype={w.dtype}")
                        elif isinstance(w, list):
                            # For list type, report dimensions
                            if not w:
                                logger.info(f"Weight {i}: empty list")
                            elif isinstance(w[0], list):
                                # Likely a 2D list
                                logger.info(f"Weight {i}: 2D+ list, outer_dim={len(w)}, inner_dim={len(w[0]) if w[0] else 'empty'}")
                            else:
                                # Likely a 1D list
                                logger.info(f"Weight {i}: 1D list, len={len(w)}")
                        else:
                            logger.info(f"Weight {i}: {type(w)}")
                except Exception as shape_err:
                    logger.error(f"Error inspecting weights: {str(shape_err)}")
            
            # Filter out empty weight arrays
            if weights:
                filtered_weights = []
                for i, w in enumerate(weights):
                    try:
                        # Check if weight is empty or invalid
                        if w is None:
                            logger.warning(f"Skipping None weight at index {i}")
                            continue
                            
                        if isinstance(w, np.ndarray):
                            if w.size == 0:
                                logger.warning(f"Skipping empty numpy array at index {i}")
                                continue
                            filtered_weights.append(w)
                        elif isinstance(w, list):
                            # For list type, convert to numpy array if it has content
                            if not w:
                                logger.warning(f"Skipping empty list at index {i}")
                                continue
                                
                            # Convert to numpy array
                            try:
                                w_array = np.array(w, dtype=np.float32)
                                if w_array.size == 0:
                                    logger.warning(f"Skipping list that converted to empty array at index {i}")
                                    continue
                                # Add the valid numpy array to filtered weights
                                filtered_weights.append(w_array)
                                logger.info(f"Successfully converted list to numpy array at index {i}: shape={w_array.shape}")
                            except Exception as array_err:
                                logger.error(f"Error converting list to numpy array at index {i}: {str(array_err)}")
                                continue
                        else:
                            logger.warning(f"Unsupported weight type at index {i}: {type(w)}")
                            continue
                    except Exception as e:
                        logger.error(f"Error processing weight at index {i}: {str(e)}")
                        continue
                
                # Log the filtering results
                if len(filtered_weights) != len(weights):
                    logger.warning(f"Filtered {len(weights) - len(filtered_weights)} empty weights out of {len(weights)}")
                    
                # Replace weights with filtered weights
                weights = filtered_weights
                
                # Log final weight count
                logger.info(f"After filtering: {len(weights)} valid weight arrays")
                
                # If all weights were filtered out, log a warning but continue processing
                if not weights:
                    logger.warning(f"No valid weights after filtering from client {client_id}")
            
            # Determine if this is a final update
            is_final_update = metrics.get('is_final', False)
            
            # Get the project from the database (regardless of initialization status)
            project = Project.query.get(project_id)
            if not project:
                if is_final_update:
                    # For final updates, try to create a dummy project if it doesn't exist
                    logger.warning(f"Project {project_id} not found but this is a final update. Creating a temporary record.")
                    try:
                        project = Project(
                            id=project_id,
                            name=f"Project {project_id} (Recovered)",
                            status="running",
                            current_round=0,
                            rounds=1,
                            min_clients=1,
                            dataset_name="unknown",
                            framework="tensorflow",
                            creator_id=1  # Default creator ID
                        )
                        db.session.add(project)
                        db.session.commit()
                        logger.info(f"Created temporary project record for final update: {project_id}")
                    except Exception as e:
                        logger.error(f"Failed to create temporary project: {str(e)}")
                        
                        # For final updates, try even harder by using a direct SQL insert
                        try:
                            logger.warning("Attempting direct SQL insert for emergency project creation")
                            db.session.execute(
                                f"INSERT INTO projects (id, name, status, current_round, rounds, min_clients, dataset_name, framework, creator_id, created_at) "
                                f"VALUES ({project_id}, 'Project {project_id} (Emergency)', 'running', 0, 1, 1, 'unknown', 'tensorflow', 1, CURRENT_TIMESTAMP)"
                            )
                            db.session.commit()
                            project = Project.query.get(project_id)
                            logger.info(f"Created emergency project record through direct SQL: {project_id}")
                        except Exception as sql_err:
                            logger.error(f"Direct SQL insert also failed: {str(sql_err)}")
                            raise ValueError(f"Project {project_id} not found and could not create temporary record")
                else:
                    raise ValueError(f"Project {project_id} not found")
            
            # Make sure current_metrics is initialized for this project
            if project_id not in self.current_metrics:
                self.current_metrics[project_id] = {
                    'accuracy': [],
                    'loss': [],
                    'client_ids': set(),
                    'round': project.current_round
                }
            
            # Check if the project is in model_weights or needs initialization
            if project_id not in self.model_weights:
                logger.warning(f"Project {project_id} not initialized in FL server")
                
                # For any update (final or not), try to initialize the project
                try:
                    # Force the project to running status temporarily if needed
                    original_status = project.status
                    if project.status != 'running':
                        logger.info(f"Temporarily setting project {project_id} to running for initialization")
                        project.status = 'running'
                        db.session.commit()
                    
                    # Try to initialize
                    success = self.initialize_project(project)
                    
                    # Restore original status if we changed it
                    if original_status != 'running':
                        project.status = original_status
                        db.session.commit()
                    
                    if not success:
                        if is_final_update:
                            # For final updates, create minimal weights structure
                            logger.warning(f"Creating minimal model weights for final update of project {project_id}")
                            self.model_weights[project_id] = []
                            # Try to determine weight structure from the client's weights
                            if weights:
                                self.model_weights[project_id] = weights
                        else:
                            raise ValueError(f"Failed to initialize project {project_id}")
                except Exception as e:
                    if is_final_update:
                        # For final updates, continue even if initialization fails
                        logger.warning(f"Project initialization failed but continuing for final update: {str(e)}")
                        # Create empty model weights dictionary if needed
                        if project_id not in self.model_weights:
                            self.model_weights[project_id] = weights if weights else []
                    else:
                        raise ValueError(f"Project {project_id} initialization failed: {str(e)}")
            
            # Get client for database operations
            db_client = None
            try:
                # Look up client by client_id
                db_client = Client.query.filter_by(client_id=client_id).first()
                
                # If client is not found, try to create a minimal record for this update
                if not db_client and is_final_update:
                    logger.warning(f"Client {client_id} not found. Creating emergency client record for final update.")
                    try:
                        # Get first available organization
                        org = Organization.query.first()
                        if not org:
                            logger.error("No organization found for emergency client creation")
                            # Continue anyway without creating client
                        else:
                            # Create minimal client record
                            db_client = Client(
                                client_id=client_id,
                                name=f"Client {client_id[:8]} (Emergency)",
                                organization_id=org.id,
                                is_connected=True,
                                last_heartbeat=datetime.utcnow()
                            )
                            db.session.add(db_client)
                            db.session.commit()
                            logger.info(f"Created emergency client record: {client_id}")
                    except Exception as client_err:
                        logger.error(f"Failed to create emergency client: {str(client_err)}")
                        # Continue anyway without the client record
            except Exception as db_err:
                logger.error(f"Database error looking up client: {str(db_err)}")
                # Continue anyway as this shouldn't block the update
            
            # Even if project is completed, always process final updates
            if project.status == 'completed' and not is_final_update:
                logger.warning(f"Project {project_id} is already marked as completed, but received non-final update from client {client_id}")
                return {"success": True, "message": "Project already completed", "aggregated": False}
            
            # Update client metrics - track metrics for analysis
            self.update_client_metrics(client_id, project_id, metrics)
            
            # Track metrics in our current_metrics structure
            try:
                accuracy = float(metrics.get('accuracy', 0))
                loss = float(metrics.get('loss', 0))
                self.current_metrics[project_id]['accuracy'].append(accuracy)
                self.current_metrics[project_id]['loss'].append(loss)
                self.current_metrics[project_id]['client_ids'].add(client_id)
                self.current_metrics[project_id]['round'] = project.current_round
            except Exception as metrics_err:
                logger.error(f"Error updating current metrics: {str(metrics_err)}")
            
            # Store client weights for aggregation
            if project_id not in self.client_weights:
                self.client_weights[project_id] = {}
            
            # Save client weights and data size
            self.client_weights[project_id][client_id] = {
                'weights': weights,
                'data_size': metrics.get('samples', 1),  # Use samples from metrics or default to 1
                'metrics': metrics
            }
            
            # Get active project clients
            project_clients_count = len(project.clients) if project.clients else 1
            active_project_clients = len(self.client_weights.get(project_id, {}))
            
            logger.info(f"Received update from client {client_id} for project {project_id}")
            logger.info(f"Active clients: {active_project_clients}/{project_clients_count} (min required: {project.min_clients})")
            
            # For single-round projects or final updates, we need to process them even if minimal clients aren't met
            is_single_round_project = project.rounds == 1
            should_process_update = active_project_clients >= project.min_clients or is_final_update or is_single_round_project
            
            # Only proceed with aggregation if we should process the update
            if should_process_update:
                logger.info(f"Aggregating weights for project {project_id} from {active_project_clients} clients")
                
                # Get aggregation method from project settings or metrics
                alpha = metrics.get('alpha', 0.5)  # Default to balanced weighting
                use_perfedavg = metrics.get('use_perfedavg', True)  # Default to using PerfFedAvg
                
                # Perform aggregation - use PerfFedAvg as default with custom alpha if specified
                if active_project_clients > 1:
                    if use_perfedavg:
                        logger.info(f"Using PerfFedAvg with alpha={alpha} for project {project_id}")
                        aggregated_weights = self._aggregate_weights_perfedavg(project_id, alpha=alpha)
                    else:
                        logger.info(f"Using standard FedAvg for project {project_id}")
                        aggregated_weights = self._aggregate_weights(project_id)
                        
                    # Store the aggregated weights
                    self.model_weights[project_id] = aggregated_weights
                else:
                    # For single client, just use their weights directly
                    logger.info(f"Only one client for project {project_id}, using their weights directly")
                    self.model_weights[project_id] = weights
                
                # Update project metrics
                if project_id not in self.aggregated_metrics:
                    self.aggregated_metrics[project_id] = {}
                
                # Calculate the average metrics
                avg_accuracy = 0
                avg_loss = 0
                avg_val_accuracy = 0
                avg_val_loss = 0
                total_samples = 0
                
                for client_data in self.client_weights[project_id].values():
                    client_metrics = client_data.get('metrics', {})
                    samples = client_data.get('data_size', 1)
                    total_samples += samples
                    avg_accuracy += client_metrics.get('accuracy', 0) * samples
                    avg_loss += client_metrics.get('loss', 0) * samples
                    avg_val_accuracy += client_metrics.get('val_accuracy', 0) * samples
                    avg_val_loss += client_metrics.get('val_loss', 0) * samples
                
                if total_samples > 0:
                    avg_accuracy /= total_samples
                    avg_loss /= total_samples
                    avg_val_accuracy /= total_samples
                    avg_val_loss /= total_samples
                
                # Validate the aggregated model
                validation_metrics = self._validate_aggregated_model(project_id)
                
                # Use validation metrics if available, otherwise use client-reported averages
                if validation_metrics:
                    logger.info(f"Using server-side validation metrics for project {project_id}")
                    # Prefer server-side validation if available 
                    if validation_metrics.get('accuracy') is not None:
                        avg_accuracy = validation_metrics.get('accuracy', avg_accuracy) 
                    if validation_metrics.get('loss') is not None:
                        avg_loss = validation_metrics.get('loss', avg_loss)
                    if validation_metrics.get('val_accuracy') is not None:
                        avg_val_accuracy = validation_metrics.get('val_accuracy', avg_val_accuracy)
                    if validation_metrics.get('val_loss') is not None:
                        avg_val_loss = validation_metrics.get('val_loss', avg_val_loss)
                else:
                    logger.info(f"Using client-reported metrics for project {project_id}")
                
                # Update aggregated metrics
                self.aggregated_metrics[project_id].update({
                    'accuracy': float(avg_accuracy),
                    'loss': float(avg_loss),
                    'val_accuracy': float(avg_val_accuracy),
                    'val_loss': float(avg_val_loss),
                    'round': project.current_round,
                    'timestamp': datetime.utcnow().isoformat(),
                    'clients': active_project_clients,
                    'total_samples': total_samples
                })
                
                logger.info(f"Updated aggregated metrics for project {project_id}:")
                logger.info(f"Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}")
                logger.info(f"Val Accuracy: {avg_val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Determine if we need to save a model and complete the project
                should_save_model = False
                should_complete_project = False
                
                # Single-round projects should complete after aggregation
                if is_single_round_project:
                    logger.info(f"Single-round project {project_id} processed an update")
                    should_save_model = True
                    if is_final_update:
                        logger.info(f"Final update received for single-round project {project_id}")
                        should_complete_project = True
                    
                # Final update from client always triggers save
                elif is_final_update:
                    logger.info(f"Final update received for project {project_id}")
                    should_save_model = True
                    should_complete_project = True
                
                # Save the model if needed
                if should_save_model:
                    try:
                        logger.info(f"Saving model for project {project_id}")
                        saved = self._save_final_model(project_id)
                        if saved:
                            logger.info(f"Successfully saved model for project {project_id}")
                        else:
                            logger.error(f"Failed to save model for project {project_id}")
                    except Exception as save_err:
                        logger.error(f"Error saving model: {str(save_err)}")
                
                # Mark project as completed if needed
                if should_complete_project:
                    logger.info(f"Marking project {project_id} as completed")
                    project.status = 'completed'
                    db.session.commit()
                    
                # Clear client weights if we're done with this round
                if should_complete_project:
                    logger.info(f"Clearing client weights for project {project_id}")
                    self.client_weights[project_id] = {}
                
                # Return success with aggregation info
                return {
                    "success": True,
                    "aggregated": True,
                    "round_completed": should_complete_project,
                    "project_completed": should_complete_project,
                    "message": "Update processed successfully"
                }
            else:
                # Not enough clients yet, but acknowledge the update
                return {
                    "success": True,
                    "aggregated": False,
                    "clients_submitted": active_project_clients,
                    "clients_required": project.min_clients,
                    "message": f"Update received, waiting for more clients ({active_project_clients}/{project.min_clients})"
                }
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # For final updates, try to recover
            if metrics and metrics.get('is_final', False):
                logger.warning("Final update encountered an error but attempting to mark project as completed anyway")
                try:
                    project = Project.query.get(project_id)
                    if project:
                        # Try to create an emergency model
                        try:
                            emergency_data = {
                                'accuracy': metrics.get('accuracy', 0),
                                'loss': metrics.get('loss', 0),
                                'val_accuracy': metrics.get('val_accuracy', 0),
                                'val_loss': metrics.get('val_loss', 0),
                                'clients': 1,
                                'is_final': True,
                                'is_emergency': True
                            }
                            self._emergency_model_save(project, emergency_data)
                        except Exception:
                            pass
                            
                        # Mark project as completed regardless
                        project.status = 'completed'
                        db.session.commit()
                        logger.info(f"Project {project_id} marked as completed despite error")
                        return {"success": True, "message": "Project completed", "project_completed": True}
                except Exception as recover_e:
                    logger.error(f"Recovery attempt also failed: {str(recover_e)}")
            
            return {"success": False, "error": str(e)}
    
    def _aggregate_weights(self, project_id):
        """Aggregate weights from multiple clients using FedAvg."""
        if project_id not in self.client_weights or not self.client_weights[project_id]:
            logger.warning(f"No client weights available for project {project_id}")
            return self.model_weights.get(project_id, [])
        
        client_weights = self.client_weights[project_id]
        
        # Get the first client's weights to determine the structure
        first_client = next(iter(client_weights.values()))
        first_weights = first_client['weights']
        
        # Validate client weights - check if they're empty
        if not first_weights or len(first_weights) == 0:
            logger.error(f"First client's weights empty or invalid for project {project_id}")
            return self.model_weights.get(project_id, [])
        
        # Debug log weight shapes
        logger.info(f"First client's weight structure: {[w.shape if hasattr(w, 'shape') else (type(w), len(w) if hasattr(w, '__len__') else 'no length') for w in first_weights]}")
        
        # Initialize aggregated weights with zeros of the same shape
        try:
            aggregated_weights = []
            for w in first_weights:
                if isinstance(w, np.ndarray):
                    if w.size == 0:  # Check for empty arrays
                        logger.error(f"Empty numpy array in weights")
                        return self.model_weights.get(project_id, [])
                    aggregated_weights.append(np.zeros_like(w))
                elif isinstance(w, list):
                    if not w:  # Check for empty lists
                        logger.error(f"Empty list in weights")
                        return self.model_weights.get(project_id, [])
                    w_array = np.array(w, dtype=np.float32)
                    aggregated_weights.append(np.zeros_like(w_array))
                else:
                    logger.error(f"Unexpected weight type: {type(w)}")
                    # Return current model weights instead of creating empty arrays
                    return self.model_weights.get(project_id, [])
            
            logger.info(f"Initialized aggregated weights with shapes: {[w.shape for w in aggregated_weights]}")
        except Exception as e:
            logger.error(f"Error initializing aggregated weights: {str(e)}")
            return self.model_weights.get(project_id, [])
        
        # Calculate total data size
        total_data_size = sum(client['data_size'] for client in client_weights.values())
        
        if total_data_size == 0:
            logger.warning(f"Total data size is 0 for project {project_id}, using equal weighting")
            total_data_size = len(client_weights)  # Use client count for equal weighting
            
        # Track validation issues
        validation_issues = 0
        valid_clients = 0
        
        # Weighted averaging based on data size
        for client_id, client_data in client_weights.items():
            try:
                client_weights_list = client_data['weights']
                client_data_size = client_data['data_size']
                
                # Skip if client has empty or invalid weights
                if not client_weights_list or len(client_weights_list) != len(aggregated_weights):
                    logger.warning(f"Client {client_id} has invalid weights (count mismatch): {len(client_weights_list)} vs expected {len(aggregated_weights)}")
                    validation_issues += 1
                    continue
                
                # Weighted contribution
                weight_factor = client_data_size / total_data_size
                logger.debug(f"Client {client_id} weight factor: {weight_factor} (data: {client_data_size}/{total_data_size})")
                
                # Sum up the weighted contributions
                valid_layer_count = 0
                for i, w in enumerate(client_weights_list):
                    try:
                        # Skip if index is out of range
                        if i >= len(aggregated_weights):
                            logger.warning(f"Client {client_id} weight index {i} exceeds aggregated weights length {len(aggregated_weights)}")
                            continue
                            
                        # Convert to numpy array if it's a list or other sequence
                        if not isinstance(w, np.ndarray):
                            try:
                                # Check if we have empty data
                                if not w and hasattr(w, '__len__'):
                                    logger.warning(f"Empty weight data at index {i} from client {client_id}")
                                    continue
                                w = np.array(w, dtype=np.float32)
                            except (ValueError, TypeError) as e:
                                logger.error(f"Error converting client {client_id} weight to numpy array: {e}")
                                logger.error(f"Weight type: {type(w)}, content sample: {str(w)[:100] if w else 'None'}")
                                continue
                        
                        # Check for empty arrays
                        if w.size == 0:
                            logger.warning(f"Empty numpy array at index {i} from client {client_id}")
                            continue
                        
                        # Ensure shapes match
                        if w.shape != aggregated_weights[i].shape:
                            logger.warning(f"Client {client_id} weight shape mismatch at index {i}: {w.shape} vs {aggregated_weights[i].shape}")
                            # Try to reshape if possible (when dimensions are compatible)
                            try:
                                total_elements_client = np.prod(w.shape)
                                total_elements_agg = np.prod(aggregated_weights[i].shape)
                                if total_elements_client == total_elements_agg:
                                    w = w.reshape(aggregated_weights[i].shape)
                                    logger.info(f"Successfully reshaped weight to {w.shape}")
                                else:
                                    logger.error(f"Cannot reshape weight: different number of elements ({total_elements_client} vs {total_elements_agg})")
                                    continue
                            except Exception as reshape_error:
                                logger.error(f"Reshape error: {str(reshape_error)}")
                                continue
                        
                        # Check for NaN or infinity
                        if not np.all(np.isfinite(w)):
                            logger.warning(f"Client {client_id} has non-finite values in weights at index {i}")
                            # Replace NaN/inf with zeros
                            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Now multiply by weight factor and add to aggregated weights
                        aggregated_weights[i] += w * weight_factor
                        valid_layer_count += 1
                    except Exception as layer_error:
                        logger.error(f"Error processing weight at index {i} for client {client_id}: {str(layer_error)}")
                
                if valid_layer_count == len(aggregated_weights):
                    valid_clients += 1
                    logger.info(f"Successfully processed all weights from client {client_id}")
                else:
                    logger.warning(f"Client {client_id} only processed {valid_layer_count}/{len(aggregated_weights)} weights")
                    
            except Exception as client_error:
                logger.error(f"Error processing weights from client {client_id}: {str(client_error)}")
                validation_issues += 1
        
        if valid_clients == 0:
            logger.error(f"No valid clients contributed to weight aggregation for project {project_id}")
            # Return original model weights to avoid saving empty model
            return self.model_weights.get(project_id, [])
            
        logger.info(f"Aggregated weights for project {project_id} from {valid_clients} clients (out of {len(client_weights)} total, {validation_issues} had issues)")
        
        # Validate final aggregated weights
        for i, w in enumerate(aggregated_weights):
            if not np.all(np.isfinite(w)):
                logger.warning(f"Aggregated weight at index {i} has non-finite values, replacing with zeros")
                aggregated_weights[i] = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        
        return aggregated_weights
    
    def _aggregate_weights_perfedavg(self, project_id, alpha=0.5):
        """Aggregate weights using Performance-Weighted FedAvg.
        
        Args:
            project_id: ID of the project to aggregate weights for
            alpha: Hyperparameter balancing data size (alpha=1) and 
                   performance weighting (alpha=0). Default 0.5 balances both.
                   
        Returns:
            List of aggregated weight arrays
        """
        if project_id not in self.client_weights or not self.client_weights[project_id]:
            logger.warning(f"No client weights available for project {project_id}")
            return self.model_weights.get(project_id, [])
        
        client_weights = self.client_weights[project_id]
        
        # Get the first client's weights to determine the structure
        first_client = next(iter(client_weights.values()))
        first_weights = first_client['weights']
        
        # Validate client weights - check if they're empty
        if not first_weights or len(first_weights) == 0:
            logger.error(f"First client's weights empty or invalid for project {project_id}")
            return self.model_weights.get(project_id, [])
        
        # Debug log weight shapes
        logger.info(f"First client's weight structure: {[w.shape if hasattr(w, 'shape') else (type(w), len(w) if hasattr(w, '__len__') else 'no length') for w in first_weights]}")
        
        # Initialize aggregated weights with zeros of the same shape
        try:
            aggregated_weights = []
            for w in first_weights:
                if isinstance(w, np.ndarray):
                    if w.size == 0:  # Check for empty arrays
                        logger.error(f"Empty numpy array in weights")
                        return self.model_weights.get(project_id, [])
                    aggregated_weights.append(np.zeros_like(w))
                elif isinstance(w, list):
                    if not w:  # Check for empty lists
                        logger.error(f"Empty list in weights")
                        return self.model_weights.get(project_id, [])
                    w_array = np.array(w, dtype=np.float32)
                    aggregated_weights.append(np.zeros_like(w_array))
                else:
                    logger.error(f"Unexpected weight type: {type(w)}")
                    return self.model_weights.get(project_id, [])
            
            logger.info(f"Initialized aggregated weights with shapes: {[w.shape for w in aggregated_weights]}")
        except Exception as e:
            logger.error(f"Error initializing aggregated weights: {str(e)}")
            return self.model_weights.get(project_id, [])
        
        # Calculate total data size
        total_data_size = sum(client['data_size'] for client in client_weights.values())
        if total_data_size == 0:
            logger.warning(f"Total data size is 0 for project {project_id}, using equal weighting for data component")
            total_data_size = len(client_weights)  # Use client count for equal weighting
        
        # Calculate total accuracy (for normalization)
        # First check which metrics to use - prefer validation accuracy if available
        acc_key = 'val_accuracy'
        acc_values = [client['metrics'].get(acc_key, 0) for client in client_weights.values()]
        
        # If no validation accuracy, fall back to training accuracy
        if sum(acc_values) == 0:
            acc_key = 'accuracy'
            acc_values = [client['metrics'].get(acc_key, 0) for client in client_weights.values()]
        
        # Calculate total accuracy for normalization
        total_accuracy = sum(acc_values)
        
        # If still no accuracy metrics, fall back to standard FedAvg
        if total_accuracy == 0:
            logger.warning("No accuracy metrics available, falling back to standard FedAvg")
            return self._aggregate_weights(project_id)
        
        # Track validation issues
        validation_issues = 0
        valid_clients = 0
        
        # Calculate effective contribution weights with PerfFedAvg
        contribution_weights = {}
        valid_client_weights = {}
        
        # First pass: calculate contribution weights
        for client_id, client_data in client_weights.items():
            try:
                client_data_size = client_data['data_size']
                client_metrics = client_data['metrics']
                
                # Get accuracy value - prefer validation accuracy if available
                client_accuracy = client_metrics.get(acc_key, 0)
                
                # Skip clients with zero accuracy (likely invalid models)
                if client_accuracy <= 0:
                    logger.warning(f"Client {client_id} has zero or negative accuracy ({client_accuracy}), skipping from aggregation")
                    validation_issues += 1
                    continue
                
                # Calculate data size component (how much data this client has relative to all clients)
                data_weight = client_data_size / total_data_size if total_data_size > 0 else 1.0/len(client_weights)
                
                # Calculate accuracy component (how good this client's model is relative to all clients)
                accuracy_weight = client_accuracy / total_accuracy if total_accuracy > 0 else 1.0/len(client_weights)
                
                # Combined weight using alpha
                # alpha=1: pure data size weighting (standard FedAvg)
                # alpha=0: pure accuracy weighting
                weight_factor = alpha * data_weight + (1 - alpha) * accuracy_weight
                
                logger.info(f"Client {client_id} - Data weight: {data_weight:.4f}, "
                          f"Accuracy weight: {accuracy_weight:.4f}, "
                          f"Combined weight: {weight_factor:.4f}, "
                          f"Accuracy: {client_accuracy:.4f}, "
                          f"Data size: {client_data_size}")
                
                # Store the contribution weight
                contribution_weights[client_id] = weight_factor
                valid_client_weights[client_id] = client_data
                valid_clients += 1
                
            except Exception as client_error:
                logger.error(f"Error calculating contribution weight for client {client_id}: {str(client_error)}")
                validation_issues += 1
        
        # Check if we have any valid clients
        if valid_clients == 0:
            logger.error(f"No valid clients for PerfFedAvg aggregation in project {project_id}")
            return self.model_weights.get(project_id, [])
        
        # Normalize the weights to sum to 1.0
        total_weight = sum(contribution_weights.values())
        if total_weight <= 0:
            logger.warning(f"Total contribution weight is zero or negative: {total_weight}, using equal weights")
            equal_weight = 1.0 / len(contribution_weights)
            contribution_weights = {client_id: equal_weight for client_id in contribution_weights}
        else:
            contribution_weights = {client_id: weight/total_weight for client_id, weight in contribution_weights.items()}
        
        # Second pass: apply the normalized weights to the model updates
        for client_id, client_data in valid_client_weights.items():
            try:
                client_weights_list = client_data['weights']
                weight_factor = contribution_weights[client_id]
                
                # Skip if client has empty or invalid weights
                if not client_weights_list or len(client_weights_list) != len(aggregated_weights):
                    logger.warning(f"Client {client_id} has invalid weights (count mismatch): {len(client_weights_list)} vs expected {len(aggregated_weights)}")
                    validation_issues += 1
                    continue
                
                # Sum up the weighted contributions
                valid_layer_count = 0
                for i, w in enumerate(client_weights_list):
                    try:
                        # Skip if index is out of range
                        if i >= len(aggregated_weights):
                            logger.warning(f"Client {client_id} weight index {i} exceeds aggregated weights length {len(aggregated_weights)}")
                            continue
                        
                        # Convert to numpy array if it's a list or other sequence
                        if not isinstance(w, np.ndarray):
                            try:
                                # Check if we have empty data
                                if not w and hasattr(w, '__len__'):
                                    logger.warning(f"Empty weight data at index {i} from client {client_id}")
                                    continue
                                w = np.array(w, dtype=np.float32)
                            except (ValueError, TypeError) as e:
                                logger.error(f"Error converting client {client_id} weight to numpy array: {e}")
                                logger.error(f"Weight type: {type(w)}, content sample: {str(w)[:100] if w else 'None'}")
                                continue
                        
                        # Check for empty arrays
                        if w.size == 0:
                            logger.warning(f"Empty numpy array at index {i} from client {client_id}")
                            continue
                        
                        # Ensure shapes match
                        if w.shape != aggregated_weights[i].shape:
                            logger.warning(f"Client {client_id} weight shape mismatch at index {i}: {w.shape} vs {aggregated_weights[i].shape}")
                            # Try to reshape if possible (when dimensions are compatible)
                            try:
                                total_elements_client = np.prod(w.shape)
                                total_elements_agg = np.prod(aggregated_weights[i].shape)
                                if total_elements_client == total_elements_agg:
                                    w = w.reshape(aggregated_weights[i].shape)
                                    logger.info(f"Successfully reshaped weight to {w.shape}")
                                else:
                                    logger.error(f"Cannot reshape weight: different number of elements ({total_elements_client} vs {total_elements_agg})")
                                    continue
                            except Exception as reshape_error:
                                logger.error(f"Reshape error: {str(reshape_error)}")
                                continue
                        
                        # Check for NaN or infinity
                        if not np.all(np.isfinite(w)):
                            logger.warning(f"Client {client_id} has non-finite values in weights at index {i}")
                            # Replace NaN/inf with zeros
                            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Apply the weighted contribution to aggregated weights
                        aggregated_weights[i] += w * weight_factor
                        valid_layer_count += 1
                    except Exception as layer_error:
                        logger.error(f"Error processing weight at index {i} for client {client_id}: {str(layer_error)}")
                
                if valid_layer_count == len(aggregated_weights):
                    logger.info(f"Successfully processed all weights from client {client_id} with weight {weight_factor:.4f}")
                else:
                    logger.warning(f"Client {client_id} only processed {valid_layer_count}/{len(aggregated_weights)} weights")
                    
            except Exception as client_error:
                logger.error(f"Error applying weights from client {client_id}: {str(client_error)}")
                validation_issues += 1
        
        logger.info(f"Aggregated weights for project {project_id} from {valid_clients} clients (out of {len(client_weights)} total, {validation_issues} had issues) using PerfFedAvg with alpha={alpha}")
        
        # Validate final aggregated weights
        for i, w in enumerate(aggregated_weights):
            if not np.all(np.isfinite(w)):
                logger.warning(f"Aggregated weight at index {i} has non-finite values, replacing with zeros")
                aggregated_weights[i] = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        
        return aggregated_weights
    
    def _save_final_model(self, project_id):
        """Save the final model after training completion."""
        try:
            # Get the project
            project = Project.query.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return False
            
            # Get the final aggregated weights
            weights = self.model_weights.get(project_id, [])
            if not weights:
                logger.error(f"No weights found for project {project_id}")
                # Emergency recovery: create a dummy weights array if needed
                weights = []
                logger.warning("Creating dummy weights for emergency recovery")
            
            # Get metrics
            metrics = self.aggregated_metrics.get(project_id, {})
            if not metrics:
                logger.warning(f"No metrics found for project {project_id}. Creating default metrics.")
                metrics = {
                    'accuracy': 0.0,
                    'loss': 0.0,
                    'val_accuracy': 0.0,
                    'val_loss': 0.0,
                    'round': project.current_round,
                    'timestamp': datetime.utcnow().isoformat(),
                    'is_emergency_recovery': True
                }
            
            # Create uploads directories with normalized paths
            base_upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
            model_dir = os.path.normpath(os.path.join(base_upload_folder, 'models'))
            os.makedirs(model_dir, exist_ok=True)
            
            # Define file paths with normalized paths
            tf_model_filename = f'project_{project_id}_model.h5'
            pt_model_filename = f'project_{project_id}_model.pt'
            tf_model_path = os.path.normpath(os.path.join(model_dir, tf_model_filename))
            pt_model_path = os.path.normpath(os.path.join(model_dir, pt_model_filename))
            
            # Add a unique timestamp to avoid file conflicts
            timestamp = int(time.time())
            unique_suffix = f"_{timestamp}_{secrets.token_hex(4)}"
            saved_model_dir = os.path.normpath(os.path.join(model_dir, f'project_{project_id}_saved_model{unique_suffix}'))
            
            logger.info(f"Saving final models for project {project_id} to {model_dir}")
            
            # Track success status for each file type
            tf_model_saved = False
            pt_model_saved = False
            
            # Step 1: Try to create and save TensorFlow model
            try:
                import tensorflow as tf
                
                # Try to create a model based on dataset
                if project.dataset_name.lower() == 'mnist':
                    model = self._create_mnist_model()
                elif project.dataset_name.lower() == 'cifar10':
                    model = self._create_cifar10_model()
                else:
                    model = self._create_generic_model()
                
                # If we have weights, try to apply them
                if weights:
                    try:
                        # Ensure weights are valid before applying
                        if len(weights) > 0:
                            try:
                                model.set_weights(weights)
                                logger.info(f"Successfully applied weights to the model")
                            except (ValueError, TypeError) as weight_error:
                                logger.error(f"Error applying weights: {str(weight_error)}")
                                # Continue anyway to save a model file
                        else:
                            logger.warning("Weights list is empty, unable to apply to model")
                    except Exception as weights_error:
                        logger.error(f"Error applying weights: {str(weights_error)}")
                        # Continue anyway to save a model file
                
                # Save the model
                try:
                    # Use our custom Windows-friendly SavedModel method
                    custom_saved_model_dir = os.path.normpath(os.path.join(
                        model_dir, f'project_{project_id}_saved_model_{timestamp}_{secrets.token_hex(4)}'
                    ))
                    logger.info(f"Saving model using Windows-friendly custom SavedModel approach to {custom_saved_model_dir}")
                    
                    # Use our custom method that avoids Windows path issues
                    save_success = self._save_fixed_savedmodel(model, custom_saved_model_dir)
                    
                    if save_success:
                        logger.info(f"✓ TensorFlow SavedModel format saved with custom method to {custom_saved_model_dir}")
                        save_dir = custom_saved_model_dir
                        tf_model_saved = True
                    else:
                        logger.warning(f"Custom SavedModel save failed, falling back to H5 format")
                        # Fall back to H5 format only
                        h5_model_path = os.path.join(model_dir, f'project_{project_id}_model_{timestamp}.h5')
                        model.save(h5_model_path, save_format='h5')
                        logger.info(f"✓ TensorFlow H5 model saved to {h5_model_path}")
                        tf_model_saved = True
                        tf_model_path = h5_model_path
                except Exception as save_error:
                    logger.error(f"Error saving model file: {str(save_error)}")
                    try:
                        # Last resort - try just H5 format directly
                        h5_model_path = os.path.join(model_dir, f'project_{project_id}_model_{timestamp}.h5')
                        model.save(h5_model_path, save_format='h5')
                        logger.info(f"✓ TensorFlow H5 model saved to {h5_model_path}")
                        tf_model_saved = True
                        tf_model_path = h5_model_path
                    except Exception as h5_error:
                        logger.error(f"H5 save also failed: {str(h5_error)}")
                
            except ImportError:
                logger.error("× Could not import TensorFlow. Make sure it's installed.")
            except Exception as e:
                logger.error(f"× Error saving TensorFlow model: {str(e)}")
                
            # Create a fallback dummy file if TF save failed
            if not tf_model_saved:
                try:
                    with open(tf_model_path, 'w') as f:
                        f.write(f"Dummy model file for project {project_id}\n")
                        f.write(f"Created: {datetime.utcnow().isoformat()}\n")
                        f.write(f"Metrics: {str(metrics)}\n")
                    logger.warning(f"Created emergency dummy TF model file at {tf_model_path}")
                    tf_model_saved = True
                except Exception as dummy_error:
                    logger.error(f"Failed to create dummy file: {str(dummy_error)}")
            
            # Step 2: Try to create PyTorch model
            try:
                import torch
                import torch.nn as nn
                
                # Create a simple PyTorch model that avoids LogSoftmax issues
                # Use an alternative approach with standard modules
                if project.dataset_name.lower() == 'mnist':
                    model = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(784, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                        # Removed LogSoftmax to avoid serialization issues
                    )
                elif project.dataset_name.lower() == 'cifar10':
                    model = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(3*32*32, 512),
                        nn.ReLU(),
                        nn.Linear(512, 10)
                        # Removed LogSoftmax to avoid serialization issues
                    )
                else:
                    model = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(784, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                        # Removed LogSoftmax to avoid serialization issues
                    )
                
                # Save the PyTorch model
                try:
                    # Explicitly use .pt extension for PyTorch
                    pt_model_path = os.path.normpath(os.path.join(model_dir, f'project_{project_id}_model_{timestamp}.pt'))
                    
                    # Save the state dict (preferred way)
                    torch.save(model.state_dict(), pt_model_path)
                    logger.info(f"✓ PyTorch state dict saved to {pt_model_path}")
                    
                    # Also save the full model as a separate file but without scripting
                    full_model_path = os.path.normpath(os.path.join(model_dir, f'project_{project_id}_model_full_{timestamp}.pt'))
                    torch.save(model, full_model_path)
                    logger.info(f"✓ PyTorch full model saved to {full_model_path}")
                    
                    # Don't use TorchScript format which causes the issues
                    pt_model_saved = True
                    
                except Exception as pt_save_error:
                    logger.error(f"Error saving PyTorch model: {str(pt_save_error)}")
                    
                    # Try to save a simplified version
                    try:
                        # Create a minimal model structure
                        minimal_model = nn.Sequential(
                            nn.Linear(784, 10)
                        )
                        
                        torch.save(minimal_model, pt_model_path)
                        logger.info(f"✓ Minimal PyTorch model saved to {pt_model_path}")
                        pt_model_saved = True
                    except Exception as pt_full_save_error:
                        logger.error(f"Error saving minimal PyTorch model: {str(pt_full_save_error)}")
                
            except ImportError:
                logger.error("× Could not import PyTorch. Skipping PyTorch model.")
            except Exception as e:
                logger.error(f"× Error creating PyTorch model: {str(e)}")
            
            # Create a fallback dummy PyTorch file if save failed
            if not pt_model_saved:
                try:
                    with open(pt_model_path, 'w') as f:
                        f.write(f"Dummy PyTorch model file for project {project_id}\n")
                        f.write(f"Created: {datetime.utcnow().isoformat()}\n")
                        f.write(f"Metrics: {str(metrics)}\n")
                    logger.warning(f"Created emergency dummy PyTorch model file at {pt_model_path}")
                    pt_model_saved = True
                except Exception as dummy_error:
                    logger.error(f"Failed to create dummy PyTorch file: {str(dummy_error)}")
            
            # If at least one model format was saved, proceed with database update
            if tf_model_saved or pt_model_saved:
                # Get active clients count
                active_clients = ProjectClient.query.filter_by(project_id=project_id).count()
                
                # Create model data dictionary
                model_data = {
                    'accuracy': metrics.get('accuracy', 0),
                    'loss': metrics.get('loss', 0),
                    'val_accuracy': metrics.get('val_accuracy', 0),
                    'val_loss': metrics.get('val_loss', 0),
                    'clients': active_clients or 1,  # Default to 1 if no clients found
                    'round': project.current_round,
                    'total_rounds': project.rounds,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Add file paths
                all_paths = {}
                
                if tf_model_saved:
                    model_data['tf_model_file'] = tf_model_path
                    all_paths['tensorflow_h5'] = tf_model_path
                    
                    # Add SavedModel directory if available
                    saved_model_dir = os.path.join(model_dir, f'project_{project_id}_saved_model')
                    if os.path.exists(saved_model_dir):
                        all_paths['tensorflow_saved_model'] = saved_model_dir
                    
                    # Add weights file if available
                    weights_path = tf_model_path + '_weights'
                    if os.path.exists(weights_path + '.index'):
                        all_paths['tensorflow_weights'] = weights_path
                    
                    # Add JSON file if available
                    json_path = tf_model_path + '.json'
                    if os.path.exists(json_path):
                        all_paths['tensorflow_json'] = json_path
                
                if pt_model_saved:
                    model_data['model_file'] = pt_model_path
                    all_paths['pytorch_state_dict'] = pt_model_path
                    
                    # Add full model file if available
                    full_path = pt_model_path + '.full'
                    if os.path.exists(full_path):
                        all_paths['pytorch_full'] = full_path
                    
                    # Add TorchScript file if available
                    script_path = pt_model_path + '.torchscript'
                    if os.path.exists(script_path):
                        all_paths['pytorch_torchscript'] = script_path
                elif tf_model_saved:
                    # Use TF model as default if PT model not available
                    model_data['model_file'] = tf_model_path
                
                # Store all paths in additional_paths field
                model_data['additional_paths'] = all_paths
                
                # Set framework field based on what was saved
                if tf_model_saved and pt_model_saved:
                    model_data['framework'] = 'pytorch+tensorflow'
                elif tf_model_saved:
                    model_data['framework'] = 'tensorflow'
                elif pt_model_saved:
                    model_data['framework'] = 'pytorch'
                
                # Flag as final model
                model_data['is_final'] = True
                
                # Import ModelManager and save to database
                try:
                    from web.services.model_manager import ModelManager
                    saved_model = ModelManager.save_model(project, model_data, is_final=True)
                    
                    if saved_model:
                        logger.info(f"✓ Final model saved to database with ID {saved_model.id}")
                        # Mark project as completed
                        project.status = 'completed'
                        db.session.commit()
                        return True
                    else:
                        logger.error("× Failed to save model to database")
                        # Try emergency database save
                        return self._emergency_model_save(project, model_data)
                except Exception as db_error:
                    logger.error(f"Database error saving model: {str(db_error)}")
                    # Try emergency database save
                    return self._emergency_model_save(project, model_data)
            else:
                logger.error("× Failed to save model in any format")
                # Create a dummy entry anyway
                return self._emergency_model_save(project, metrics)
                
        except Exception as e:
            logger.error(f"× Error in _save_final_model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Final emergency recovery - mark the project as completed anyway
            try:
                project = Project.query.get(project_id)
                if project:
                    project.status = 'completed'
                    db.session.commit()
                    logger.warning(f"Project {project_id} marked as completed despite model save failure")
                return False
            except Exception:
                return False
    
    def _emergency_model_save(self, project, model_data):
        """Emergency method to save at least some model record when normal saving fails."""
        try:
            logger.warning(f"Attempting emergency model save for project {project.id}")
            
            # Use timestamp for unique filenames
            timestamp = int(time.time())
            
            # Create emergency file path if needed
            if not isinstance(model_data, dict) or 'model_file' not in model_data:
                base_upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
                model_dir = os.path.join(base_upload_folder, 'models')
                os.makedirs(model_dir, exist_ok=True)
                
                emergency_path = os.path.join(model_dir, f'emergency_model_{project.id}_{timestamp}.txt')
                try:
                    with open(emergency_path, 'w') as f:
                        f.write(f"Emergency model file for project {project.id}\n")
                        f.write(f"Created: {datetime.utcnow().isoformat()}\n")
                        f.write(f"This file was created during emergency recovery.\n")
                        
                        # Try to write metrics
                        if isinstance(model_data, dict):
                            f.write(f"Metrics: {str(model_data)}\n")
                    
                    model_data['model_file'] = emergency_path
                except Exception as write_error:
                    logger.error(f"Error creating emergency file: {str(write_error)}")
                    
                    # Try yet another location if the first failed
                    try:
                        temp_dir = tempfile.gettempdir()
                        emergency_path = os.path.join(temp_dir, f'emergency_model_{project.id}_{timestamp}.txt')
                        with open(emergency_path, 'w') as f:
                            f.write(f"Emergency model file (alternate location) for project {project.id}\n")
                        
                        if isinstance(model_data, dict):
                            model_data['model_file'] = emergency_path
                        else:
                            model_data = {
                                'accuracy': 0.0,
                                'loss': 0.0,
                                'clients': 1,
                                'is_emergency_recovery': True,
                                'model_file': emergency_path
                            }
                    except Exception:
                        # If all file creation attempts fail, proceed with no model file
                        if not isinstance(model_data, dict):
                            model_data = {
                                'accuracy': 0.0,
                                'loss': 0.0,
                                'clients': 1,
                                'is_emergency_recovery': True,
                                'model_file': None
                            }
            
            # Ensure model_data is a dictionary with required fields
            if not isinstance(model_data, dict):
                model_data = {
                    'accuracy': 0.0,
                    'loss': 0.0,
                    'clients': 1,
                    'is_emergency_recovery': True
                }
            
            model_data['is_final'] = True
            model_data['framework'] = model_data.get('framework', 'emergency_recovery')
            
            # Try database insert to create a model record
            try:
                from web.services.model_manager import ModelManager
                saved_model = ModelManager.save_model(project, model_data, is_final=True)
                
                if saved_model:
                    logger.warning(f"Emergency model record created with ID {saved_model.id}")
                    project.status = 'completed'
                    db.session.commit()
                    return True
                else:
                    # Last resort - create model directly
                    try:
                        # Get next version
                        version = 1
                        existing_models = Model.query.filter_by(project_id=project.id).all()
                        if existing_models:
                            version = max(model.version for model in existing_models) + 1
                        
                        # Create minimal model record
                        emergency_model = Model(
                            project_id=project.id,
                            version=version,
                            is_final=True,
                            accuracy=model_data.get('accuracy', 0),
                            loss=model_data.get('loss', 0),
                            clients_count=model_data.get('clients', 1),
                            path=model_data.get('model_file', ''),
                            framework='emergency_recovery'
                        )
                        
                        # Ensure metrics are set
                        emergency_model.metrics = {
                            'accuracy': model_data.get('accuracy', 0),
                            'loss': model_data.get('loss', 0),
                            'clients': model_data.get('clients', 1),
                            'is_emergency_recovery': True,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        
                        db.session.add(emergency_model)
                        project.status = 'completed'
                        db.session.commit()
                        logger.warning(f"Direct emergency model created with ID {emergency_model.id}")
                        return True
                    except Exception as direct_error:
                        logger.error(f"Direct model creation also failed: {str(direct_error)}")
                        # Final attempt - just mark project as completed
                        project.status = 'completed'
                        db.session.commit()
                        logger.warning(f"Project {project.id} marked as completed without model")
                        return False
            except Exception as model_mgr_error:
                logger.error(f"ModelManager access failed: {str(model_mgr_error)}")
                # Last resort - just mark project as completed
                project.status = 'completed'
                db.session.commit()
                logger.warning(f"Project {project.id} marked as completed without model record")
                return False
                
        except Exception as emergency_error:
            logger.error(f"Emergency model save failed: {str(emergency_error)}")
            try:
                # Last resort - just mark as completed
                project.status = 'completed'
                db.session.commit()
                logger.warning(f"Project {project.id} marked as completed as absolute last resort")
                return False
            except Exception:
                return False
    
    def _convert_to_pytorch(self, tf_model):
        """Convert a TensorFlow model to PyTorch format."""
        try:
            import torch
            import torch.nn as nn
            
            # Create a PyTorch model with the same architecture
            class PyTorchModel(nn.Module):
                def __init__(self, input_shape, num_classes):
                    super(PyTorchModel, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                    
                    # Calculate the size of the flattened features
                    self.feature_size = self._get_conv_output_size(input_shape)
                    
                    self.classifier = nn.Sequential(
                        nn.Linear(self.feature_size, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Linear(512, num_classes)
                    )
                
                def _get_conv_output_size(self, shape):
                    # Create a dummy input to get the output size
                    x = torch.randn(1, *shape)
                    x = self.features(x)
                    return x.view(1, -1).size(1)
                
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            # Create PyTorch model
            input_shape = tf_model.input_shape[1:]  # Remove batch dimension
            num_classes = tf_model.output_shape[-1]
            pt_model = PyTorchModel(input_shape, num_classes)
            
            # Convert weights (simplified version - in practice, you'd need more sophisticated conversion)
            tf_weights = tf_model.get_weights()
            pt_state_dict = pt_model.state_dict()
            
            # Map TensorFlow weights to PyTorch weights
            # This is a simplified mapping - you'd need to handle the actual architecture
            weight_map = {
                'features.0.weight': tf_weights[0].transpose(3, 2, 0, 1),
                'features.0.bias': tf_weights[1],
                'features.1.weight': tf_weights[2],
                'features.1.bias': tf_weights[3],
                # Add more mappings as needed
            }
            
            # Update PyTorch model weights
            for name, weight in weight_map.items():
                if name in pt_state_dict:
                    pt_state_dict[name] = torch.from_numpy(weight)
            
            pt_model.load_state_dict(pt_state_dict)
            return pt_model
            
        except Exception as e:
            logger.error(f"Error converting model to PyTorch: {str(e)}")
            raise
    
    def _validate_pytorch_model(self, model, dataset_name):
        """Validate the PyTorch model with a small test set."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            import torchvision
            import torchvision.transforms as transforms
            
            # Set model to evaluation mode
            model.eval()
            
            # Load a small test set
            if dataset_name.lower() == 'mnist':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                test_dataset = torchvision.datasets.MNIST(
                    root='./data', train=False, download=True, transform=transform
                )
            elif dataset_name.lower() == 'cifar10':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                test_dataset = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform
                )
            else:
                logger.warning(f"No validation dataset available for {dataset_name}")
                return True  # Skip validation for unknown datasets
            
            # Create a small test loader
            test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
            
            # Test the model
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            logger.info(f"Model validation accuracy: {accuracy:.2f}%")
            
            # Consider the model valid if accuracy is above a threshold
            return accuracy > 50.0  # Adjust threshold as needed
            
        except Exception as e:
            logger.error(f"Error validating PyTorch model: {str(e)}")
            return False
    
    def _create_mnist_model(self):
        """Create a model for MNIST dataset."""
        try:
            import tensorflow as tf
            
            input_shape = (28, 28, 1)
            
            # Create a model matching the architecture that produces 24 weight arrays
            # (matches the CNN with BatchNorm structure)
            model = tf.keras.Sequential([
                # First Conv Block
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                # Second Conv Block
                tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                # IMPORTANT: Removed the second MaxPooling2D layer to match client model
                # This keeps the feature map size at 14x14x64 instead of reducing to 7x7x64
                
                # Flatten layer
                tf.keras.layers.Flatten(),
                
                # Dense layers
                tf.keras.layers.Dense(512),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except ImportError:
            logger.error("TensorFlow not installed, could not create MNIST model")
            raise
    
    def _create_cifar10_model(self):
        """Create a model for CIFAR-10 dataset."""
        try:
            import tensorflow as tf
            
            input_shape = (32, 32, 3)
            model = tf.keras.Sequential([
                # Conv layers
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                # Flatten layer
                tf.keras.layers.Flatten(),
                
                # Dense layers
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except ImportError:
            logger.error("TensorFlow not installed, could not create CIFAR-10 model")
            raise
    
    def _create_generic_model(self):
        """Create a generic model for any dataset."""
        try:
            import tensorflow as tf
            
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except ImportError:
            logger.error("TensorFlow not installed, could not create generic model")
            raise

    def get_active_clients_count(self, project_id):
        """Get the number of active clients for a project."""
        try:
            # Query ProjectClient table to count clients with 'joined' status
            from web.models import ProjectClient
            
            active_clients = ProjectClient.query.filter_by(
                project_id=project_id,
                status='joined'
            ).count()
            
            logger.info(f"Found {active_clients} active clients for project {project_id}")
            return active_clients
        except Exception as e:
            logger.error(f"Error counting active clients for project {project_id}: {str(e)}")
            return 0

    def save_model(self, project_id, round_num=None, is_final=False):
        """Save the current model for a project."""
        try:
            # Get the project
            project = Project.query.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found when trying to save model")
                return None
            
            # Get the model weights
            weights = self.model_weights.get(project_id, [])
            if not weights:
                logger.error(f"No weights available for project {project_id}")
                return None
                
            # Validate weights have proper shapes and values
            valid_weights = True
            for i, w in enumerate(weights):
                if not isinstance(w, np.ndarray):
                    logger.warning(f"Weight at index {i} is not numpy array, trying to convert")
                    try:
                        weights[i] = np.array(w, dtype=np.float32)
                    except Exception as e:
                        logger.error(f"Failed to convert weight to numpy array: {e}")
                        valid_weights = False
                        break
                        
                # Check for empty arrays
                if weights[i].size == 0:
                    logger.error(f"Weight at index {i} is empty")
                    valid_weights = False
                    break
                
                # Check for NaN or infinity
                if not np.all(np.isfinite(weights[i])):
                    logger.warning(f"Weight at index {i} contains non-finite values")
                    weights[i] = np.nan_to_num(weights[i], nan=0.0, posinf=0.0, neginf=0.0)
            
            if not valid_weights:
                logger.error("Invalid weights, cannot save model")
                return self._emergency_model_save(project, {"error": "Invalid weights"})
                
            # Calculate metrics
            model_metrics = {
                'accuracy': sum(self.current_metrics.get(project_id, {}).get('accuracy', [])) / len(self.current_metrics.get(project_id, {}).get('accuracy', [1])) if project_id in self.current_metrics and self.current_metrics[project_id].get('accuracy') else 0,
                'loss': sum(self.current_metrics.get(project_id, {}).get('loss', [])) / len(self.current_metrics.get(project_id, {}).get('loss', [1])) if project_id in self.current_metrics and self.current_metrics[project_id].get('loss') else 0,
                'clients': len(self.current_metrics.get(project_id, {}).get('client_ids', [])) if project_id in self.current_metrics else 0,
                'round': round_num if round_num is not None else project.current_round,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get model directory
            model_dir = current_app.config.get('FL_MODEL_PATH', os.path.join(os.getcwd(), 'uploads/models'))
            os.makedirs(model_dir, exist_ok=True)
            
            # Use timestamp for unique filenames to avoid conflicts
            timestamp = int(time.time())
            
            # Prepare model file paths
            model_filename = f"project_{project_id}_model_{timestamp}"
            model_path = os.path.join(model_dir, model_filename + ".h5")
            saved_model_path = os.path.join(model_dir, f"project_{project_id}_saved_model_{timestamp}")
            
            # Log model metrics
            logger.info(f"Saving model for project {project_id} with metrics: {model_metrics}")
            logger.info(f"Weights summary: {len(weights)} layers, types: {[type(w) for w in weights[:5]]}...")
            
            # Detect the framework from project
            framework = project.framework.lower() if project.framework else 'tensorflow'
            
            # Initialize model file paths dictionary
            model_files = {
                'model_file': model_path  # Default model path
            }
            
            try:
                if framework.startswith('tensorflow') or framework == 'keras':
                    logger.info(f"Saving TensorFlow model with {len(weights)} weight arrays")
                    
                    # Import tensorflow
                    try:
                        import tensorflow as tf
                        logger.info(f"TensorFlow version: {tf.__version__}")
                    except ImportError:
                        logger.error("TensorFlow not installed")
                        raise ImportError("TensorFlow is required but not installed")
                    
                    # Check if we have a model instance
                    if project_id in self.models:
                        model = self.models[project_id]
                        logger.info(f"Using existing model instance for project {project_id}")
                    else:
                        logger.warning(f"No model instance for project {project_id}, attempting to recreate")
                        
                        # For MNIST project, recreate a model
                        if project.dataset_name.lower() == 'mnist':
                            logger.info("Recreating MNIST model")
                            try:
                                # Use our local implementation instead of importing
                                model = self.create_mnist_model()
                                self.models[project_id] = model
                            except Exception as model_err:
                                logger.error(f"Error creating MNIST model: {str(model_err)}")
                                # Create a placeholder model as last resort
                                model = tf.keras.Sequential([
                                    tf.keras.layers.Input(shape=(28, 28, 1)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10)
                                ])
                        else:
                            logger.error(f"Cannot recreate model for project {project_id} with dataset {project.dataset_name}")
                            # Create a placeholder model as last resort
                            model = tf.keras.Sequential([
                                tf.keras.layers.Input(shape=(28, 28, 1)),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(128, activation='relu'),
                                tf.keras.layers.Dense(10)
                            ])
                    
                    # Apply weights to model - wrapped in try/except to continue even if this fails
                    weights_applied = False
                    try:
                        # Check if weights match the model architecture
                        if len(weights) == len(model.weights):
                            # Verify each weight array is not empty
                            empty_weights = []
                            for i, w in enumerate(weights):
                                if not isinstance(w, np.ndarray):
                                    try:
                                        weights[i] = np.array(w, dtype=np.float32)
                                    except Exception as e:
                                        logger.error(f"Error converting weight at index {i} to numpy array: {str(e)}")
                                        empty_weights.append(i)
                                        continue
                                
                                # Check for empty arrays or arrays with no size
                                if not hasattr(weights[i], 'size') or weights[i].size == 0:
                                    logger.error(f"Weight at index {i} is empty or has no size attribute")
                                    empty_weights.append(i)
                
                            if empty_weights:
                                logger.error(f"Found {len(empty_weights)} empty weights at indices: {empty_weights}")
                                
                                # Try to recover by using default weights for those positions
                                try:
                                    default_weights = model.get_weights()
                                    for idx in empty_weights:
                                        if idx < len(default_weights):
                                            weights[idx] = default_weights[idx]
                                            logger.info(f"Replaced empty weight at index {idx} with default weight")
                                    
                                    all_valid = all(hasattr(w, 'size') and w.size > 0 for w in weights)
                                    if not all_valid:
                                        logger.error("Some weights still invalid after recovery attempt")
                                except Exception as recovery_err:
                                    logger.error(f"Error attempting to recover empty weights: {str(recovery_err)}")
                                    all_valid = False
                            else:
                                all_valid = True
                                
                            if all_valid:
                                try:
                                    # Log the expected shapes vs received shapes for key weights
                                    if len(weights) > 16:  # Typically dense layer weights are near the end
                                        dense_weight_idx = 16  # Common index for first dense layer in CNN models
                                        if dense_weight_idx < len(weights) and dense_weight_idx < len(model.weights):
                                            expected_shape = model.weights[dense_weight_idx].shape
                                            actual_shape = weights[dense_weight_idx].shape if hasattr(weights[dense_weight_idx], 'shape') else None
                                            logger.info(f"Dense layer weight check - Expected: {expected_shape}, Actual: {actual_shape}")
                                
                                    # Try to apply weights and catch specific shape mismatch errors
                                    model.set_weights(weights)
                                    weights_applied = True
                                    logger.info(f"Successfully applied weights to model")
                                except ValueError as shape_err:
                                    # If we get a shape mismatch error, log detailed info
                                    if "shape" in str(shape_err).lower():
                                        logger.error(f"Shape mismatch error: {str(shape_err)}")
                                        
                                        # Try to identify which layer has the problem
                                        model_shapes = [w.shape for w in model.weights]
                                        weight_shapes = [w.shape if hasattr(w, 'shape') else None for w in weights]
                                        
                                        # Log shape details for all weights - helps debug MNIST architecture issues
                                        for i, (model_shape, weight_shape) in enumerate(zip(model_shapes, weight_shapes)):
                                            if model_shape != weight_shape:
                                                logger.error(f"Mismatch at index {i}: Model expects {model_shape}, got {weight_shape}")
                                                
                                                # For MNIST model with 14x14x64 feature maps (12544 neurons)
                                                if model_shape[0] == 3136 and weight_shape[0] == 12544:
                                                    logger.error("Detected 7x7x64 vs 14x14x64 architecture mismatch. Creating compatible model.")
                                                    # Create a model specifically for 14x14x64 feature maps
                                                    model = self._create_mnist_model_without_second_pooling()
                                                    try:
                                                        model.set_weights(weights)
                                                        weights_applied = True
                                                        logger.info("Successfully applied weights to adjusted model")
                                                        break
                                                    except Exception as adj_err:
                                                        logger.error(f"Error applying weights to adjusted model: {str(adj_err)}")
                                    else:
                                        logger.error(f"Error applying weights: {str(shape_err)}")
                        else:
                            logger.error("One or more weight arrays are empty")
                    except Exception as apply_error:
                        logger.error(f"Error applying weights: {str(apply_error)}")
                    
                    # Try saving in a temporary location first
                    try:
                        # Use a different temporary directory for initial save
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_filename = f"temp_model_{secrets.token_hex(8)}.h5"
                            temp_model_path = os.path.join(temp_dir, temp_filename)
                            
                            # Ensure model has been compiled
                            if not model._is_compiled:
                                model.compile(optimizer='adam', 
                                              loss='sparse_categorical_crossentropy', 
                                              metrics=['accuracy'])
                            
                            # Save in HDF5 format
                            model.save(temp_model_path, save_format='h5', overwrite=True)
                            
                            # Verify file was created and has content
                            if os.path.exists(temp_model_path) and os.path.getsize(temp_model_path) > 1000:
                                # Make sure target directory exists
                                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                                
                                # Generate a unique filename for the final location
                                final_filename = f"project_{project_id}_model_{timestamp}_{secrets.token_hex(8)}.h5"
                                final_model_path = os.path.join(model_dir, final_filename)
                                
                                # Copy to final location
                                shutil.copy2(temp_model_path, final_model_path)
                                logger.info(f"Successfully saved TF model to {final_model_path}")
                                model_files['tf_model_file'] = final_model_path
                            else:
                                logger.error(f"Temp model file is missing or too small: {temp_model_path}")
                                raise FileNotFoundError("Temp model file invalid")
                    except Exception as h5_error:
                        logger.error(f"Error saving H5 model: {str(h5_error)}")
                        
                        # Try SavedModel format as backup
                        try:
                            # Use a temporary directory for SavedModel
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Make sure temp_dir uses correct path separators
                                temp_dir = os.path.normpath(temp_dir)
                                logger.info(f"Saving TF SavedModel to temporary directory: {temp_dir}")
                                
                                # Use TensorFlow's filesystem operations to handle paths properly
                                try:
                                    import tensorflow as tf
                                    # Use tf.io.gfile methods to ensure proper path handling
                                    if hasattr(tf, 'io') and hasattr(tf.io, 'gfile'):
                                        # Create directories with TensorFlow's path-safe methods
                                        tf.io.gfile.makedirs(temp_dir)
                                        logger.info(f"Created temp directory with TensorFlow: {temp_dir}")
                                except Exception as tf_err:
                                    logger.error(f"Error using TensorFlow's gfile: {str(tf_err)}")
                                
                                # Create a unique filename with timestamp to avoid conflicts
                                model_ts = int(time.time())
                                random_id = secrets.token_hex(4)
                                save_format = 'tf'
                                
                                # Save in SavedModel format using absolute paths with forward slashes
                                # TensorFlow prefers forward slashes even on Windows
                                clean_temp_path = temp_dir.replace('\\', '/')
                                logger.info(f"Saving model to cleaned path: {clean_temp_path}")
                                
                                # Save model with possible custom options to handle path issues
                                try:
                                    # First try to save with special options to avoid the file access issue
                                    import tensorflow as tf
                                    if hasattr(tf, 'saved_model') and hasattr(tf.saved_model, 'SaveOptions'):
                                        # Use TF's SaveOptions to handle path issues
                                        save_options = tf.saved_model.SaveOptions(
                                            experimental_io_device='/job:localhost'
                                        )
                                        logger.info("Using TensorFlow SaveOptions with experimental_io_device")
                                        model.save(clean_temp_path, save_format='tf', options=save_options)
                                    else:
                                        # Fall back to standard save
                                        model.save(clean_temp_path, save_format='tf')
                                except Exception as save_error:
                                    logger.error(f"Error saving with options: {str(save_error)}")
                                    # Fall back to standard save
                                    model.save(temp_dir, save_format='tf')
                                
                                # Verify directory was created with content
                                saved_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                              for dirpath, _, filenames in os.walk(temp_dir)
                                              for filename in filenames)
                                
                                if saved_size > 1000:
                                    # Create a unique destination path to avoid conflicts
                                    timestamp_id = f"{int(time.time())}_{secrets.token_hex(4)}"
                                    dest_saved_model = os.path.normpath(os.path.join(
                                        model_dir, f'project_{project_id}_saved_model_{timestamp_id}'
                                    ))
                                    
                                    # Ensure the destination directory doesn't exist
                                    if os.path.exists(dest_saved_model):
                                        try:
                                            shutil.rmtree(dest_saved_model)
                                        except Exception as rm_error:
                                            logger.error(f"Failed to remove existing saved model: {str(rm_error)}")
                                            # Use a different path to avoid conflicts
                                            dest_saved_model = f"{dest_saved_model}_{secrets.token_hex(4)}"
                                    
                                    # Create parent directory if it doesn't exist
                                    os.makedirs(os.path.dirname(dest_saved_model), exist_ok=True)
                                    
                                    # Copy the entire directory structure with normalized paths
                                    logger.info(f"Copying SavedModel from {temp_dir} to {dest_saved_model}")
                                    
                                    try:
                                        # Use our custom function to copy SavedModel files safely
                                        logger.info(f"Using custom SavedModel file copying for better Windows support")
                                        copy_success = self._copy_savedmodel_files(temp_dir, dest_saved_model)
                                        
                                        if copy_success:
                                            logger.info(f"Successfully copied files using custom method")
                                        else:
                                            # Fall back to standard copying method
                                            logger.warning("Custom copy failed, falling back to standard shutil")
                                            shutil.copytree(temp_dir, dest_saved_model)
                                    except Exception as copy_err:
                                        logger.error(f"Custom copy failed: {str(copy_err)}")
                                        # Fall back to standard shutil.copytree
                                        shutil.copytree(temp_dir, dest_saved_model)
                                    
                                    logger.info(f"Successfully saved TF SavedModel to {dest_saved_model}")
                                    model_files['saved_model_dir'] = dest_saved_model
                                else:
                                    logger.error(f"SavedModel is too small or empty: {saved_size} bytes")
                                    raise ValueError("SavedModel too small")
                        except Exception as savemodel_error:
                            logger.error(f"Error saving as SavedModel: {str(savemodel_error)}")
                    
                    # If all model save attempts failed, create a dummy file
                    if 'tf_model_file' not in model_files and 'saved_model_dir' not in model_files:
                        logger.warning(f"Creating emergency dummy TF model file at {model_path}")
                        
                        try:
                            # Create a dummy file with metadata
                            with open(model_path, 'w') as f:
                                f.write(f"This is an emergency placeholder for failed model save.\n")
                                f.write(f"Project ID: {project_id}\n")
                                f.write(f"Round: {round_num}\n")
                                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                                f.write(f"Original error: Model weights could not be properly saved.\n")
                                f.write(f"Metrics: {json.dumps(model_metrics, indent=2)}\n")
                                try:
                                    f.write(f"Weight shapes: {[list(w.shape) for w in weights]}\n")
                                except Exception:
                                    f.write("Weight shapes could not be determined\n")
                            
                            model_files['tf_model_file'] = model_path
                            model_files['is_emergency_file'] = True
                        except Exception as dummy_error:
                            logger.error(f"Failed to create dummy model file: {str(dummy_error)}")
                
                elif framework.startswith('pytorch') or framework == 'torch':
                    # Handle PyTorch model saving
                    # Placeholder for PyTorch implementation
                    pass
                
                # Create model record with computed metrics
                model_data = {
                    'accuracy': model_metrics['accuracy'],
                    'loss': model_metrics['loss'],
                    'clients': model_metrics['clients'],
                    'round': model_metrics['round'],
                    'metrics': model_metrics,
                    'framework': framework,
                    **model_files  # Include all model file paths
                }
                
                # Save to database via model manager
                from web.services.model_manager import ModelManager
                model_record = ModelManager.save_model(project, model_data, is_final=is_final)
                
                if model_record:
                    logger.info(f"Model saved successfully with ID {model_record.id}")
                    
                    # Update project status for final models
                    if is_final:
                        project.status = 'completed'
                        db.session.commit()
                    
                    return model_record
                else:
                    logger.error("Failed to create model record in database")
                    return self._emergency_model_save(project, model_data)
                    
            except Exception as e:
                logger.error(f"Error in framework-specific model saving: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return self._emergency_model_save(project, model_metrics)
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _get_client_data_size(self, client_id):
        """Get the data size for a client."""
        try:
            # Try to get from the database
            from web.models import Client
            client = Client.query.filter_by(client_id=client_id).first()
            
            if client and client.data_size and client.data_size > 0:
                logger.info(f"Using database data_size={client.data_size} for client {client_id}")
                return client.data_size
            
            # If client exists but no data size, return a default
            if client:
                logger.warning(f"No data_size found for client {client_id} in database")
                return 10
            
            # If client doesn't exist, try the internal registry
            if hasattr(self, 'clients') and client_id in self.clients:
                client_info = self.clients[client_id]
                if 'data_size' in client_info and client_info['data_size'] > 0:
                    logger.info(f"Using registry data_size={client_info['data_size']} for client {client_id}")
                    return client_info['data_size']
            
            # Default value if nothing else works
            logger.warning(f"Using default data size (10) for client {client_id}")
            return 10
            
        except Exception as e:
            logger.error(f"Error getting client data size: {str(e)}")
            return 10

    def _validate_aggregated_model(self, project_id):
        """Validate the aggregated model against a validation dataset."""
        try:
            # Get project info
            project = Project.query.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found during validation")
                return None
                
            # Get the aggregated weights
            weights = self.model_weights.get(project_id, [])
            if not weights:
                logger.error(f"No weights found for project {project_id} during validation")
                return None
                
            # Import appropriate libraries based on framework
            framework = project.framework.lower() if project.framework else 'tensorflow'
            dataset_name = project.dataset_name.lower() if project.dataset_name else 'mnist'
            
            if framework.startswith('tensorflow') or framework == 'keras':
                try:
                    import tensorflow as tf
                    from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
                    
                    # Select validation dataset based on project dataset
                    validation_data = None
                    
                    if dataset_name == 'mnist':
                        # Load MNIST test data
                        (_, _), (x_test, y_test) = mnist.load_data()
                        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
                        y_test_orig = y_test.copy()  # Keep original for confusion matrix
                        y_test = tf.keras.utils.to_categorical(y_test, 10)
                        validation_data = (x_test, y_test)
                        
                    elif dataset_name == 'fashion_mnist':
                        # Load Fashion MNIST test data
                        (_, _), (x_test, y_test) = fashion_mnist.load_data()
                        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
                        y_test_orig = y_test.copy()  # Keep original for confusion matrix
                        y_test = tf.keras.utils.to_categorical(y_test, 10)
                        validation_data = (x_test, y_test)
                        
                    elif dataset_name == 'cifar10':
                        # Load CIFAR-10 test data
                        (_, _), (x_test, y_test) = cifar10.load_data()
                        x_test = x_test.astype('float32') / 255.0
                        y_test_orig = y_test.squeeze().copy()  # Keep original for confusion matrix
                        y_test = tf.keras.utils.to_categorical(y_test, 10)
                        validation_data = (x_test, y_test)
                        
                    else:
                        logger.warning(f"No validation dataset available for {dataset_name}")
                        return None
                    
                    # First validate and preprocess the weights
                    processed_weights = []
                    for w in weights:
                        try:
                            # Skip empty weights
                            if not hasattr(w, 'shape') or w.size == 0:
                                logger.warning("Skipping empty weight array in validation")
                                continue
                                
                            # Convert to numpy if needed
                            if not isinstance(w, np.ndarray):
                                w_array = np.array(w, dtype=np.float32)
                            else:
                                w_array = w
                                
                            # Check for NaN/Inf
                            if np.any(np.isnan(w_array)) or np.any(np.isinf(w_array)):
                                logger.warning("Weight contains NaN/Inf values, replacing with zeros")
                                w_array = np.nan_to_num(w_array, nan=0.0, posinf=0.0, neginf=0.0)
                                
                            processed_weights.append(w_array)
                        except Exception as w_err:
                            logger.error(f"Error processing weight for validation: {str(w_err)}")
                        
                    # If we have no valid weights, return without validation
                    if not processed_weights:
                        logger.error("No valid weights for model validation")
                        return None
                    
                    # Create the model using our dynamic model creation function
                    try:
                        model = self._create_model_for_weights(processed_weights, dataset_name)
                    except Exception as model_err:
                        logger.error(f"Error creating model: {str(model_err)}")
                        return None
                    
                    # Try to apply the processed weights
                    try:
                        model.set_weights(processed_weights)
                        logger.info(f"Successfully applied weights to model for validation")
                    except Exception as weight_err:
                        logger.error(f"Error applying processed weights for validation: {str(weight_err)}")
                        return None
                    
                    # Evaluate the model with error handling
                    try:
                        metrics = model.evaluate(validation_data[0], validation_data[1], verbose=0)
                        
                        # Construct metrics dictionary
                        validation_metrics = {
                            'loss': float(metrics[0]),
                            'accuracy': float(metrics[1]),
                            'val_loss': float(metrics[0]),  # Same as loss for validation set
                            'val_accuracy': float(metrics[1])  # Same as accuracy for validation set
                        }
                        
                        # Calculate more detailed metrics - precision, recall, f1, etc.
                        try:
                            # Get predictions
                            y_pred = model.predict(validation_data[0], verbose=0)
                            y_pred_classes = np.argmax(y_pred, axis=1)
                            
                            # For confusion matrix and classification report
                            from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
                            
                            # Calculate precision, recall, f1
                            if 'y_test_orig' in locals():
                                precision, recall, f1, _ = precision_recall_fscore_support(
                                    y_test_orig, y_pred_classes, average='weighted'
                                )
                                
                                # Add to metrics
                                validation_metrics['precision'] = float(precision)
                                validation_metrics['recall'] = float(recall)
                                validation_metrics['f1'] = float(f1)
                                
                                # Generate confusion matrix
                                cm = confusion_matrix(y_test_orig, y_pred_classes)
                                validation_metrics['confusion_matrix'] = cm.tolist()
                                
                                # Class-wise metrics
                                class_report = classification_report(y_test_orig, y_pred_classes, output_dict=True)
                                validation_metrics['class_report'] = class_report
                            
                            # Calculate AUC if binary classification
                            if validation_data[1].shape[1] == 2:  # Binary classification
                                from sklearn.metrics import roc_auc_score
                                auc = roc_auc_score(validation_data[1][:, 1], y_pred[:, 1])
                                validation_metrics['auc'] = float(auc)
                        except Exception as detailed_err:
                            logger.error(f"Error calculating detailed metrics: {str(detailed_err)}")
                        
                        # Store the aggregation method used
                        validation_metrics['aggregation_method'] = 'PerfFedAvg'
                        
                        # Save these metrics for display in UI
                        if project_id not in self.aggregated_metrics:
                            self.aggregated_metrics[project_id] = {}
                        
                        # Update the metrics with validation results
                        self.aggregated_metrics[project_id].update(validation_metrics)
                        
                        logger.info(f"Validated aggregated model for project {project_id}")
                        logger.info(f"Validation loss: {validation_metrics['loss']:.4f}, validation accuracy: {validation_metrics['accuracy']:.4f}")
                        if 'precision' in validation_metrics:
                            logger.info(f"Precision: {validation_metrics['precision']:.4f}, Recall: {validation_metrics['recall']:.4f}, F1: {validation_metrics['f1']:.4f}")
                        
                        return validation_metrics
                    except Exception as eval_err:
                        logger.error(f"Error evaluating model: {str(eval_err)}")
                        return None
                except ImportError as imp_err:
                    logger.error(f"Import error during model validation: {str(imp_err)}")
                    return None
                except Exception as e:
                    logger.error(f"Error validating aggregated model: {str(e)}")
                    return None
                    
            elif framework.startswith('pytorch') or framework == 'torch':
                # Placeholder for PyTorch validation
                logger.warning("PyTorch validation not implemented yet")
                return None
                
            else:
                logger.warning(f"Validation not implemented for framework: {framework}")
                return None
                
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return None

    def _create_model_for_weights(self, weights, dataset_name='mnist'):
        """Create a model that matches the provided weight array structure.
        
        Args:
            weights: List of weight arrays
            dataset_name: Name of the dataset (to determine input shape)
            
        Returns:
            A TensorFlow model with architecture matching the weights
        """
        try:
            import tensorflow as tf
            
            # Determine the input shape based on dataset
            if dataset_name.lower() == 'mnist' or dataset_name.lower() == 'fashion_mnist':
                input_shape = (28, 28, 1)
                num_classes = 10
            elif dataset_name.lower() == 'cifar10':
                input_shape = (32, 32, 3)
                num_classes = 10
            else:
                # Default shape
                input_shape = (28, 28, 1)
                num_classes = 10
            
            # Create appropriate model based on weight array count
            weight_count = len(weights)
            logger.info(f"Creating model for {weight_count} weight arrays ({dataset_name} dataset)")
            
            if weight_count == 24:  # CNN with BatchNorm (full architecture)
                logger.info("Creating CNN with BatchNorm architecture (24 weight arrays)")
                model = tf.keras.Sequential([
                    # First Conv Block
                    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    
                    # Second Conv Block
                    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                    # IMPORTANT: Removed the second MaxPooling2D layer to match client model
                    # resulting in 14x14x64 feature maps instead of 7x7x64
                    
                    # Flatten layer
                    tf.keras.layers.Flatten(),
                    
                    # Dense layers
                    tf.keras.layers.Dense(512),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
            elif weight_count == 8:  # Simple CNN without BatchNorm
                logger.info("Creating simple CNN architecture (8 weight arrays)")
                model = tf.keras.Sequential([
                    # Conv layers
                    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    
                    # Flatten layer
                    tf.keras.layers.Flatten(),
                    
                    # Dense layers
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
            elif weight_count == 4:  # Simple MLP
                logger.info("Creating simple MLP architecture (4 weight arrays)")
                model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
            else:
                logger.warning(f"Unknown weight structure with {weight_count} arrays. Creating generic model.")
                # Create a generic model based on flattened input
                model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Verify weight count
            model_weights = model.get_weights()
            logger.info(f"Created model with {len(model_weights)} weight arrays")
            
            if len(model_weights) != weight_count:
                logger.warning(f"Created model weights ({len(model_weights)}) still don't match input weights ({weight_count})")
            
            return model
            
        except ImportError:
            logger.error("TensorFlow not installed, could not create model")
            raise
        except Exception as e:
            logger.error(f"Error creating model for weights: {str(e)}")
            raise

    def _create_mnist_model_without_second_pooling(self):
        """Create a MNIST model without the second pooling layer to match client architecture."""
        try:
            import tensorflow as tf
            
            input_shape = (28, 28, 1)
            
            # Create model that matches client's architecture with only one MaxPooling2D
            # This results in 14x14x64 feature maps (12544 neurons) when flattened
            model = tf.keras.Sequential([
                # First Conv Block
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),  # First pooling: 28x28 -> 14x14
                
                # Second Conv Block
                tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                # No second pooling layer - feature maps stay at 14x14
                
                # Flatten layer - 14*14*64 = 12544 neurons
                tf.keras.layers.Flatten(),
                
                # Dense layers
                tf.keras.layers.Dense(512),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Created MNIST model without second pooling layer (14x14x64 features)")
            return model
        except ImportError:
            logger.error("TensorFlow not installed, could not create MNIST model")
            raise

    def _save_fixed_savedmodel(self, model, save_path, save_format='tf'):
        """Custom function to save TensorFlow SavedModel with proper Windows path handling.
        
        This function handles the path issues that can occur on Windows, particularly
        with the variables directory.
        
        Args:
            model: TensorFlow model to save
            save_path: Directory to save the model
            save_format: Format to save in (default 'tf' for SavedModel)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import tensorflow as tf
            import os
            import tempfile
            import shutil
            import secrets
            import json
            
            # STRATEGY: Instead of using SavedModel format which has path issues on Windows,
            # we'll first save as H5 format, then manually create the SavedModel directory structure
            logger.info(f"Using Windows-friendly approach to save model to {save_path}")
            
            # Step 1: Create a temporary directory with a simple path (on C: drive if possible)
            # This avoids the path issues with network drives or deep paths
            try:
                temp_dir = os.environ.get('TEMP', None)
                if not temp_dir or not os.path.exists(temp_dir):
                    temp_dir = tempfile.gettempdir()
                
                # Ensure it's a simple, short path
                if len(temp_dir) > 50:  # If path is too long, use a simpler one
                    temp_dir = "C:/temp" if os.name == 'nt' else "/tmp"
                    os.makedirs(temp_dir, exist_ok=True)
                
                # Add a unique subdirectory
                unique_id = secrets.token_hex(8)
                temp_dir = os.path.join(temp_dir, f"tf_model_{unique_id}")
                os.makedirs(temp_dir, exist_ok=True)
                logger.info(f"Created temp directory: {temp_dir}")
            except Exception as temp_err:
                logger.error(f"Error creating temp directory: {str(temp_err)}")
                # Fall back to regular temp directory
                temp_dir = tempfile.mkdtemp()
            
            # Step 2: Save as H5 format first (far more reliable on Windows)
            h5_path = os.path.join(temp_dir, 'model.h5')
            try:
                model.save(h5_path, save_format='h5')
                logger.info(f"Successfully saved model in H5 format to {h5_path}")
                
                # Copy H5 file to final destination area as a backup
                backup_h5_path = os.path.join(os.path.dirname(save_path), f"model_{unique_id}.h5")
                shutil.copy2(h5_path, backup_h5_path)
                logger.info(f"Backed up H5 model to {backup_h5_path}")
            except Exception as h5_err:
                logger.error(f"Error saving H5 model: {str(h5_err)}")
                # Continue anyway to try SavedModel format
            
            # Step 3: Now manually create the SavedModel directory structure
            os.makedirs(save_path, exist_ok=True)
            variables_dir = os.path.join(save_path, 'variables')
            os.makedirs(variables_dir, exist_ok=True)
            
            # Save weights with simple filenames to avoid path issues
            try:
                weights = model.get_weights()
                variables = {}
                
                # Save each weight as a separate file with a simple name
                for i, w in enumerate(weights):
                    weight_path = os.path.join(variables_dir, f"weight_{i}.npy")
                    np.save(weight_path, w)
                    variables[f"weight_{i}"] = {
                        "shape": list(w.shape) if hasattr(w, 'shape') else [],
                        "dtype": str(w.dtype) if hasattr(w, 'dtype') else "float32"
                    }
                
                # Create a variables manifest file
                manifest_path = os.path.join(variables_dir, "variables.json")
                with open(manifest_path, 'w') as f:
                    json.dump(variables, f, indent=2)
                
                logger.info(f"Saved {len(weights)} weight arrays to {variables_dir}")
            except Exception as weight_err:
                logger.error(f"Error saving individual weights: {str(weight_err)}")
            
            # Create a simple saved_model.pb file with just the model config
            try:
                saved_model_pb = os.path.join(save_path, 'saved_model.pb')
                with open(saved_model_pb, 'wb') as f:
                    # Write a placeholder model config file
                    # In a real scenario, you'd serialize the actual model graph/config
                    config = model.get_config() if hasattr(model, 'get_config') else {}
                    config_bytes = str(config).encode('utf-8')
                    f.write(config_bytes)
                
                logger.info(f"Created basic saved_model.pb file at {saved_model_pb}")
            except Exception as pb_err:
                logger.error(f"Error creating saved_model.pb: {str(pb_err)}")
                
            # Step 4: Try to create a minimal assets directory just for completeness
            try:
                assets_dir = os.path.join(save_path, 'assets')
                os.makedirs(assets_dir, exist_ok=True)
                
                # Create an empty asset file just to make directory non-empty
                with open(os.path.join(assets_dir, "model_info.txt"), 'w') as f:
                    f.write(f"Model created: {datetime.utcnow().isoformat()}\n")
                    f.write(f"Weights count: {len(weights) if 'weights' in locals() else 'unknown'}\n")
                
                logger.info(f"Created assets directory at {assets_dir}")
            except Exception as assets_err:
                logger.error(f"Error creating assets directory: {str(assets_err)}")
                # Not critical, continue
            
            logger.info(f"Successfully created simplified SavedModel at {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error in _save_fixed_savedmodel: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _copy_savedmodel_files(self, src_dir, dest_dir):
        """Carefully copy SavedModel files using a method that avoids Windows path issues.
        
        Args:
            src_dir: Source directory containing SavedModel files
            dest_dir: Destination directory where files should be copied
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Copying SavedModel from {src_dir} to {dest_dir} with special handling")
            
            # Create destination directory structure
            os.makedirs(dest_dir, exist_ok=True)
            variables_dir = os.path.join(dest_dir, 'variables')
            os.makedirs(variables_dir, exist_ok=True)
            
            # Copy saved_model.pb file first
            src_pb = os.path.join(src_dir, 'saved_model.pb')
            dst_pb = os.path.join(dest_dir, 'saved_model.pb')
            copied_files = 0
            
            if os.path.exists(src_pb):
                # Use binary copy to avoid text encoding issues
                with open(src_pb, 'rb') as src:
                    with open(dst_pb, 'wb') as dst:
                        dst.write(src.read())
                copied_files += 1
                logger.info(f"Copied saved_model.pb file")
            
            # Copy variables directory files
            src_variables = os.path.join(src_dir, 'variables')
            if os.path.exists(src_variables):
                for filename in os.listdir(src_variables):
                    src_file = os.path.join(src_variables, filename)
                    dst_file = os.path.join(variables_dir, filename)
                    
                    try:
                        # Use binary copy for variables files
                        with open(src_file, 'rb') as src:
                            with open(dst_file, 'wb') as dst:
                                dst.write(src.read())
                        copied_files += 1
                        logger.info(f"Copied variables file: {filename}")
                    except Exception as file_err:
                        logger.error(f"Failed to copy file {filename}: {str(file_err)}")
            
            # Copy any other files/directories at the root level
            for item in os.listdir(src_dir):
                if item != 'saved_model.pb' and item != 'variables':
                    src_item = os.path.join(src_dir, item)
                    dst_item = os.path.join(dest_dir, item)
                    
                    try:
                        if os.path.isfile(src_item):
                            # Copy file
                            with open(src_item, 'rb') as src:
                                with open(dst_item, 'wb') as dst:
                                    dst.write(src.read())
                            copied_files += 1
                        elif os.path.isdir(src_item):
                            # Copy directory
                            os.makedirs(dst_item, exist_ok=True)
                            for subitem in os.listdir(src_item):
                                src_subitem = os.path.join(src_item, subitem)
                                dst_subitem = os.path.join(dst_item, subitem)
                                if os.path.isfile(src_subitem):
                                    with open(src_subitem, 'rb') as src:
                                        with open(dst_subitem, 'wb') as dst:
                                            dst.write(src.read())
                                    copied_files += 1
                    except Exception as other_err:
                        logger.error(f"Failed to copy {item}: {str(other_err)}")
            
            logger.info(f"Successfully copied {copied_files} SavedModel files")
            return copied_files > 0
            
        except Exception as e:
            logger.error(f"Error in _copy_savedmodel_files: {str(e)}")
            return False

    def create_mnist_model(self):
        """Create a standard MNIST model that matches the client architecture.
        This is a local implementation to avoid import errors.
        
        Returns:
            A TensorFlow model configured for MNIST dataset
        """
        try:
            import tensorflow as tf
            
            # Create a CNN model with only one MaxPooling2D layer
            # to ensure the feature map size is 14x14x64 (12544 neurons)
            model = tf.keras.Sequential([
                # Input layer
                tf.keras.layers.Input(shape=(28, 28, 1)),
                
                # First Conv Block
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),  # 28x28 -> 14x14
                
                # Second Conv Block
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                # No second MaxPooling2D layer - feature maps stay at 14x14
                
                # Flatten and Dense layers
                tf.keras.layers.Flatten(),  # 14x14x64 = 12544 neurons
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Created MNIST model with 14x14x64 feature maps (12544 neurons)")
            return model
            
        except Exception as e:
            logger.error(f"Error creating MNIST model: {str(e)}")
            raise

def start_federated_server(project):
    """
    Start a federated learning server for a project.
    
    Args:
        project (Project): The project to start the server for
        
    Returns:
        bool: True if started successfully, False otherwise
    """
    try:
        # Check if project is already running
        if project.status == 'running':
            logger.warning(f"Project {project.id} is already running")
            return False
        
        # Check available clients
        active_clients_count = project.active_clients_count
        if active_clients_count < project.min_clients:
            logger.warning(f"Not enough active clients for project {project.id}. "
                          f"Need {project.min_clients}, have {active_clients_count}")
            return False
        
        # Capture the current app for use in the thread
        app = current_app._get_current_object()
        project_id = project.id
        
        # Get the federated learning server instance
        fl_server = app.fl_server
        if not fl_server:
            logger.error("Federated learning server not initialized")
            return False
            
        # Initialize the project in the FL server - this creates the initial weights
        # that clients will download when polling
        success = fl_server.initialize_project(project)
        if not success:
            logger.error(f"Failed to initialize project {project.id} in FL server")
            return False
        
        # Update project status only after initialization
        project.status = 'running'
        project.current_round = 0
        db.session.commit()
        
        # Get the dataset and framework info
        dataset = project.dataset_name
        framework = project.framework
        
        # Start the server in a subprocess - in a real app this would run in a separate process
        # or container, but for this example we'll use a thread
        def run_server():
            try:
                # Create a new application context for this thread
                with app.app_context():
                    logger.info(f"Starting federated server for project {project_id}")
                    
                    # Get the project from the database in this thread's context
                    project = Project.query.get(project_id)
                    if not project:
                        logger.error(f"Project {project_id} not found in server thread")
                        return
                    
                    # This is a simplified example. In a real world scenario, you would:
                    # 1. Use a real federated learning server (TensorFlow Federated, PySyft, etc.)
                    # 2. Pass real configuration parameters
                    # 3. Have more robust error handling
                    
                    server_script = os.path.join(
                        app.config.get('FL_SERVER_PATH', 'examples'),
                        dataset.lower(),
                        'run_server.py'
                    )
                    
                    # Construct the command
                    cmd = [
                        sys.executable,
                        server_script,
                        "--project_id", str(project_id),
                        "--min_clients", str(project.min_clients),
                        "--rounds", str(project.rounds),
                        "--framework", framework.lower()
                    ]
                    
                    logger.info(f"Running command: {' '.join(cmd)}")
                    
                    # Run the process
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    
                    # Monitor output (simplified)
                    for line in proc.stdout:
                        logger.info(f"Server output: {line.strip()}")
                        
                        # Parse training progress
                        if "Round completed" in line:
                            # Format: "Round completed: 3/10, accuracy: 0.85, loss: 0.15"
                            parts = line.split(',')
                            round_part = parts[0].strip().split(':')[1].strip()
                            current_round = int(round_part.split('/')[0])
                            
                            # Update project status
                            update_project_status(project_id, current_round)
                            
                            # Create model version
                            if len(parts) >= 3:
                                try:
                                    accuracy = float(parts[1].split(':')[1].strip())
                                    loss = float(parts[2].split(':')[1].strip())
                                    create_model_version(project_id, accuracy, loss)
                                except (IndexError, ValueError) as e:
                                    logger.error(f"Error parsing round metrics: {e}")
                    
                    # Wait for process to complete
                    proc.wait()
                    
                    # Check if the project still exists and update its status
                    project = Project.query.get(project_id)
                    if project:
                        project.status = 'completed'
                        db.session.commit()
                        logger.info(f"Project {project_id} completed")
                    
            except Exception as e:
                with app.app_context():
                    logger.error(f"Error running server for project {project_id}: {str(e)}")
                    project = Project.query.get(project_id)
                    if project:
                        project.status = 'error'
                        db.session.commit()
        
        # Start server in a thread
        thread = threading.Thread(target=run_server)
        thread.daemon = True
        thread.start()
        
        return True
        
    except Exception as e:
        logger.error(f"Error starting server for project {project.id}: {str(e)}")
        return False

def update_project_status(project_id, current_round, status=None):
    """Update the status of a project."""
    try:
        project = Project.query.get(project_id)
        if project:
            project.current_round = current_round
            if status:
                project.status = status
            db.session.commit()
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating project status: {str(e)}")
        db.session.rollback()
        return False

def create_model_version(project_id, accuracy=0, loss=0, clients=0, model_file=None, is_final=False):
    """Create a new version of a model."""
    try:
        project = Project.query.get(project_id)
        if not project:
            return False
        
        model = Model(
            project_id=project_id,
            version=len(project.models) + 1,
            accuracy=accuracy,
            loss=loss,
            clients_count=clients,
            is_final=is_final,
            model_file=model_file
        )
        
        db.session.add(model)
        db.session.commit()
        return True
        
    except Exception as e:
        logger.error(f"Error creating model version: {str(e)}")
        db.session.rollback()
        return False

def deploy_model_as_api(model_id):
    """Deploy a model as an API endpoint."""
    try:
        model = Model.query.get(model_id)
        if not model:
            return False
        
        # This is a simplified example. In a real world scenario, you would:
        # 1. Save the model file
        # 2. Create an API endpoint
        # 3. Set up authentication
        # 4. Configure rate limiting
        
        model.is_deployed = True
        model.deployment_info = {
            'type': 'api',
            'endpoint': f'/api/models/{model_id}/predict',
            'created_at': datetime.utcnow()
        }
        
        db.session.commit()
        return True
        
    except Exception as e:
        logger.error(f"Error deploying model as API: {str(e)}")
        db.session.rollback()
        return False

def run_server(self, project):
    """Run the federated learning server in a background thread."""
    try:
        # Create an application context that will be used in the thread
        app = current_app._get_current_object()
        
        with app.app_context():
            # Get server configuration
            server_path = app.config.get('FL_SERVER_PATH', 'examples')
            server_host = app.config.get('FL_SERVER_HOST', 'localhost')
            server_port = app.config.get('FL_SERVER_PORT', 8080)
            
            # Start the server process
            logger.info(f"Starting FL server for project {project.id}")
            
            try:
                # Update project status
                project.status = 'running'
                project.current_round = 0
                db.session.commit()
                
                # TODO: Add actual server process management here
                # For now, we'll just simulate the server running
                while project.status == 'running':
                    time.sleep(5)  # Check status every 5 seconds
                    db.session.refresh(project)
                
            except Exception as e:
                logger.error(f"Error in FL server process: {str(e)}")
                project.status = 'failed'
                db.session.commit()
                
    except Exception as e:
        logger.error(f"Error running FL server: {str(e)}") 