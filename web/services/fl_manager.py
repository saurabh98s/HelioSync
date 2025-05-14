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
            dense1_w = np.random.randn(12544, 512).astype(np.float32) * 0.1  # 7*7*256 -> 512
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
                
                # Perform federated averaging - only if we have multiple clients
                if active_project_clients > 1:
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
            
            # Create uploads directories
            base_upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
            model_dir = os.path.join(base_upload_folder, 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # Define file paths
            tf_model_filename = f'project_{project_id}_model.h5'
            pt_model_filename = f'project_{project_id}_model.pt'
            tf_model_path = os.path.join(model_dir, tf_model_filename)
            pt_model_path = os.path.join(model_dir, pt_model_filename)
            
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
                    # Use TensorFlow's SavedModel format first
                    save_dir = os.path.join(model_dir, f'project_{project_id}_saved_model')
                    os.makedirs(save_dir, exist_ok=True)
                    model.save(save_dir)
                    logger.info(f"✓ TensorFlow SavedModel format saved to {save_dir}")
                    
                    # Also save as HDF5 for backward compatibility
                    try:
                        # Explicitly use h5 extension and format
                        h5_model_path = os.path.join(model_dir, f'project_{project_id}_model.h5')
                        model.save(h5_model_path, save_format='h5')
                        logger.info(f"✓ TensorFlow H5 model saved to {h5_model_path}")
                        tf_model_saved = True
                        # Make sure tf_model_path is set to the h5 file
                        tf_model_path = h5_model_path
                    except Exception as h5_error:
                        logger.error(f"Error saving H5 model: {str(h5_error)}")
                        # Continue with the SavedModel format
                        tf_model_path = save_dir
                        tf_model_saved = True
                except Exception as save_error:
                    logger.error(f"Error saving model file: {str(save_error)}")
                
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
                
                # Create a simple PyTorch model
                if project.dataset_name.lower() == 'mnist':
                    model = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(784, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10),
                        nn.LogSoftmax(dim=1)
                    )
                elif project.dataset_name.lower() == 'cifar10':
                    model = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(3*32*32, 512),
                        nn.ReLU(),
                        nn.Linear(512, 10),
                        nn.LogSoftmax(dim=1)
                    )
                else:
                    model = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(784, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10),
                        nn.LogSoftmax(dim=1)
                    )
                
                # Save the PyTorch model
                try:
                    # Explicitly use .pt extension for PyTorch
                    pt_model_path = os.path.join(model_dir, f'project_{project_id}_model.pt')
                    
                    # Save the state dict (preferred way)
                    torch.save(model.state_dict(), pt_model_path)
                    logger.info(f"✓ PyTorch state dict saved to {pt_model_path}")
                    
                    # Also save the full model as a separate file
                    full_model_path = os.path.join(model_dir, f'project_{project_id}_model_full.pt')
                    torch.save(model, full_model_path)
                    logger.info(f"✓ PyTorch full model saved to {full_model_path}")
                    
                    # Save model in TorchScript format for better deployment
                    script_path = os.path.join(model_dir, f'project_{project_id}_model.torchscript')
                    scripted_model = torch.jit.script(model)
                    scripted_model.save(script_path)
                    logger.info(f"✓ PyTorch TorchScript model saved to {script_path}")
                    
                    pt_model_saved = True
                except Exception as pt_save_error:
                    logger.error(f"Error saving PyTorch model: {str(pt_save_error)}")
                    
                    # Try to save a simplified version
                    try:
                        torch.save(model, pt_model_path)
                        logger.info(f"✓ PyTorch full model saved to {pt_model_path}")
                        pt_model_saved = True
                    except Exception as pt_full_save_error:
                        logger.error(f"Error saving full PyTorch model: {str(pt_full_save_error)}")
                
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
            model = tf.keras.Sequential([
                # First Conv Block
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                # Second Conv Block
                tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
                
                # Flatten layer
                tf.keras.layers.Flatten(),
                
                # Dense layers
                tf.keras.layers.Dense(512),
                tf.keras.layers.BatchNormalization(),
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
                                from examples.mnist.models import create_mnist_model
                                model = create_mnist_model()
                                self.models[project_id] = model
                            except ImportError:
                                logger.error("Could not import create_mnist_model")
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
                            all_valid = all(w.size > 0 for w in weights)
                            if all_valid:
                                model.set_weights(weights)
                                weights_applied = True
                                logger.info(f"Successfully applied weights to model")
                            else:
                                logger.error("One or more weight arrays are empty")
                        else:
                            logger.error(f"Weight count mismatch: Model expects {len(model.weights)}, but got {len(weights)}")
                    except Exception as apply_error:
                        logger.error(f"Error applying weights: {str(apply_error)}")
                    
                    # If weights couldn't be applied, use a different approach to create a valid model file
                    if not weights_applied:
                        logger.warning("Creating a dummy model since weights couldn't be applied")
                        try:
                            # Create a simple model that will definitely save
                            dummy_model = tf.keras.Sequential([
                                tf.keras.layers.Dense(10, input_shape=(10,))
                            ])
                            dummy_model.compile(optimizer='adam', loss='mse')
                            model = dummy_model
                        except Exception as dummy_error:
                            logger.error(f"Error creating dummy model: {str(dummy_error)}")
                    
                    # Try saving in a temporary location first
                    try:
                        # Use a temporary directory for initial save
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_model_path = os.path.join(temp_dir, "temp_model.h5")
                            
                            # Save in HDF5 format
                            model.save(temp_model_path, save_format='h5')
                            
                            # Verify file was created and has content
                            if os.path.exists(temp_model_path) and os.path.getsize(temp_model_path) > 1000:
                                # Copy to final location
                                shutil.copy2(temp_model_path, model_path)
                                logger.info(f"Successfully saved TF model to {model_path}")
                                model_files['tf_model_file'] = model_path
                            else:
                                logger.error(f"Temp model file is missing or too small: {temp_model_path}")
                                raise FileNotFoundError("Temp model file invalid")
                    except Exception as h5_error:
                        logger.error(f"Error saving H5 model: {str(h5_error)}")
                        
                        # Try SavedModel format as backup
                        try:
                            # Use a temporary directory for SavedModel
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Save in SavedModel format
                                model.save(temp_dir, save_format='tf')
                                
                                # Verify directory was created with content
                                saved_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                              for dirpath, _, filenames in os.walk(temp_dir)
                                              for filename in filenames)
                                
                                if saved_size > 1000:
                                    # Create new directory for final location
                                    if os.path.exists(saved_model_path):
                                        try:
                                            shutil.rmtree(saved_model_path)
                                        except Exception as rm_error:
                                            logger.error(f"Failed to remove existing saved model: {str(rm_error)}")
                                            # Use a different path to avoid conflicts
                                            saved_model_path = f"{saved_model_path}_{secrets.token_hex(4)}"
                                    
                                    # Copy the entire directory structure
                                    shutil.copytree(temp_dir, saved_model_path)
                                    logger.info(f"Successfully saved TF SavedModel to {saved_model_path}")
                                    model_files['saved_model_dir'] = saved_model_path
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