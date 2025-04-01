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

from web.app import db
from web.models import Project, Client, Model, ProjectClient
from web.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

class FederatedLearningServer:
    """Manages the federated learning server and client connections."""
    
    def __init__(self):
        self.clients = {}  # Dictionary to store connected clients
        self.projects = {}  # Dictionary to store active projects
        self.client_metrics = {}  # Dictionary to store client metrics
        self.aggregated_metrics = {}  # Dictionary to store aggregated metrics
        self.client_projects = {}  # Dictionary to store which projects each client is participating in
        self.model_weights = {}  # Dictionary to store model weights for each project
        self.current_round = 0
        self.rounds = 0
        self.min_clients = 0
    
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
            if client_id not in self.clients:
                logger.error(f"Client {client_id} not found")
                return False
            
            if project_id not in self.projects:
                logger.error(f"Project {project_id} not found")
                return False
            
            # Add client to project's client set
            self.client_projects[client_id].add(project_id)
            
            # Create project client association in database
            project_client = ProjectClient(
                project_id=project_id,
                client_id=client_id,
                status='registered',
                joined_at=datetime.utcnow()
            )
            db.session.add(project_client)
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
            
            self.client_metrics[client_id][project_id] = {
                'accuracy': metrics.get('accuracy', 0),
                'loss': metrics.get('loss', 0),
                'epochs': metrics.get('epochs', 0),
                'timestamp': datetime.utcnow()
            }
            
            # Update project client status
            project_client = ProjectClient.query.filter_by(
                project_id=project_id,
                client_id=client_id
            ).first()
            
            if project_client:
                project_client.status = 'completed'
                project_client.accuracy = metrics.get('accuracy', 0)
                project_client.loss = metrics.get('loss', 0)
                project_client.local_epochs = metrics.get('epochs', 0)
                project_client.last_update = datetime.utcnow()
                db.session.commit()
            
            logger.info(f"Updated metrics for client {client_id} in project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metrics for client {client_id}: {str(e)}")
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
                if metrics.get('round', 0) >= project.rounds:
                    project.status = 'completed'
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
                'current_round': 0,
                'total_rounds': project.rounds,
                'min_clients': project.min_clients
            }
            
            # Initialize project-specific attributes
            self.current_round = 0
            self.rounds = project.rounds
            self.min_clients = project.min_clients
            
            # Initialize model based on framework and dataset
            if project.framework.lower() == 'tensorflow':
                self.model_weights[project_id] = self._initialize_tensorflow_model(project.dataset_name)
            else:
                raise ValueError(f"Unsupported framework: {project.framework}")
            
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
        
        # MNIST weights structure (simple CNN)
        if dataset_name.lower() == 'mnist':
            # Conv1 (28x28x1 -> 26x26x32)
            conv1_w = np.random.randn(3, 3, 1, 32).astype(np.float32) * 0.1
            conv1_b = np.zeros(32, dtype=np.float32)
            
            # Conv2 (26x26x32 -> 24x24x64)
            conv2_w = np.random.randn(3, 3, 32, 64).astype(np.float32) * 0.1
            conv2_b = np.zeros(64, dtype=np.float32)
            
            # Dense (9216 -> 128)
            dense1_w = np.random.randn(9216, 128).astype(np.float32) * 0.1
            dense1_b = np.zeros(128, dtype=np.float32)
            
            # Output (128 -> 10)
            output_w = np.random.randn(128, 10).astype(np.float32) * 0.1
            output_b = np.zeros(10, dtype=np.float32)
            
            weights = [conv1_w, conv1_b, conv2_w, conv2_b, dense1_w, dense1_b, output_w, output_b]
            logger.info(f"Initialized mock MNIST model weights")
            
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
            if project_id is None:
                # Update first project if no project_id specified
                if not self.model_weights:
                    raise ValueError("No projects initialized")
                project_id = list(self.model_weights.keys())[0]
            
            if project_id not in self.model_weights:
                raise ValueError(f"Project {project_id} not initialized")
            
            # Update client metrics
            self.update_client_metrics(client_id, project_id, metrics)
            
            # For now, just store the latest weights
            # In a real implementation, you would aggregate weights from multiple clients
            self.model_weights[project_id] = weights
            
            # Update project metrics
            if project_id not in self.aggregated_metrics:
                self.aggregated_metrics[project_id] = {}
            
            self.aggregated_metrics[project_id].update({
                'accuracy': metrics.get('accuracy', 0),
                'loss': metrics.get('loss', 0),
                'round': self.current_round,
                'timestamp': datetime.utcnow()
            })
            
            logger.info(f"Updated model for project {project_id} with weights from client {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            return False

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
        
        # Update project status to running
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
                logger.info(f"Starting federated server for project {project.id}")
                
                # This is a simplified example. In a real world scenario, you would:
                # 1. Use a real federated learning server (TensorFlow Federated, PySyft, etc.)
                # 2. Pass real configuration parameters
                # 3. Have more robust error handling
                
                server_script = os.path.join(
                    current_app.config.get('FL_SERVER_PATH', 'examples'),
                    dataset.lower(),
                    'run_server.py'
                )
                
                # Construct the command
                cmd = [
                    sys.executable,
                    server_script,
                    "--project_id", str(project.id),
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
                        update_project_status(project.id, current_round)
                        
                        # Create model version
                        accuracy = float(parts[1].split(':')[1].strip())
                        loss = float(parts[2].split(':')[1].strip())
                        create_model_version(project.id, accuracy, loss)
                
                # Wait for process to complete
                proc.wait()
                
                # Update project status to completed
                project.status = 'completed'
                db.session.commit()
                
            except Exception as e:
                logger.error(f"Error running server for project {project.id}: {str(e)}")
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