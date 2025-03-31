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

from web.app import db
from web.models import Project, Client, Model
from web.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

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
                        
                    # Check if training is completed
                    elif "Training completed" in line:
                        # Format: "Training completed: accuracy: 0.92, loss: 0.08"
                        parts = line.split(',')
                        accuracy = float(parts[0].split(':')[2].strip())
                        loss = float(parts[1].split(':')[2].strip())
                        
                        # Create final model
                        create_model_version(
                            project.id, 
                            accuracy=accuracy, 
                            loss=loss, 
                            is_final=True
                        )
                        
                        # Mark project as completed
                        update_project_status(project.id, project.rounds, status='completed')
                
                # Process completed - check exit code
                proc.wait()
                if proc.returncode != 0:
                    logger.error(f"Server process failed with code {proc.returncode}")
                    for line in proc.stderr:
                        logger.error(f"Server error: {line.strip()}")
                    
                    # Mark project as failed
                    update_project_status(project.id, project.current_round, status='failed')
                    return False
                
                return True
                
            except Exception as e:
                logger.exception(f"Error running federated server: {str(e)}")
                update_project_status(project.id, project.current_round, status='failed')
                return False
        
        # Start the server in a background thread
        thread = threading.Thread(target=run_server)
        thread.daemon = True
        thread.start()
        
        return True
        
    except Exception as e:
        logger.exception(f"Error starting federated server: {str(e)}")
        return False

def update_project_status(project_id, current_round, status=None):
    """
    Update the project's status and current round in the database.
    
    Args:
        project_id (int): The project ID
        current_round (int): The current round number
        status (str, optional): The project status if changed
        
    Returns:
        bool: True if updated successfully, False otherwise
    """
    try:
        # Get the project
        project = Project.query.get(project_id)
        if not project:
            logger.error(f"Project {project_id} not found")
            return False
        
        # Update the round
        project.current_round = current_round
        
        # Update the status if provided
        if status:
            project.status = status
        
        db.session.commit()
        logger.info(f"Updated project {project_id} to round {current_round}, status: {project.status}")
        return True
        
    except Exception as e:
        logger.exception(f"Error updating project status: {str(e)}")
        return False

def create_model_version(project_id, accuracy=0, loss=0, clients=0, model_file=None, is_final=False):
    """
    Create a new model version in the database.
    
    Args:
        project_id (int): The project ID
        accuracy (float): The model accuracy
        loss (float): The model loss
        clients (int): Number of clients that participated
        model_file (str): Path to the model file
        is_final (bool): Whether this is the final model
        
    Returns:
        Model: The created model object or None if failed
    """
    try:
        # Get the project
        project = Project.query.get(project_id)
        if not project:
            logger.error(f"Project {project_id} not found")
            return None
        
        # Create a model object with the provided metrics
        model_data = {
            'accuracy': accuracy,
            'loss': loss,
            'clients': clients,
            'model_file': model_file
        }
        
        # Save the model
        model = ModelManager.save_model(project, model_data, is_final)
        
        logger.info(f"Created model version {model.version} for project {project_id}")
        return model
        
    except Exception as e:
        logger.exception(f"Error creating model version: {str(e)}")
        return None

def deploy_model_as_api(model_id):
    """
    Deploy a model as an API.
    
    Args:
        model_id (int): The model ID
        
    Returns:
        dict: The deployment result
    """
    try:
        # Get the model
        model = Model.query.get(model_id)
        if not model:
            logger.error(f"Model {model_id} not found")
            return {"success": False, "error": "Model not found"}
        
        # Deploy the model
        result = ModelManager.deploy_model(model, deploy_type='api')
        return result
        
    except Exception as e:
        logger.exception(f"Error deploying model as API: {str(e)}")
        return {"success": False, "error": str(e)} 