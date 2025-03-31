"""
Model Manager Service

This module provides services for managing trained federated learning models,
including saving, loading, and deploying models.
"""

import os
import json
import shutil
import logging
from datetime import datetime
import subprocess
from flask import current_app

from web.app import db
from web.models import Model, Project

logger = logging.getLogger(__name__)

class ModelManager:
    """Service for managing machine learning models."""
    
    @staticmethod
    def save_model(project, model_data, is_final=False):
        """
        Save a model to disk and database.
        
        Args:
            project (Project): The project the model belongs to
            model_data (dict): Dictionary containing model information
            is_final (bool): Whether this is the final model of the project
            
        Returns:
            Model: The created model object
        """
        # Get the version number (latest + 1)
        version = 1
        existing_models = Model.query.filter_by(project_id=project.id).all()
        if existing_models:
            version = max(model.version for model in existing_models) + 1
        
        # Create the model directory if it doesn't exist
        model_base_path = os.path.join(current_app.config['FL_MODEL_PATH'], f'project_{project.id}')
        os.makedirs(model_base_path, exist_ok=True)
        
        # Path to save the model
        model_path = os.path.join(model_base_path, f'model_v{version}')
        
        # Get metrics from model_data
        metrics = {
            'accuracy': model_data.get('accuracy', 0),
            'loss': model_data.get('loss', 0),
            'round': project.current_round,
            'clients': model_data.get('clients', 0)
        }
        
        # Save the model file
        model_file = model_data.get('model_file')
        if model_file and os.path.exists(model_file):
            if os.path.isfile(model_file):
                shutil.copy2(model_file, model_path)
            else:
                shutil.copytree(model_file, model_path)
        
        # Create model object
        model = Model(
            project_id=project.id,
            version=version,
            path=model_path,
            metrics=metrics,
            created_at=datetime.utcnow(),
            is_final=is_final,
            clients_count=model_data.get('clients', 0)
        )
        
        db.session.add(model)
        db.session.commit()
        logger.info(f"Saved model version {version} for project {project.name}")
        
        return model
    
    @staticmethod
    def deploy_model(model, deploy_type='api'):
        """
        Deploy a model for inference.
        
        Args:
            model (Model): The model to deploy
            deploy_type (str): Type of deployment (api or download)
            
        Returns:
            dict: Deployment information
        """
        if not model.path or not os.path.exists(model.path):
            logger.error(f"Model path does not exist: {model.path}")
            return {"success": False, "error": "Model file not found"}
        
        deployment_info = {
            "type": deploy_type,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "deployed"
        }
        
        if deploy_type == 'api':
            # In a real application, you would start an API server here
            # For demonstration, we'll just update the model status
            deployment_info["endpoint"] = f"/api/models/{model.id}/predict"
            deployment_info["port"] = 5001
            
            # Sample code to start a model serving process
            # This would be replaced with actual code in production
            try:
                # Start a subprocess to run the model server
                # cmd = [
                #     "python", 
                #     "serve_model.py",
                #     "--model_path", model.path,
                #     "--port", "5001"
                # ]
                # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Simulate a process
                logger.info(f"Started API server for model {model.id}")
                deployment_info["process_id"] = 12345  # This would be process.pid in a real app
            except Exception as e:
                logger.error(f"Failed to start API server: {str(e)}")
                return {"success": False, "error": str(e)}
        
        elif deploy_type == 'download':
            # Create a downloadable version of the model
            download_dir = os.path.join(current_app.config['FL_MODEL_PATH'], 'downloads')
            os.makedirs(download_dir, exist_ok=True)
            
            project = Project.query.get(model.project_id)
            download_filename = f"{project.name.replace(' ', '_')}_model_v{model.version}"
            
            # Package the model for download
            download_path = os.path.join(download_dir, download_filename)
            
            try:
                if os.path.isfile(model.path):
                    shutil.copy2(model.path, download_path)
                else:
                    # Create a zip file for directory
                    shutil.make_archive(download_path, 'zip', model.path)
                    download_path += '.zip'
                
                deployment_info["download_path"] = download_path
                deployment_info["download_url"] = f"/models/download/{model.id}"
            except Exception as e:
                logger.error(f"Failed to prepare model for download: {str(e)}")
                return {"success": False, "error": str(e)}
        
        # Update the model in the database
        model.is_deployed = True
        model.deployment_info = deployment_info
        db.session.commit()
        
        return {"success": True, "deployment_info": deployment_info}
    
    @staticmethod
    def get_model_metrics(project_id):
        """
        Get metrics for all models of a project.
        
        Args:
            project_id (int): The project ID
            
        Returns:
            dict: Dictionary with metrics data for visualization
        """
        models = Model.query.filter_by(project_id=project_id).order_by(Model.version).all()
        
        if not models:
            return {
                "versions": [],
                "accuracy": [],
                "loss": [],
                "clients": []
            }
        
        metrics = {
            "versions": [],
            "accuracy": [],
            "loss": [],
            "clients": []
        }
        
        for model in models:
            metrics["versions"].append(f"v{model.version}")
            
            if model.metrics:
                metrics["accuracy"].append(model.metrics.get("accuracy", 0))
                metrics["loss"].append(model.metrics.get("loss", 0))
                metrics["clients"].append(model.clients_count)
            else:
                metrics["accuracy"].append(0)
                metrics["loss"].append(0)
                metrics["clients"].append(0)
        
        return metrics 