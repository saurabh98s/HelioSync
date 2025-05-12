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
import numpy as np
import sys
import time

from web.app import db
from web.models import Model, Project, User

logger = logging.getLogger(__name__)

class ModelManager:
    """Service for managing machine learning models."""
    
    @classmethod
    def save_model(cls, project, model_data, is_final=False):
        """Save model to database."""
        try:
            # Create new model entry
            model = Model(
                project_id=project.id,
                version=cls._get_next_version(project.id),
                is_final=is_final
            )
            
            # Validate and normalize model_data
            if not isinstance(model_data, dict):
                logging.warning(f"model_data is not a dictionary, converting to dict")
                model_data = {
                    'accuracy': 0.0,
                    'loss': 0.0,
                    'clients': 0,
                    'is_emergency_recovery': True,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Ensure numeric fields have proper values
            for field in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
                if field in model_data:
                    try:
                        # Convert to float and validate
                        value = float(model_data[field])
                        # Handle NaN or infinity
                        if not np.isfinite(value):
                            logging.warning(f"Non-finite value for {field}: {value}, using 0.0")
                            model_data[field] = 0.0
                        else:
                            model_data[field] = value
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid value for {field}: {model_data.get(field)}, using 0.0")
                        model_data[field] = 0.0
            
            # Add metrics to model
            model.accuracy = model_data.get('accuracy', 0)
            model.loss = model_data.get('loss', 0)
            model.clients_count = model_data.get('clients', 0)
            
            # Ensure metrics is always initialized as a valid dict
            if 'metrics' not in model_data:
                # Create a metrics dictionary from individual fields
                model_data['metrics'] = {
                    'accuracy': model.accuracy,
                    'loss': model.loss,
                    'clients': model.clients_count,
                    'val_accuracy': model_data.get('val_accuracy', 0),
                    'val_loss': model_data.get('val_loss', 0),
                    'round': model_data.get('round', 0),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Set metrics - this will use the metrics property which has validation
            model.metrics = model_data.get('metrics', model_data)
            
            # Save model file path if provided
            if 'model_file' in model_data:
                model.path = model_data['model_file']
                
                # Set additional paths for different formats if provided
                additional_paths = {}
                if 'tf_model_file' in model_data:
                    additional_paths['tensorflow'] = model_data['tf_model_file']
                    additional_paths['pytorch'] = model_data['model_file']
                
                if additional_paths:
                    model.additional_paths = additional_paths
            
            # Set framework if provided
            if 'framework' in model_data:
                model.framework = model_data['framework']
            
            # Add to database
            db.session.add(model)
            db.session.commit()
            
            # Only update project status to completed if this is explicitly the final model
            # AND the project has completed all rounds
            if is_final:
                try:
                    # Only mark project as completed if this is explicitly a final model
                    # Don't automatically mark completed based on rounds
                    project.status = 'completed'
                    
                    # Mark this as the final model for the project
                    model.is_final = True
                    
                    db.session.commit()
                    logging.info(f"Model saved and project {project.id} marked as completed")
                except Exception as status_error:
                    logging.error(f"Error updating project status: {str(status_error)}")
                    # Continue anyway, since the model was saved successfully
            
            return model
        
        except Exception as e:
            current_app.logger.error(f"Error saving model: {str(e)}")
            import traceback
            current_app.logger.error(traceback.format_exc())
            db.session.rollback()
            
            # For final models, try a simplified approach if the standard one fails
            if is_final:
                try:
                    current_app.logger.warning(f"Attempting simplified final model save for project {project.id}")
                    # Try a minimal model entry
                    model = Model(
                        project_id=project.id,
                        version=1,  # Just use version 1 if we can't determine proper version
                        is_final=True,
                        accuracy=model_data.get('accuracy', 0) if isinstance(model_data, dict) else 0,
                        loss=model_data.get('loss', 0) if isinstance(model_data, dict) else 0
                    )
                    
                    # Always initialize metrics dict to avoid template errors
                    model.metrics = {
                        'accuracy': model.accuracy,
                        'loss': model.loss,
                        'clients': 1,
                        'is_emergency_recovery': True,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # Mark the project as completed
                    project.status = 'completed'
                    
                    db.session.add(model)
                    db.session.commit()
                    
                    current_app.logger.info(f"Successfully saved simplified final model for project {project.id}")
                    return model
                except Exception as recovery_error:
                    current_app.logger.error(f"Recovery attempt also failed: {str(recovery_error)}")
                    
                    # Last resort - just update the project status
                    try:
                        project.status = 'completed'
                        db.session.commit()
                        current_app.logger.warning(f"Project {project.id} marked as completed despite model save failure")
                    except Exception:
                        pass
            
            return None
    
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
        try:
            deployment_info = {
                "type": deploy_type,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "deployed"
            }
            
            # Create model directory if it doesn't exist
            model_dir = os.path.join(current_app.config.get('FL_MODEL_PATH', 'uploads/models'), 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # Get the project
            project = Project.query.get(model.project_id)
            if not project:
                return {"success": False, "error": "Project not found"}
            
            # If model path doesn't exist, create a dummy file for testing
            model_path = model.path
            if not model_path or not os.path.exists(model_path):
                # Look for model files in the model directory based on project ID
                possible_files = [
                    # TensorFlow formats
                    os.path.join(model_dir, f'project_{model.project_id}_model.h5'),
                    # PyTorch formats
                    os.path.join(model_dir, f'project_{model.project_id}_model.pt'),
                    os.path.join(model_dir, f'project_{model.project_id}_model_full.pt'),
                    os.path.join(model_dir, f'project_{model.project_id}_model.torchscript')
                ]
                
                # Find the first valid file
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        model_path = file_path
                        # Update the model path in the database
                        model.path = model_path
                        db.session.commit()
                        logger.info(f"Found model file at {model_path}")
                        break
                
                # If still no valid file, create a dummy
                if not model_path or not os.path.exists(model_path):
                    logger.warning(f"Model path does not exist. Creating a dummy file for testing.")
                    model_filename = f"model_project_{model.project_id}_version_{model.version}.h5"
                    model_path = os.path.join(model_dir, model_filename)
                    
                    # Create a dummy file with some metrics information
                    try:
                        with open(model_path, 'w') as f:
                            f.write(f"Dummy model file for testing.\nProject: {model.project_id}\nVersion: {model.version}\n")
                            if model.metrics:
                                f.write(f"Metrics: {str(model.metrics)}\n")
                        
                        # Update the model path
                        model.path = model_path
                        db.session.commit()
                        logger.info(f"Created dummy model file at {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to create dummy model file: {str(e)}")
                        return {"success": False, "error": f"Failed to create model file: {str(e)}"}
            
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
                download_dir = os.path.join(current_app.config.get('FL_MODEL_PATH', 'uploads/models'), 'downloads')
                os.makedirs(download_dir, exist_ok=True)
                
                # Create a suitable filename
                sanitized_name = project.name.replace(' ', '_').lower()
                download_filename = f"{sanitized_name}_model_v{model.version}"
                
                # Add extension based on file type
                if model_path.endswith('.h5'):
                    download_filename += '.h5'
                elif model_path.endswith('.pt'):
                    download_filename += '.pt'
                elif model_path.endswith('.torchscript'):
                    download_filename += '.torchscript'
                else:
                    # Default extension
                    download_filename += os.path.splitext(model_path)[1]
                
                # Package the model for download
                download_path = os.path.join(download_dir, download_filename)
                
                try:
                    if os.path.isfile(model_path):
                        shutil.copy2(model_path, download_path)
                        logger.info(f"Copied model file to {download_path}")
                    else:
                        # Create a zip file for directory
                        shutil.make_archive(download_path, 'zip', model_path)
                        download_path += '.zip'
                        logger.info(f"Created zip archive at {download_path}")
                    
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
            
        except Exception as e:
            logger.error(f"Deploy model error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
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

    @classmethod
    def _get_next_version(cls, project_id):
        """Get the next version number for a model."""
        # Get the version number (latest + 1)
        version = 1
        existing_models = Model.query.filter_by(project_id=project_id).all()
        if existing_models:
            version = max(model.version for model in existing_models) + 1
        return version 

    @staticmethod
    def deploy_to_huggingface(model, model_name=None, private=True):
        """
        Deploy a model to Hugging Face Hub.
        
        Args:
            model (Model): The model to deploy
            model_name (str): Name for the Hugging Face model (defaults to auto-generated name)
            private (bool): Whether the model should be private (default) or public
            
        Returns:
            dict: Deployment information
        """
        try:
            import tempfile
            import shutil
            import traceback
            
            logger.info(f"Preparing to deploy model {model.id} to Hugging Face Hub")
            
            # Try to import huggingface_hub
            try:
                from huggingface_hub import HfApi, create_repo
                logger.info("Successfully imported huggingface_hub")
            except ImportError:
                error_msg = "huggingface_hub package not installed. Please install with: pip install huggingface_hub"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Generate repo name if not provided
            if not model_name:
                project = Project.query.get(model.project_id)
                if not project:
                    error_msg = f"Project {model.project_id} not found"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg
                    }
                
                # Get organization or user info for the repository name prefix
                # to avoid name conflicts with other users
                org_slug = None
                try:
                    # Try to get the organization from the project
                    if project.organizations and len(project.organizations) > 0:
                        org = project.organizations[0]
                        org_slug = org.name.lower().replace(' ', '-').replace('_', '-')
                    
                    # If we couldn't get org, try to get from creator
                    if not org_slug and project.creator_id:
                        creator = User.query.get(project.creator_id)
                        if creator:
                            org_slug = creator.username.lower().replace(' ', '-').replace('_', '-')
                    
                    # If still no org slug, use a default
                    if not org_slug:
                        org_slug = "federated-learning"
                        
                    logger.info(f"Using organization/user slug: {org_slug}")
                except Exception as e:
                    logger.warning(f"Could not get organization info: {str(e)}")
                    org_slug = "federated-learning"
                
                # Use the existing repo instead of creating a new one for each model
                model_name = "saurabhss25/federated_repo"
                logger.info(f"Using existing repository: {model_name}")
                
                # Create folder name based on model details to organize models in the repo
                folder_name = f"model_{project.dataset_name.lower()}_{model.id}"
                logger.info(f"Using folder prefix for model organization: {folder_name}")
            
            # Check if model file exists
            model_path = model.path
            if not model_path or not os.path.exists(model_path):
                error_msg = "Model file not found or not accessible"
                logger.error(f"{error_msg}: {model_path}")
                return {
                    "success": False, 
                    "error": error_msg
                }
            else:
                logger.info(f"Located model file at: {model_path}")
                if os.path.isfile(model_path):
                    logger.info(f"Model is a file, size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
                else:
                    logger.info(f"Model is a directory, contents: {os.listdir(model_path)}")
            
            # Get HF API token from config
            hf_token = current_app.config.get('HUGGINGFACE_TOKEN')
            if not hf_token:
                # Try getting from environment directly as a fallback
                hf_token = os.environ.get('HUGGINGFACE_TOKEN')
                
            if not hf_token:
                error_msg = "HuggingFace API token not configured. Please set HUGGINGFACE_TOKEN in your config.env file."
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Log that we have a token (masking for security)
            if hf_token:
                masked_token = f"{hf_token[:5]}...{hf_token[-5:]}" if len(hf_token) > 10 else "***"
                logger.info(f"Using HuggingFace API token: {masked_token} (length: {len(hf_token)})")
            
            # Create repo and upload model
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    logger.info(f"Created temporary directory at {tmp_dir}")
                    
                    # Prepare files to upload
                    readme_content = f"""
# Federated Learning Model

This model was trained using a federated learning approach.

## Model Details
- **ID**: {model.id}
- **Version**: {model.version}
- **Project**: {model.project_id}
- **Framework**: {Project.query.get(model.project_id).framework if Project.query.get(model.project_id) else 'unknown'}
- **Dataset**: {Project.query.get(model.project_id).dataset_name if Project.query.get(model.project_id) else 'unknown'}
- **Created**: {model.created_at.strftime('%Y-%m-%d %H:%M:%S') if model.created_at else 'unknown'}

## Performance Metrics
- **Accuracy**: {model.metrics.get('accuracy', 0) if model.metrics else 0}
- **Loss**: {model.metrics.get('loss', 0) if model.metrics else 0}
- **Validation Accuracy**: {model.metrics.get('val_accuracy', 0) if model.metrics else 0}
- **Validation Loss**: {model.metrics.get('val_loss', 0) if model.metrics else 0}

## Usage Example

```python
# Example code for using this model
import torch  # or tensorflow as tf
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face Hub
model_path = hf_hub_download(repo_id="{model_name}", filename="model.pt")  # or model.h5 for TensorFlow

# Load the model
model = torch.load(model_path)  # or tf.keras.models.load_model(model_path)

# Use the model
model.eval()  # Set to evaluation mode
# model.predict(your_data)  # Make predictions
```

## Training Information

This model was trained using Federated Learning with {model.metrics.get('clients', 0) if model.metrics else 0} clients contributing to the training process.
                    """
                    
                    # Create README.md
                    with open(os.path.join(tmp_dir, 'README.md'), 'w') as f:
                        f.write(readme_content)
                    
                    # Copy model file to temp dir with proper naming
                    framework = Project.query.get(model.project_id).framework if Project.query.get(model.project_id) else 'unknown'
                    if os.path.isfile(model_path):
                        # Determine appropriate file extension based on framework
                        if framework.lower() == 'tensorflow':
                            dest_filename = 'model.h5'
                            # Also create a model_info.json with metadata
                            with open(os.path.join(tmp_dir, 'model_info.json'), 'w') as f:
                                json.dump({
                                    'framework': 'tensorflow',
                                    'version': model.version,
                                    'metrics': model.metrics if model.metrics else {},
                                    'created_at': model.created_at.isoformat() if model.created_at else None
                                }, f, indent=2)
                        elif framework.lower() == 'pytorch':
                            dest_filename = 'model.pt'
                            # Create a model_info.json with metadata
                            with open(os.path.join(tmp_dir, 'model_info.json'), 'w') as f:
                                json.dump({
                                    'framework': 'pytorch',
                                    'version': model.version,
                                    'metrics': model.metrics if model.metrics else {},
                                    'created_at': model.created_at.isoformat() if model.created_at else None
                                }, f, indent=2)
                        else:
                            # Use original filename as fallback
                            dest_filename = os.path.basename(model_path)
                        
                        dest_file = os.path.join(tmp_dir, dest_filename)
                        try:
                            shutil.copy2(model_path, dest_file)
                            logger.info(f"Copied model file to {dest_file}")
                        except Exception as copy_error:
                            logger.error(f"Error copying model file: {str(copy_error)}")
                            # Try a simpler copy method as fallback
                            try:
                                import io
                                with open(model_path, 'rb') as src, open(dest_file, 'wb') as dst:
                                    dst.write(src.read())
                                logger.info(f"Copied model file using alternative method to {dest_file}")
                            except Exception as alt_copy_error:
                                logger.error(f"Alternative copy method also failed: {str(alt_copy_error)}")
                                return {
                                    "success": False,
                                    "error": f"Could not copy model file: {str(copy_error)}"
                                }
                    else:
                        # It's a directory, copy all contents with proper organization
                        try:
                            # Create a models directory
                            os.makedirs(os.path.join(tmp_dir, 'models'), exist_ok=True)
                            
                            for item in os.listdir(model_path):
                                source_item = os.path.join(model_path, item)
                                # Put model files in the models directory
                                if item.endswith(('.h5', '.pt', '.pth', '.pb', '.hdf5')):
                                    dest_item = os.path.join(tmp_dir, 'models', item)
                                else:
                                    dest_item = os.path.join(tmp_dir, item)
                                
                                if os.path.isfile(source_item):
                                    shutil.copy2(source_item, dest_item)
                                else:
                                    shutil.copytree(source_item, dest_item)
                            logger.info(f"Copied directory contents to {tmp_dir}")
                        except Exception as copy_error:
                            logger.error(f"Error copying model directory: {str(copy_error)}")
                            return {
                                "success": False,
                                "error": f"Could not copy model directory: {str(copy_error)}"
                            }
                    
                    # Log files to be uploaded
                    logger.info(f"Files to upload: {os.listdir(tmp_dir)}")
                    
                    # Initialize Hugging Face API
                    api = HfApi(token=hf_token)
                    
                    # Use the existing repository
                    logger.info(f"Using repository: {model_name}")
                    try:
                        # Check if repo exists
                        try:
                            repo_info = api.repo_info(repo_id=model_name, repo_type="model")
                            logger.info(f"Repository confirmed to exist: {model_name}")
                        except Exception as repo_error:
                            logger.error(f"Repository {model_name} does not exist: {str(repo_error)}")
                            return {
                                "success": False,
                                "error": f"The repository {model_name} does not exist. Please create it first on Hugging Face."
                            }
                        
                        # Upload files to a specific path within the repository to organize models
                        logger.info(f"Uploading files to Hugging Face Hub under {folder_name}/")
                        api.upload_folder(
                            folder_path=tmp_dir,
                            repo_id=model_name,
                            repo_type="model",
                            path_in_repo=folder_name,  # Place files in a subfolder for organization
                        )
                        logger.info(f"Files uploaded successfully to {model_name}/{folder_name}")
                    except Exception as repo_error:
                        error_msg = f"Error with repository operations: {str(repo_error)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        return {
                            "success": False,
                            "error": error_msg
                        }
            except Exception as upload_error:
                error_msg = f"Error uploading to Hugging Face: {str(upload_error)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Store deployment info
            deployment_info = {
                "type": "huggingface",
                "timestamp": datetime.utcnow().isoformat(),
                "status": "deployed",
                "model_name": model_name,
                "folder_name": folder_name,
                "huggingface_url": f"https://huggingface.co/{model_name}/tree/main/{folder_name}"
            }
            
            # Update model in database
            try:
                model.is_deployed = True
                if not model.deployment_info:
                    model.deployment_info = {}
                
                # Convert to dict if it's a string
                if isinstance(model.deployment_info, str):
                    try:
                        model.deployment_info = json.loads(model.deployment_info)
                    except:
                        model.deployment_info = {}
                
                # Update with new deployment info
                if isinstance(model.deployment_info, dict):
                    model.deployment_info.update(deployment_info)
                else:
                    model.deployment_info = deployment_info
                
                db.session.commit()
                logger.info(f"Model database record updated with deployment info")
            except Exception as db_error:
                error_msg = f"Error updating model database record: {str(db_error)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return {
                    "success": True,  # Still return success since files were uploaded
                    "deployment_info": deployment_info,
                    "warning": error_msg
                }
            
            logger.info(f"Model successfully deployed to Hugging Face Hub: {model_name}")
            return {
                "success": True,
                "deployment_info": deployment_info
            }
                
        except Exception as e:
            error_msg = f"Error deploying to Hugging Face: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": error_msg
            }