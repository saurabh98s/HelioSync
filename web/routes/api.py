"""
API Routes

This module provides API endpoints for client device communication with the server.
"""

import os
import logging
import tempfile
import json
import zipfile
import secrets
import traceback
from datetime import datetime, timedelta
from functools import wraps

from flask import Blueprint, request, jsonify, current_app, send_file
from flask_login import current_user

from web.app import db
from web.models import ApiKey, Organization, Client, Project, ProjectClient

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Cache for project availability (to speed up repeated no-project checks)
_project_cache = {
    'last_updated': datetime.utcnow(),
    'has_projects': False,
    'by_organization': {}
}

def require_api_key(view_func):
    """Decorator to check for valid API key in request."""
    @wraps(view_func)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({"error": "API key is required"}), 401
        
        key_obj = ApiKey.query.filter_by(key=api_key).first()
        
        if not key_obj or not key_obj.is_valid():
            return jsonify({"error": "Invalid or expired API key"}), 401
        
        # Add organization to request
        request.organization = key_obj.organization
        
        return view_func(*args, **kwargs)
    
    return decorated

@api_bp.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    })

@api_bp.route('/register_client', methods=['POST'])
@require_api_key
def register_client():
    """Register a new client or update an existing one."""
    try:
        data = request.get_json()
        current_app.logger.info(f"Received registration request: {data}")
        
        if not data or 'client_id' not in data:
            current_app.logger.error("Missing client_id in registration request")
            return jsonify({"error": "client_id is required"}), 400
        
        client_id = data['client_id']
        current_app.logger.info(f"Processing registration for client {client_id}")
        
        # Check if client already exists
        client = Client.query.filter_by(client_id=client_id).first()
        
        try:
            if not client:
                # Create new client with basic fields first
                client = Client(
                    client_id=client_id,
                    name=data.get('name', f'Client {client_id}'),
                    organization_id=request.organization.id,
                    is_connected=True,
                    last_heartbeat=datetime.utcnow()
                )
                db.session.add(client)
                db.session.flush()  # Flush to get the client ID
                
                current_app.logger.info(f"Created new client: {client_id}")
            
            # Update client fields
            client.is_connected = True
            client.last_heartbeat = datetime.utcnow()
            client.name = data.get('name', client.name)
            client.device_info = data.get('device_info', '')
            
            # Try to set each field individually
            try:
                client.platform = data.get('platform', '')
            except Exception as e:
                current_app.logger.warning(f"Could not set platform: {str(e)}")
            
            try:
                client.machine = data.get('machine', '')
            except Exception as e:
                current_app.logger.warning(f"Could not set machine: {str(e)}")
            
            try:
                client.python_version = data.get('python_version', '')
            except Exception as e:
                current_app.logger.warning(f"Could not set python_version: {str(e)}")
            
            try:
                client.data_size = data.get('data_size', 0)
            except Exception as e:
                current_app.logger.warning(f"Could not set data_size: {str(e)}")
            
            db.session.commit()
            current_app.logger.info(f"Database updated for client {client_id}")
            
        except Exception as e:
            current_app.logger.error(f"Database error: {str(e)}")
            db.session.rollback()
            return jsonify({"error": "Database error"}), 500
        
        # Add client to federated learning server
        fl_server = current_app.fl_server
        if fl_server:
            try:
                # Pass all required arguments to add_client
                success = fl_server.add_client(
                    client_id=client_id,
                    name=data.get('name', f'Client {client_id}'),
                    data_size=data.get('data_size', 0),
                    device_info=data.get('device_info', ''),
                    platform=data.get('platform', ''),
                    machine=data.get('machine', ''),
                    python_version=data.get('python_version', '')
                )
                if success:
                    current_app.logger.info(f"Added client {client_id} to FL server")
                else:
                    current_app.logger.error(f"Failed to add client {client_id} to FL server")
                    return jsonify({"error": "Failed to add client to FL server"}), 500
            except Exception as e:
                current_app.logger.error(f"Error adding client to FL server: {str(e)}")
                return jsonify({"error": "Error adding client to FL server"}), 500
        else:
            current_app.logger.warning("FL server not initialized")
            return jsonify({"error": "FL server not initialized"}), 500
        
        return jsonify({
            "status": "success",
            "client_id": client_id,
            "message": "Client registered successfully"
        })
        
    except Exception as e:
        current_app.logger.error(f"Registration error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/clients/<client_id>/tasks', methods=['GET'])
@require_api_key
def get_client_tasks(client_id):
    """Get training tasks for a client."""
    global _project_cache
    
    try:
        # Quick check for client existence to avoid deeper processing if invalid
        client = Client.query.filter_by(client_id=client_id).with_entities(Client.id, Client.organization_id).first()
        if not client:
            return jsonify({
                "status": "error",
                "message": "Client not found"
            }), 404
            
        # Check if client is requesting a short timeout version (lightweight response)
        short_timeout = request.args.get('short_timeout', 'false').lower() == 'true'
        
        # Check for project_id parameter - direct project request
        project_id = request.args.get('project_id')
        if project_id:
            # Specific project requested - no need for caching
            project = Project.query.get(project_id)
            
            if not project:
                return jsonify({
                    'status': 'error',
                    'message': f'Project {project_id} not found'
                }), 404
            
            # If project is completed, notify client
            if project.status == 'completed':
                return jsonify({
                    'status': 'waiting',
                    'message': f'Project {project_id} is already completed',
                    'details': {
                        'project_id': project_id,
                        'project_status': 'completed'
                    }
                })
        else:
            # Check cache first to quickly return "no projects" without DB query
            cache_age = (datetime.utcnow() - _project_cache['last_updated']).total_seconds()
            org_id = client.organization_id
            
            # Use cache if recent (less than 30 seconds old)
            if cache_age < 30 and org_id in _project_cache['by_organization']:
                if not _project_cache['by_organization'][org_id]:
                    # No projects available based on cache
                    return jsonify({
                        'status': 'waiting',
                        'message': 'No active projects for your organization (cached)',
                        'details': {}
                    })
            
            # Need to query for projects - update cache
            projects_query = Project.query.join(Project.organizations).filter(
                Project.status == 'running',
                Organization.id == org_id
            )
            
            # Optimize query to just count first
            projects_exist = db.session.query(projects_query.exists()).scalar()
            
            # Update cache
            _project_cache['last_updated'] = datetime.utcnow()
            _project_cache['by_organization'][org_id] = projects_exist
            
            if not projects_exist:
                return jsonify({
                    'status': 'waiting',
                    'message': 'No active projects for your organization',
                    'details': {}
                })
            
            # If we get here, projects exist, so get the actual projects
            projects = projects_query.order_by(Project.created_at.desc()).all()
            
            # Always select the most active project (the one with most clients)
            # This ensures all clients join the same project for federated learning
            if projects:
                # Count clients per project to find the most active one
                project_client_counts = {}
                for p in projects:
                    count = ProjectClient.query.filter_by(project_id=p.id).count()
                    project_client_counts[p.id] = count
                
                # Sort projects by client count (descending) then by creation date (newest first)
                sorted_projects = sorted(
                    projects, 
                    key=lambda p: (-project_client_counts.get(p.id, 0), -p.id)
                )
                
                # Select the most active project
                project = sorted_projects[0]
                
                # Assign client to this project if not already assigned
                client_assignment = ProjectClient.query.filter_by(
                    client_id=client.id, project_id=project.id
                ).first()
                
                if not client_assignment:
                    client_assignment = ProjectClient(
                        project_id=project.id,
                        client_id=client.id,
                        status='active'
                    )
                    db.session.add(client_assignment)
                    db.session.commit()
            else:
                project = None
        
        # Get the federated learning server instance
        fl_server = current_app.fl_server
        if not fl_server:
            return jsonify({
                "status": "error",
                "message": "Federated learning server not initialized"
            }), 500
            
        if not project:
            return jsonify({
                "status": "error",
                "message": "No suitable project found"
            }), 500
        
        if project.status != 'running':
            return jsonify({
                "status": "waiting",
                "message": f"Project {project.name} is not currently running",
                "details": {
                    "project_id": project.id,
                    "project_name": project.name,
                    "project_status": project.status,
                    "current_round": project.current_round,
                    "total_rounds": project.rounds
                }
            })
        
        try:
            # Get current model weights for the project
            weights = [w.tolist() for w in fl_server.get_model_weights(project.id)]
            
            # For short timeout requests, don't include weights to speed up response
            response_data = {
                "status": "training",
                "message": "Training task available",
                "details": {
                    "project_id": project.id,
                    "project_name": project.name,
                    "round": fl_server.current_round,
                    "total_rounds": project.rounds,
                    "framework": project.framework,
                    "dataset": project.dataset_name
                }
            }
            
            # Only include weights for full requests
            if not short_timeout:
                response_data["details"]["weights"] = weights
            else:
                # Include a flag indicating weights are available but not included
                response_data["details"]["weights_available"] = True
            
            return jsonify(response_data)
            
        except Exception as e:
            current_app.logger.error(f"Error getting model weights: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Error getting model weights",
                "details": {
                    "error": str(e),
                    "project_id": project.id,
                    "project_name": project.name
                }
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Error getting client tasks: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "details": {
                "error": str(e),
                "client_id": client_id
            }
        }), 500

@api_bp.route('/clients/<client_id>/model_update', methods=['POST'])
@require_api_key
def model_update(client_id):
    """Update the model with client's weights."""
    try:
        data = request.get_json()
        weights = data.get('weights', [])
        metrics = data.get('metrics', {})
        
        # Extract project_id from metrics if available, otherwise use the first active project
        project_id = metrics.get('project_id')
        
        # Check if this is the final update - this impacts many decisions below
        is_final = metrics.get('is_final', False)
        
        # First step: Record that we received the update so client doesn't timeout
        # This quick initial response prevents client timeouts
        if weights and project_id:
            # Start a background thread to process the update
            def process_update_async():
                try:
                    # All the processing code that was previously here
                    process_model_update(client_id, project_id, weights, metrics, is_final)
                except Exception as e:
                    logging.error(f"Async model update processing error: {str(e)}")
                    logging.error(traceback.format_exc())
            
            # Start background thread
            import threading
            update_thread = threading.Thread(target=process_update_async)
            update_thread.daemon = True
            update_thread.start()
            
            # Return success immediately while processing continues in background
            return jsonify({
                'status': 'success',
                'message': 'Model update received and processing started',
                'details': {
                    'project_id': project_id,
                    'is_final_update': is_final,
                    'background_processing': True
                }
            })
        
        # Get the client
        client = Client.query.filter_by(client_id=client_id).first()
        if not client:
            # Normal handling for client not found...
            return jsonify({'status': 'error', 'message': 'Client not found'}), 404
        
        # Always update client last seen time
        if client:
            client.last_seen = datetime.utcnow()
            db.session.commit()
        
        # Just return success if we have nothing to process
        return jsonify({
            'status': 'success',
            'message': 'Model update received (no weights or project_id)'
        })
            
    except Exception as e:
        logging.error(f"Error in model_update: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

def process_model_update(client_id, project_id, weights, metrics, is_final):
    """Background processing function for model updates."""
    # All the update processing code from the original model_update function
    try:
        # Get the client
        client = Client.query.filter_by(client_id=client_id).first()
        if not client:
            logging.error(f"Client {client_id} not found in background update")
            return
        
        # Always update client last seen time
        client.last_seen = datetime.utcnow()
        db.session.commit()
        
        # Find the project
        project = Project.query.get(project_id)
        if not project:
            logging.error(f"Project {project_id} not found in background update")
            return
        
        # Update model weights in FL server
        server = current_app.fl_server
        if not server:
            logging.error("FL server not available in background update")
            return
        
        # Log the final update status
        if is_final:
            logging.info(f"Processing FINAL update from client {client_id} for project {project_id}")
            
            # Determine if this is actually the final round
            is_final_round = project.current_round >= (project.rounds - 1)
            
            # Only mark as final in the server if we're actually in the final round
            if is_final_round:
                logging.info(f"Confirming this is the final round ({project.current_round}/{project.rounds})")
            else:
                logging.warning(f"Client sent final flag but project is only at round {project.current_round}/{project.rounds}")
                # Override is_final flag if we're not actually in the final round
                is_final = False
        
        # Update the model with the client's weights
        success = server.update_model(client_id, weights, metrics, project_id)
        
        if success:
            # Get latest project status after update
            db.session.refresh(project)
            
            # Update client-project association to track participation
            client_project = ProjectClient.query.filter_by(
                client_id=client.id, project_id=project.id
            ).first()
            
            if client_project:
                client_project.last_update = datetime.utcnow()
                client_project.training_samples = metrics.get('samples', 0)
                client_project.status = 'completed' if is_final else 'training'
                
                # Store metrics as JSON
                if not client_project.metrics:
                    client_project.metrics = {}
                
                # Update metrics with latest round
                round_metrics = {
                    'round': project.current_round,
                    'loss': metrics.get('loss', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'val_loss': metrics.get('val_loss', 0),
                    'val_accuracy': metrics.get('val_accuracy', 0)
                }
                
                # Add to metrics history
                metrics_dict = dict(client_project.metrics)  # Convert from JSON
                if 'rounds' not in metrics_dict:
                    metrics_dict['rounds'] = []
                metrics_dict['rounds'].append(round_metrics)
                client_project.metrics = metrics_dict
                
                db.session.commit()
            
            # For final updates, make sure the project is marked as completed
            if is_final and project.status != 'completed':
                # Ensure project has actually progressed through enough rounds
                if project.current_round >= (project.rounds - 1):
                    logging.info(f"Marking project {project_id} as completed for final update")
                    project.status = 'completed'
                    db.session.commit()
                else:
                    logging.warning(f"Not marking project as completed: only at round {project.current_round} of {project.rounds}")
            
            logging.info(f"Successfully processed model update for client {client_id}, project {project_id}")
        else:
            logging.error(f"Failed to process model update for client {client_id}, project {project_id}")
        
    except Exception as e:
        logging.error(f"Error in background model update: {str(e)}")
        logging.error(traceback.format_exc())

@api_bp.route('/clients/<client_id>/heartbeat', methods=['POST'])
@require_api_key
def client_heartbeat(client_id):
    """Handle client heartbeat to update last seen timestamp."""
    try:
        # Optimized query that just updates timestamp without fetching the client first
        updated = db.session.query(Client).filter_by(client_id=client_id).update({
            'last_heartbeat': datetime.utcnow(),
            'is_connected': True
        })
        
        db.session.commit()
        
        if not updated:
            return jsonify({
                "status": "error",
                "message": "Client not found"
            }), 404
        
        return jsonify({
            "status": "success",
            "message": "Heartbeat received"
        })
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error updating heartbeat: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error updating heartbeat"
        }), 500

@api_bp.route('/clients/tasks', methods=['GET'])
@require_api_key
def client_tasks():
    """Get global task information for all clients."""
    try:
        # Get the federated learning server instance
        fl_server = current_app.fl_server
        
        if not fl_server:
            return jsonify({
                "status": "error",
                "message": "Federated learning server not initialized"
            }), 500
        
        # Get current round information
        current_round = fl_server.current_round
        total_rounds = fl_server.rounds
        
        return jsonify({
            "status": "success",
            "message": "Server status retrieved",
            "details": {
                "current_round": current_round,
                "total_rounds": total_rounds,
                "min_clients": fl_server.min_clients,
                "connected_clients": len(fl_server.clients)
            }
        })
    except Exception as e:
        current_app.logger.error(f"Error getting server status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error getting server status",
            "details": {
                "error": str(e)
            }
        }), 500

# Client registration aliases
@api_bp.route('/client/register', methods=['POST'])
@require_api_key
def register_client_alias1():
    """Alias for register_client endpoint."""
    return register_client()

@api_bp.route('/clients/register', methods=['POST'])
@require_api_key
def register_client_alias2():
    """Alias for register_client endpoint."""
    return register_client()

@api_bp.route('/register', methods=['POST'])
@require_api_key
def register_client_alias3():
    """Alias for register_client endpoint."""
    return register_client()

@api_bp.route('/projects')
@require_api_key
def get_projects():
    """Get available projects for a client."""
    try:
        # Get all projects associated with the organization that are running
        projects = Project.query.filter(
            Project.organizations.contains(request.organization),
            Project.status == 'running'
        ).all()
        
        result = []
        for project in projects:
            result.append({
                "id": project.id,
                "name": project.name,
                "dataset": project.dataset_name,
                "framework": project.framework,
                "current_round": project.current_round,
                "total_rounds": project.rounds
            })
        
        return jsonify({
            "success": True,
            "projects": result
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting projects: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/projects/<int:project_id>/server_info')
@require_api_key
def get_server_info(project_id):
    """Get server information for a project."""
    try:
        # Get the project
        project = Project.query.get_or_404(project_id)
        
        # Check if the organization has access to this project
        if request.organization not in project.organizations:
            return jsonify({"error": "Access denied"}), 403
        
        # Get the server information
        server_info = {
            "host": current_app.config['FL_SERVER_HOST'],
            "port": current_app.config['FL_SERVER_PORT'],
            "token": current_app.config['FL_SERVER_TOKEN']
        }
        
        return jsonify({
            "success": True,
            "server_info": server_info,
            "project": {
                "id": project.id,
                "name": project.name,
                "dataset": project.dataset_name,
                "framework": project.framework,
                "current_round": project.current_round,
                "total_rounds": project.rounds
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting server info: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/projects/<int:project_id>/update_status', methods=['POST'])
@require_api_key
def update_project_status(project_id):
    """Update project status - used by the federated learning server."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get the project
        project = Project.query.get_or_404(project_id)
        
        # Check if the organization has access to this project
        if request.organization not in project.organizations:
            return jsonify({"error": "Access denied"}), 403
        
        # Update the project status
        if 'status' in data:
            project.status = data['status']
        
        if 'current_round' in data:
            project.current_round = data['current_round']
        
        # Handle project completion
        if data.get('status') == 'completed':
            # Create a new model version if metrics are provided
            if 'metrics' in data:
                metrics = data['metrics']
                
                from web.services.model_manager import ModelManager
                
                model_data = {
                    'accuracy': metrics.get('accuracy', 0),
                    'loss': metrics.get('loss', 0),
                    'clients': metrics.get('clients', 0),
                    'model_file': metrics.get('model_file')
                }
                
                ModelManager.save_model(project, model_data, is_final=True)
        
        db.session.commit()
        
        return jsonify({
            "success": True,
            "project": {
                "id": project.id,
                "status": project.status,
                "current_round": project.current_round
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error updating project status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/clients/<client_id>/status', methods=['GET'])
@require_api_key
def get_client_status(client_id):
    """Get current status and instructions for a client."""
    fl_server = current_app.fl_server
    
    if not fl_server:
        return jsonify({"error": "Federated learning server not initialized"}), 500
    
    # Get client's current status from server
    status = fl_server.get_client_status(client_id)
    
    return jsonify({
        "status": status,
        "message": "Client status retrieved successfully"
    })

@api_bp.route('/clients/<client_id>/control', methods=['POST'])
@require_api_key
def control_client(client_id):
    """Control client's training process."""
    data = request.get_json()
    
    if not data or 'action' not in data:
        return jsonify({"error": "action is required"}), 400
    
    fl_server = current_app.fl_server
    
    if not fl_server:
        return jsonify({"error": "Federated learning server not initialized"}), 500
    
    action = data['action']
    if action not in ['continue', 'stop', 'wait']:
        return jsonify({"error": "Invalid action"}), 400
    
    # Update client's status in server
    fl_server.update_client_status(client_id, action)
    
    return jsonify({
        "status": "success",
        "message": f"Client {client_id} status updated to {action}"
    })

@api_bp.route('/metrics', methods=['GET'])
@require_api_key
def get_metrics():
    """Get current training metrics."""
    fl_server = current_app.fl_server
    
    if not fl_server:
        return jsonify({"error": "Federated learning server not initialized"}), 500
    
    metrics = fl_server.get_metrics()
    
    return jsonify(metrics)

@api_bp.route('/clients/<client_id>/metrics', methods=['GET'])
@require_api_key
def get_client_metrics(client_id):
    """Get metrics for a specific client."""
    fl_server = current_app.fl_server
    
    if not fl_server:
        return jsonify({"error": "Federated learning server not initialized"}), 500
    
    metrics = fl_server.get_metrics()
    current_round = metrics['current_round']
    
    # Get client's metrics for current round
    client_metrics = None
    if current_round in metrics['round_metrics']:
        for metric in metrics['round_metrics'][current_round]:
            if metric['client_id'] == client_id:
                client_metrics = metric
                break
    
    return jsonify({
        'client_id': client_id,
        'current_round': current_round,
        'metrics': client_metrics
    })

@api_bp.route('/disconnect_client', methods=['POST'])
@require_api_key
def disconnect_client():
    """Handle client disconnection."""
    try:
        data = request.get_json()
        if not data or 'client_id' not in data:
            return jsonify({'error': 'Missing client_id'}), 400
        
        client_id = data['client_id']
        project_id = data.get('project_id')  # Optional: if provided, disconnect from specific project
        
        # Get the FL server instance
        fl_server = current_app.fl_server
        
        if project_id:
            # Disconnect from specific project
            success = fl_server.remove_client_from_project(client_id, project_id)
            if success:
                return jsonify({
                    'message': f'Client {client_id} disconnected from project {project_id}',
                    'status': 'success'
                })
            return jsonify({
                'error': f'Failed to disconnect client {client_id} from project {project_id}',
                'status': 'error'
            }), 500
        else:
            # Disconnect from all projects and remove client
            success = fl_server.remove_client(client_id)
            if success:
                return jsonify({
                    'message': f'Client {client_id} disconnected successfully',
                    'status': 'success'
                })
            return jsonify({
                'error': f'Failed to disconnect client {client_id}',
                'status': 'error'
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Error disconnecting client: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500 