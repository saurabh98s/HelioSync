"""
API Routes

This module provides API endpoints for client device communication with the server.
"""

import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from functools import wraps
import numpy as np

from web.app import db
from web.models import ApiKey, Organization, Client, Project, ProjectClient

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

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
    try:
        # Get the federated learning server instance
        fl_server = current_app.fl_server
        
        if not fl_server:
            return jsonify({
                "status": "error",
                "message": "Federated learning server not initialized"
            }), 500
        
        # Check if client is registered
        client = Client.query.filter_by(client_id=client_id).first()
        if not client:
            return jsonify({
                "status": "error",
                "message": "Client not found"
            }), 404
        
        # Get client's active projects
        active_projects = fl_server.get_client_projects(client_id)
        if not active_projects:
            return jsonify({
                "status": "waiting",
                "message": "No active projects available. Please wait for project assignment.",
                "details": {
                    "client_id": client_id,
                    "is_connected": client.is_connected,
                    "last_heartbeat": client.last_heartbeat.isoformat() if client.last_heartbeat else None
                }
            })
        
        # For now, we'll use the first active project
        # In a real implementation, you might want to handle multiple projects
        project_id = active_projects[0]
        project = Project.query.get(project_id)
        
        if not project:
            return jsonify({
                "status": "error",
                "message": f"Project {project_id} not found"
            }), 404
        
        if project.status != 'running':
            return jsonify({
                "status": "waiting",
                "message": f"Project {project.name} is not currently running",
                "details": {
                    "project_id": project_id,
                    "project_name": project.name,
                    "project_status": project.status,
                    "current_round": project.current_round,
                    "total_rounds": project.rounds
                }
            })
        
        try:
            # Get current model weights for the project
            weights = [w.tolist() for w in fl_server.get_model_weights(project_id)]
            
            return jsonify({
                "status": "training",
                "message": "Training task available",
                "details": {
                    "project_id": project_id,
                    "project_name": project.name,
                    "round": fl_server.current_round,
                    "total_rounds": project.rounds,
                    "framework": project.framework,
                    "dataset": project.dataset_name,
                    "weights": weights
                }
            })
            
        except Exception as e:
            current_app.logger.error(f"Error getting model weights: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Error getting model weights",
                "details": {
                    "error": str(e),
                    "project_id": project_id,
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
def update_model(client_id):
    """Handle model updates from clients."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
        
        if 'weights' not in data:
            return jsonify({
                "status": "error",
                "message": "weights are required"
            }), 400
            
        if 'metrics' not in data or not isinstance(data['metrics'], dict):
            return jsonify({
                "status": "error",
                "message": "valid metrics dictionary is required"
            }), 400
        
        # Get the federated learning server instance
        fl_server = current_app.fl_server
        
        if not fl_server:
            return jsonify({
                "status": "error",
                "message": "Federated learning server not initialized"
            }), 500
        
        try:
            # Convert weights to numpy arrays
            weights = [np.array(w) for w in data['weights']]
            
            # Update the model with client's weights
            project_id = data['metrics'].get('project_id')
            if not project_id:
                return jsonify({
                    "status": "error",
                    "message": "project_id is required in metrics"
                }), 400
                
            success = fl_server.update_model(client_id, weights, data['metrics'])
            
            # Update client's last seen timestamp
            client = Client.query.filter_by(client_id=client_id).first()
            if client:
                client.last_heartbeat = datetime.utcnow()
                db.session.commit()
            
            return jsonify({
                "status": "success",
                "message": "Model update received",
                "details": {
                    "client_id": client_id,
                    "project_id": project_id,
                    "round": data['metrics'].get('round', 0),
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
        except Exception as e:
            current_app.logger.error(f"Error updating model: {str(e)}")
            db.session.rollback()
            return jsonify({
                "status": "error",
                "message": "Error updating model",
                "details": {
                    "error": str(e),
                    "client_id": client_id
                }
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Error processing model update: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error processing request",
            "details": {
                "error": str(e),
                "client_id": client_id
            }
        }), 500

@api_bp.route('/clients/<client_id>/heartbeat', methods=['POST'])
@require_api_key
def client_heartbeat(client_id):
    """Handle client heartbeat to update last seen timestamp."""
    try:
        client = Client.query.filter_by(client_id=client_id).first()
        
        if not client:
            return jsonify({
                "status": "error",
                "message": "Client not found"
            }), 404
        
        client.last_heartbeat = datetime.utcnow()
        client.is_connected = True
        db.session.commit()
        
        return jsonify({
            "status": "success",
            "message": "Heartbeat received",
            "details": {
                "client_id": client_id,
                "last_heartbeat": client.last_heartbeat.isoformat()
            }
        })
    except Exception as e:
        current_app.logger.error(f"Error updating heartbeat: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error updating heartbeat",
            "details": {
                "error": str(e),
                "client_id": client_id
            }
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