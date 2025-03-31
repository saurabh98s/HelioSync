"""
API Routes

This module provides API endpoints for client device communication with the server.
"""

import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from functools import wraps

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
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        client_id = data.get('client_id')
        client_name = data.get('name', 'Unnamed Client')
        ip_address = request.remote_addr
        device_info = data.get('device_info', '')
        
        # Check if client already exists
        client = Client.query.filter_by(client_id=client_id).first()
        
        if client:
            # Update existing client
            client.name = client_name
            client.ip_address = ip_address
            client.device_info = device_info
            client.is_connected = True
            client.last_heartbeat = datetime.utcnow()
        else:
            # Create new client
            client = Client(
                client_id=client_id,
                name=client_name,
                ip_address=ip_address,
                device_info=device_info,
                is_connected=True,
                last_heartbeat=datetime.utcnow(),
                organization=request.organization
            )
            db.session.add(client)
        
        db.session.commit()
        
        return jsonify({
            "success": True,
            "client_id": client.client_id,
            "name": client.name,
            "organization": {
                "id": request.organization.id,
                "name": request.organization.name
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error registering client: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/clients/<client_id>/heartbeat', methods=['POST'])
@require_api_key
def client_heartbeat(client_id):
    """Update client heartbeat status."""
    try:
        client = Client.query.filter_by(
            client_id=client_id, 
            organization_id=request.organization.id
        ).first()
        
        if not client:
            return jsonify({"error": "Client not found"}), 404
        
        client.last_heartbeat = datetime.utcnow()
        client.is_connected = True
        db.session.commit()
        
        return jsonify({
            "success": True,
            "client_id": client.client_id,
            "timestamp": client.last_heartbeat.isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error updating client heartbeat: {str(e)}")
        return jsonify({"error": str(e)}), 500

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