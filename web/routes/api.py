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
from web.models import ApiKey, Organization, Client, Project, ProjectClient, Model

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
                    'status': 'completed',
                    'message': f'Project {project_id} has been completed. Training cycle finished.',
                    'details': {
                        'project_id': project_id,
                        'project_status': 'completed',
                        'should_stop': True,
                        'training_complete': True,
                        'next_action': 'stop_training'
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
        
        # Special handling for completed projects
        if project.status == 'completed':
            return jsonify({
                "status": "completed",
                "message": f"Project {project.name} has been completed. Training cycle finished.",
                "details": {
                    "project_id": project.id,
                    "project_name": project.name,
                    "project_status": "completed",
                    "current_round": project.current_round,
                    "total_rounds": project.rounds,
                    "should_stop": True,  # Explicit flag telling client to stop
                    "training_complete": True,  # Additional flag to make the stop message even clearer
                    "next_action": "stop_training"  # Direct instruction for the client
                }
            })

        # Other non-running states
        if project.status != 'running':
            return jsonify({
                "status": "waiting",
                "message": f"Project {project.name} is not currently running",
                "details": {
                    "project_id": project.id,
                    "project_name": project.name,
                    "project_status": project.status,
                    "current_round": project.current_round,
                    "total_rounds": project.rounds,
                    "should_wait": True  # Explicit flag telling client to wait
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
                    "dataset": project.dataset_name,
                    "aggregation": {
                        "method": "perfedavg", # Default method
                        "alpha": 0.5  # Default alpha value
                    }
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
    """Update the model with client's weights.
    
    Clients can send model updates with optional aggregation preferences:
    
    Request JSON format:
    {
        "weights": [...],  # Array of model weight arrays
        "metrics": {       # Training metrics
            "accuracy": 0.95,
            "loss": 0.05,
            "val_accuracy": 0.92,
            "val_loss": 0.08,
            "round": 1,
            "is_final": false
        },
        "aggregation": {   # Optional aggregation preferences
            "method": "perfedavg",  # "perfedavg" or "fedavg"
            "alpha": 0.5    # Balance between data size (1.0) and accuracy (0.0)
        }
    }
    
    The "aggregation" field is optional. If provided:
    - "method" can be "perfedavg" (performance-weighted) or "fedavg" (standard)
    - "alpha" controls the balance in PerfFedAvg:
      - alpha=1.0: Pure data size weighting (equivalent to FedAvg)
      - alpha=0.0: Pure accuracy weighting (only performance matters)
      - alpha=0.5: Balanced weighting (default)
    """
    try:
        data = request.get_json()
        weights = data.get('weights', [])
        metrics = data.get('metrics', {})
        
        # Process aggregation preferences
        aggregation_prefs = data.get('aggregation', {})
        if aggregation_prefs:
            # Add the aggregation preferences to metrics for the FL server
            if 'alpha' in aggregation_prefs:
                metrics['alpha'] = float(aggregation_prefs['alpha'])
            if 'method' in aggregation_prefs:
                # Convert method name to boolean flag for PerfFedAvg
                metrics['use_perfedavg'] = aggregation_prefs['method'].lower() == 'perfedavg'
        
        # Validate that weights were provided
        if not weights:
            current_app.logger.error(f"Client {client_id} submitted empty weights")
            return jsonify({
                "success": False,
                "message": "No weights provided in request",
                "should_retry": True
            }), 400
        
        # Check weight format and ensure they're valid arrays
        valid_weights = True
        weight_info = []
        weight_shapes = []
        
        # Sample a few weights to validate format
        for i, w in enumerate(weights[:5]):  # Log info for first few weights
            try:
                if isinstance(w, list):
                    # Get shape information
                    shape = [len(w)]
                    if len(w) > 0 and isinstance(w[0], list):
                        shape.append(len(w[0]))
                        # Check for deeper nesting if needed
                        if len(w[0]) > 0 and isinstance(w[0][0], list):
                            shape.append(len(w[0][0]))
                    
                    # Check for empty arrays
                    if len(w) == 0:
                        weight_info.append(f"Empty list at index {i}")
                        valid_weights = False
                    else:
                        dtype = type(w[0]).__name__ if len(w) > 0 else "unknown"
                        weight_info.append(f"List shape:{shape}, dtype:{dtype}")
                        weight_shapes.append(shape)
                elif isinstance(w, dict):
                    weight_info.append(f"Dict with keys: {list(w.keys())}")
                    valid_weights = False
                else:
                    weight_info.append(f"Non-list type:{type(w).__name__}")
                    valid_weights = False
            except Exception as e:
                weight_info.append(f"Error inspecting weight at index {i}: {str(e)}")
                valid_weights = False
        
        if not valid_weights:
            current_app.logger.error(f"Client {client_id} submitted invalid weight format: {weight_info}")
            return jsonify({
                "success": False,
                "message": f"Invalid weight format: {', '.join(weight_info[:3])}",
                "should_retry": True,
                "details": {
                    "weight_info": weight_info
                }
            }), 400
            
        # Extract project_id from metrics if available, otherwise use the first active project
        project_id = metrics.get('project_id')
        
        # Check if this is the final update - this impacts many decisions below
        is_final = metrics.get('is_final', False)
        current_round = metrics.get('round', 0)
        
        # Try to get project if not specified in metrics
        if not project_id:
            # Try to find the project this client is currently working on
            client_obj = Client.query.filter_by(client_id=client_id).first()
            if client_obj:
                # Look for active project assignments
                project_client = ProjectClient.query.filter_by(
                    client_id=client_obj.id, 
                    status='active'
                ).order_by(ProjectClient.last_update.desc()).first()
                
                if project_client:
                    project_id = project_client.project_id
                    current_app.logger.info(f"Found active project {project_id} for client {client_id}")
                    
                    # Update metrics with project_id
                    metrics['project_id'] = project_id
        
        # Get the FL server and app context for the background thread
        fl_server = current_app.fl_server
        app = current_app._get_current_object()
        
        current_app.logger.info(f"Received model update from client {client_id} for project {project_id}, round {current_round}")
        current_app.logger.info(f"Weight count: {len(weights)}, first few weights: {weight_info}")
        
        # For final updates, make sure we set the is_final flag in metrics
        if is_final and 'is_final' not in metrics:
            metrics['is_final'] = True
            
        # Process the update in a separate thread to avoid blocking the API
        def process_model_update():
            with app.app_context():
                try:
                    # Capture the project_id from the outer scope
                    local_project_id = project_id
                    
                    # Validate the project exists and is active
                    project = None
                    
                    if local_project_id:
                        project = Project.query.get(local_project_id)
                        
                    if not project and fl_server:
                        # Try to find from client assignments in FL server
                        active_projects = fl_server.get_client_projects(client_id)
                        if active_projects:
                            local_project_id = active_projects[0]
                            project = Project.query.get(local_project_id)
                    
                    if not project:
                        current_app.logger.error(f"No project found for client {client_id}")
                        return
                    
                    # If project is already completed, just log and exit thread - don't process update
                    if project.status == 'completed':
                        current_app.logger.info(f"Project {local_project_id} is already completed, ignoring update from client {client_id}")
                        return
                    
                    # If project is not running, but this is a final update, we might still want to process it
                    # to ensure proper project completion
                    if project.status != 'running' and not is_final:
                        current_app.logger.warning(f"Project {local_project_id} is not running (status: {project.status}), but processing update anyway")
                    
                    # Update the model with the client's weights
                    result = fl_server.update_model(client_id, weights, metrics, project_id=project.id)
                    
                    if result and result.get('aggregated', False):
                        # The model was aggregated - potentially save it
                        current_app.logger.info(f"Model aggregated for project {project.id}, round {project.current_round}")
                        
                        # Check if we need to save this model
                        should_save = False
                        is_final_round = False
                        round_completed = result.get('round_completed', False)
                        
                        # Single-round projects should save and complete after first round
                        if project.rounds == 1 and round_completed:
                            current_app.logger.info(f"Single-round project {project.id} completed")
                            should_save = True
                            is_final_round = True
                        
                        # Multi-round projects save after each round and mark as final at end
                        elif project.rounds > 1:
                            if round_completed:
                                current_app.logger.info(f"Round {project.current_round} completed for project {project.id}")
                                should_save = True
                                
                                # Check if this was the final round
                                if project.current_round >= project.rounds:
                                    current_app.logger.info(f"Final round {project.current_round} completed for project {project.id}")
                                    is_final_round = True
                        
                        # Save the model if needed
                        if should_save:
                            current_app.logger.info(f"Saving model for project {project.id}, round {project.current_round}, is_final={is_final_round}")
                            model = fl_server.save_model(project.id, project.current_round, is_final=is_final_round)
                            
                            if model:
                                current_app.logger.info(f"Model saved successfully: {model.id}")
                                
                                # Check if we need to update the project round
                                if round_completed and not is_final_round:
                                    project.current_round += 1
                                    db.session.commit()
                                    current_app.logger.info(f"Project {project.id} advanced to round {project.current_round}")
                            else:
                                current_app.logger.error(f"Failed to save model for project {project.id}")
                        
                        # If final round, update project status
                        if is_final_round or is_final:
                            project.status = 'completed'
                            db.session.commit()
                            current_app.logger.info(f"Project {project.id} marked as completed")
                    
                except Exception as e:
                    current_app.logger.error(f"Error processing model update: {str(e)}")
                    import traceback
                    current_app.logger.error(traceback.format_exc())
        
        # Start the processing thread
        import threading
        thread = threading.Thread(target=process_model_update)
        thread.daemon = True
        thread.start()
        
        # Get project status to add to response
        project_status = None
        current_project_round = None
        total_rounds = None
        
        if project_id:
            project = Project.query.get(project_id)
            if project:
                project_status = project.status
                current_project_round = project.current_round
                total_rounds = project.rounds
        
        # Return success immediately while processing happens in the background
        response = {
            "success": True, 
            "message": "Update received and being processed",
            "project_id": project_id,
            "round": current_round
        }
        
        # Add project status information if available
        if project_status:
            response["project_status"] = project_status
            response["current_round"] = current_project_round
            response["total_rounds"] = total_rounds
            
            # Calculate if client should stop training
            should_stop = (
                project_status == 'completed' or
                is_final or  # Always stop if client sent final flag
                (current_project_round >= total_rounds) or  # Stop if we've reached the total rounds
                (total_rounds == 1 and len(weights) > 0)  # For single-round projects, stop after first submission
            )
            
            response["should_stop"] = should_stop
            
            if should_stop:
                response["message"] = "Training complete. Thank you for your contribution!"
                response["status"] = "completed"  # Explicit status to help client decide to stop
                response["training_complete"] = True  # Extra flag to make it super clear
                response["next_action"] = "stop_training"  # Explicit instruction for the client
        
        return jsonify(response)
    
    except Exception as e:
        current_app.logger.error(f"Error handling model update: {str(e)}")
        import traceback
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "should_retry": True
        }), 500

@api_bp.route('/projects/<int:project_id>/aggregation', methods=['POST'])
@require_api_key
def update_project_aggregation(project_id):
    """Update aggregation settings for a project.
    
    Request JSON format:
    {
        "method": "perfedavg",  # "perfedavg" or "fedavg"
        "alpha": 0.5,           # Balance between data size and performance
        "save_settings": true   # Whether to save these settings for the project
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get the project
        project = Project.query.get_or_404(project_id)
        
        # Check if the organization has access to this project
        if request.organization not in project.organizations:
            return jsonify({"error": "Access denied"}), 403
        
        # Get current settings or create default
        settings = project.settings if hasattr(project, 'settings') and project.settings else {}
        
        # Extract aggregation settings
        method = data.get('method', 'perfedavg')
        alpha = float(data.get('alpha', 0.5))
        
        # Validate inputs
        if method not in ['perfedavg', 'fedavg']:
            return jsonify({"error": "Invalid aggregation method. Use 'perfedavg' or 'fedavg'."}), 400
        
        if not 0 <= alpha <= 1:
            return jsonify({"error": "Alpha must be between 0 and 1"}), 400
        
        # Update aggregation settings
        if 'aggregation' not in settings:
            settings['aggregation'] = {}
            
        settings['aggregation']['method'] = method
        settings['aggregation']['alpha'] = alpha
        
        # Save settings to project if requested
        if data.get('save_settings', False):
            project.settings = settings
            db.session.commit()
            current_app.logger.info(f"Updated aggregation settings for project {project_id}: {settings['aggregation']}")
        
        # Pass settings to FL server 
        fl_server = current_app.fl_server
        if fl_server:
            try:
                # If project is initialized in FL server, update its settings
                if project_id in fl_server.projects:
                    if 'settings' not in fl_server.projects[project_id]:
                        fl_server.projects[project_id]['settings'] = {}
                    fl_server.projects[project_id]['settings']['aggregation'] = {
                        'method': method,
                        'alpha': alpha,
                        'use_perfedavg': method == 'perfedavg'
                    }
                    current_app.logger.info(f"Updated FL server aggregation settings for project {project_id}")
            except Exception as e:
                current_app.logger.error(f"Error updating FL server settings: {str(e)}")
        
        return jsonify({
            "success": True,
            "message": f"Aggregation settings updated for project {project_id}",
            "settings": {
                "method": method,
                "alpha": alpha
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error updating aggregation settings: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500

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

@api_bp.route('/models/<model_id>/comparison', methods=['GET'])
def get_model_comparison(model_id):
    """
    Get model comparison data for display in the UI.
    
    This endpoint returns comparison metrics for the specified model,
    including evaluation metrics from testing on standard datasets.
    """
    try:
        # Check for API key or user login
        has_api_key = False
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if api_key:
            key_obj = ApiKey.query.filter_by(key=api_key).first()
            has_api_key = bool(key_obj and key_obj.is_valid())
        
        # Only proceed if API key is valid or user is logged in
        if not has_api_key and not current_user.is_authenticated:
            return jsonify({'status': 'error', 'message': 'Authentication required'}), 401
        
        # Get the model details
        model = Model.query.get(model_id)
        if not model:
            return jsonify({'status': 'error', 'message': 'Model not found'}), 404
            
        # Get the project for this model
        project = Project.query.get(model.project_id)
        if not project:
            return jsonify({'status': 'error', 'message': 'Project not found'}), 404
            
        # Get aggregated metrics from FL manager
        fl_manager = current_app.config.get('FL_MANAGER')
        if not fl_manager:
            return jsonify({'status': 'error', 'message': 'FL manager not available'}), 500
            
        metrics = fl_manager.aggregated_metrics.get(str(project.id), {})
        
        # If no metrics available, return placeholder data
        if not metrics:
            current_app.logger.warning(f"No comparison metrics available for model {model_id}")
            return jsonify({
                'status': 'success',
                'message': 'No comparison metrics available',
                'comparison': {
                    'accuracy': {
                        'current': model.accuracy or 0.0,
                        'baseline': None,
                        'improvement': None
                    },
                    'loss': {
                        'current': model.loss or 0.0,
                        'baseline': None,
                        'improvement': None
                    },
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'aggregation_method': 'FedAvg'
                }
            })
        
        # Construct comparison data
        comparison = {
            'accuracy': {
                'current': float(metrics.get('accuracy', model.accuracy or 0.0)),
                'baseline': 0.85,  # Typical baseline for MNIST/CIFAR
                'improvement': float(metrics.get('accuracy', model.accuracy or 0.0)) - 0.85
            },
            'loss': {
                'current': float(metrics.get('loss', model.loss or 0.0)),
                'baseline': 0.5,  # Typical baseline for MNIST/CIFAR
                'improvement': 0.5 - float(metrics.get('loss', model.loss or 0.0))
            },
            'aggregation_method': metrics.get('aggregation_method', 'FedAvg')
        }
        
        # Add additional metrics if available
        if 'precision' in metrics:
            comparison['precision'] = float(metrics['precision'])
        if 'recall' in metrics:
            comparison['recall'] = float(metrics['recall'])
        if 'f1' in metrics:
            comparison['f1'] = float(metrics['f1'])
            
        # Return comparison data
        return jsonify({
            'status': 'success',
            'comparison': comparison
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting model comparison: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error getting model comparison: {str(e)}'}), 500 