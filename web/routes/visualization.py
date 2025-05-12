"""
Visualization Routes

This module provides routes for visualizing training metrics and model performance.
"""

from flask import Blueprint, render_template, jsonify, current_app, abort, flash
from flask_login import login_required, current_user
from web.models import Project, Model, ProjectClient
from datetime import datetime, timedelta
import random
from flask_sqlalchemy import SQLAlchemy
from web.extensions import db
import os

# Create blueprint
visualization_bp = Blueprint('visualization', __name__)

@visualization_bp.route('/')
@login_required
def index():
    """Show all projects with visualization options"""
    if current_user.is_admin:
        projects = Project.query.all()
    else:
        projects = Project.query.filter_by(creator_id=current_user.id).all()
    return render_template('visualization/index.html', projects=projects)

def create_sample_models(project):
    """
    Create sample models for visualization if none exist.
    NOTE: This function is no longer used as we're only showing real models now.
    """
    current_app.logger.info("Sample model creation is disabled")
    return False

def remove_sample_models(project_id):
    """Remove sample models when real models are available."""
    try:
        # Get sample models
        sample_models = Model.query.filter_by(project_id=project_id, is_sample=True).all()
        if sample_models:
            current_app.logger.info(f"Removing {len(sample_models)} sample models for project {project_id}")
            
            # Delete sample model files
            for model in sample_models:
                if model.path and os.path.exists(model.path):
                    try:
                        os.remove(model.path)
                    except Exception as e:
                        current_app.logger.error(f"Error removing sample model file {model.path}: {str(e)}")
            
            # Delete sample models from database
            Model.query.filter_by(project_id=project_id, is_sample=True).delete()
            db.session.commit()
            return True
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error removing sample models: {str(e)}")
    
    return False

@visualization_bp.route('/project/<int:project_id>')
@login_required
def project_metrics(project_id):
    """Show metrics for a specific project"""
    project = Project.query.get_or_404(project_id)
    if not current_user.is_admin and project.creator_id != current_user.id:
        abort(403)
    
    # Only get real models - no sample models
    models = Model.query.filter_by(project_id=project.id, is_sample=False).order_by(Model.version).all()
    
    if models:
        current_app.logger.info(f"Using {len(models)} real models for visualization")
    else:
        current_app.logger.info("No real models available for visualization")
    
    # Generate dates based on model creation times
    dates = [model.created_at.strftime('%Y-%m-%d') for model in models] if models else []
    
    # Get the FL server metrics if available
    fl_server = current_app.fl_server
    server_metrics = {}
    client_metrics = []
    
    if fl_server and project_id in fl_server.aggregated_metrics:
        server_metrics = fl_server.aggregated_metrics[project_id]
        
        # Add server metrics if newer than the latest model
        if models and server_metrics.get('timestamp'):
            server_timestamp = server_metrics.get('timestamp')
            if isinstance(server_timestamp, str):
                try:
                    server_timestamp = datetime.fromisoformat(server_timestamp)
                except ValueError:
                    server_timestamp = datetime.utcnow()  # Default to now if parsing fails
            
            latest_model_time = models[-1].created_at
            if server_timestamp > latest_model_time:
                # Add server metrics as if it's the next model version
                dates.append(server_timestamp.strftime('%Y-%m-%d'))
        
        # Get client metrics
        for client_id, client_data in fl_server.client_metrics.items():
            if project_id in client_data:
                client_metrics.append({
                    'client_id': client_id,
                    'metrics': client_data[project_id]
                })
    
    # Initialize data structures
    accuracy_data = {
        'global': [model.metrics.get('accuracy', 0.0) for model in models] if models else [],
        'clients': []
    }
    
    loss_data = {
        'global': [model.metrics.get('loss', 0.0) for model in models] if models else [],
        'clients': []
    }
    
    # Add server metrics if they exist and are newer
    if server_metrics and 'accuracy' in server_metrics:
        accuracy_data['global'].append(server_metrics['accuracy'])
        loss_data['global'].append(server_metrics['loss'])
    
    # Add client metrics
    for i, client_metric in enumerate(client_metrics):
        metrics = client_metric['metrics']
        
        # Initialize client arrays if needed
        while len(accuracy_data['clients']) <= i:
            accuracy_data['clients'].append([0.0] * len(dates))
            loss_data['clients'].append([0.0] * len(dates))
        
        # Add client metrics for the latest date
        if dates:
            accuracy_data['clients'][i][-1] = metrics.get('accuracy', 0.0)
            loss_data['clients'][i][-1] = metrics.get('loss', 0.0)
    
    # Get client participation data
    client_counts = [model.clients_count for model in models] if models else []
    
    # Add server client count if available
    if server_metrics and 'clients' in server_metrics:
        client_counts.append(server_metrics['clients'])
    
    participation_data = {
        'dates': dates if dates else [datetime.utcnow().strftime('%Y-%m-%d')],
        'values': client_counts if client_counts else [0]
    }
    
    return render_template('visualization/project_metrics.html',
                         project=project,
                         dates=dates if dates else [datetime.utcnow().strftime('%Y-%m-%d')],
                         accuracy_data=accuracy_data,
                         loss_data=loss_data,
                         participation_data=participation_data)

@visualization_bp.route('/model/<int:project_id>/<int:model_id>')
@login_required
def model_metrics(project_id, model_id):
    """Show metrics for a specific model"""
    project = Project.query.get_or_404(project_id)
    model = Model.query.get_or_404(model_id)
    
    if not current_user.is_admin and project.creator_id != current_user.id:
        abort(403)
    
    # Get model metrics
    metrics = model.metrics or {}
    
    # Get the FL server for more detailed metrics if available
    fl_server = current_app.fl_server
    detailed_metrics = {}
    
    if fl_server and project_id in fl_server.aggregated_metrics:
        detailed_metrics = fl_server.aggregated_metrics[project_id]
    
    # Combine metrics
    training_data = {
        'accuracy': [metrics.get('accuracy', 0)],
        'loss': [metrics.get('loss', 0)]
    }
    
    # Use metrics from the model creation time
    dates = [model.created_at.strftime('%Y-%m-%d %H:%M')]
    
    return render_template('visualization/model_metrics.html',
                         project=project,
                         model=model,
                         dates=dates,
                         training_data=training_data,
                         detailed_metrics=detailed_metrics)

@visualization_bp.route('/api/project/<int:project_id>/metrics')
@login_required
def get_project_metrics(project_id):
    """API endpoint to get project metrics"""
    project = Project.query.get_or_404(project_id)
    if not current_user.is_admin and project.creator_id != current_user.id:
        abort(403)
    
    # Only get real models - no sample models
    models = Model.query.filter_by(project_id=project.id, is_sample=False).order_by(Model.version).all()
    
    if models:
        current_app.logger.info(f"API: Using {len(models)} real models for metrics")
    else:
        current_app.logger.info("API: No real models available for metrics")
    
    # Prepare data for visualization
    versions = []
    accuracy_data = []
    loss_data = []
    client_counts = []
    
    for model in models:
        versions.append(f"v{model.version}")
        if model.metrics:
            accuracy_data.append(model.metrics.get('accuracy', 0))
            loss_data.append(model.metrics.get('loss', 0))
            client_counts.append(model.clients_count)
        else:
            accuracy_data.append(0)
            loss_data.append(0)
            client_counts.append(0)
    
    # Count active clients
    active_clients = ProjectClient.query.filter_by(
        project_id=project_id, 
        status='joined'
    ).count()
    
    return jsonify({
        'versions': versions,
        'accuracy': accuracy_data,
        'loss': loss_data,
        'clients': client_counts,
        'active_clients': active_clients,
        'current_round': project.current_round,
        'total_rounds': project.rounds,
        'status': project.status,
        'using_sample_data': False  # Always using real data now
    })

@visualization_bp.route('/metrics')
@login_required
def metrics():
    """Display training metrics visualization."""
    return render_template('visualization/metrics.html')

@visualization_bp.route('/api/metrics')
@login_required
def get_metrics():
    """Get training metrics data."""
    try:
        # Get the federated learning server instance
        fl_server = current_app.fl_server
        
        if not fl_server:
            return jsonify({"error": "Federated learning server not initialized"}), 500
        
        # Get metrics from the server
        metrics = {
            'rounds': fl_server.current_round,
            'total_rounds': fl_server.rounds,
            'connected_clients': len(fl_server.clients),
            'min_clients': fl_server.min_clients,
            'client_metrics': fl_server.client_metrics,
            'aggregated_metrics': fl_server.aggregated_metrics
        }
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@visualization_bp.route('/model')
@login_required
def model():
    """Display model architecture visualization."""
    return render_template('visualization/model.html')

@visualization_bp.route('/api/model')
@login_required
def get_model_info():
    """Get model architecture information."""
    try:
        # Get the federated learning server instance
        fl_server = current_app.fl_server
        
        if not fl_server:
            return jsonify({"error": "Federated learning server not initialized"}), 500
        
        # Get model summary
        model = fl_server.model
        model_info = {
            'layers': [],
            'total_params': model.count_params(),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }
        
        # Get layer information
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': layer.output_shape,
                'params': layer.count_params()
            }
            model_info['layers'].append(layer_info)
        
        return jsonify(model_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@visualization_bp.route('/data')
@login_required
def data():
    """Display data distribution visualization."""
    return render_template('visualization/data.html')

@visualization_bp.route('/api/data')
@login_required
def get_data_info():
    """Get data distribution information."""
    try:
        # Get the federated learning server instance
        fl_server = current_app.fl_server
        
        if not fl_server:
            return jsonify({"error": "Federated learning server not initialized"}), 500
        
        # Get data distribution information
        data_info = {
            'total_samples': fl_server.total_samples,
            'samples_per_client': fl_server.samples_per_client,
            'client_data_distribution': fl_server.client_data_distribution
        }
        
        return jsonify(data_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500 