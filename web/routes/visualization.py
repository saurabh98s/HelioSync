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
    """Create sample models for visualization if none exist."""
    if not project.models:
        from datetime import datetime, timedelta
        
        # Create 5 sample models with increasing accuracy
        for i in range(5):
            model = Model(
                project_id=project.id,
                version=i + 1,
                metrics={
                    'accuracy': 0.5 + (i * 0.1),  # Accuracy increases from 0.5 to 0.9
                    'loss': 1.0 - (i * 0.15),     # Loss decreases from 1.0 to 0.25
                    'round': i + 1
                },
                clients_count=project.min_clients,
                created_at=datetime.now() - timedelta(days=i)
            )
            db.session.add(model)
        
        try:
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error creating sample models: {str(e)}")
            return False
    return False

@visualization_bp.route('/project/<int:project_id>')
@login_required
def project_metrics(project_id):
    """Show metrics for a specific project"""
    project = Project.query.get_or_404(project_id)
    if not current_user.is_admin and project.creator_id != current_user.id:
        abort(403)
    
    # Create sample models if none exist
    create_sample_models(project)
    
    # Get real metrics data from models
    models = Model.query.filter_by(project_id=project.id).order_by(Model.version).all()
    
    # Generate dates based on model creation times
    dates = [model.created_at.strftime('%Y-%m-%d') for model in models] if models else []
    
    # Initialize data structures with default values
    accuracy_data = {
        'global': [0.0] if not models else [model.metrics.get('accuracy', 0.0) for model in models],
        'clients': [[0.0] * len(dates)] * project.min_clients  # Initialize client data with zeros
    }
    
    loss_data = {
        'global': [0.0] if not models else [model.metrics.get('loss', 0.0) for model in models],
        'clients': [[0.0] * len(dates)] * project.min_clients  # Initialize client data with zeros
    }
    
    # Get client participation data
    client_counts = [model.clients_count for model in models] if models else [0]
    participation_data = {
        'dates': dates if dates else [datetime.now().strftime('%Y-%m-%d')],
        'values': client_counts
    }
    
    # If we don't have any models yet, show message
    if not models:
        flash('No models available for this project yet.', 'info')
    
    return render_template('visualization/project_metrics.html',
                         project=project,
                         dates=dates if dates else [datetime.now().strftime('%Y-%m-%d')],
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
    
    # Get all models for this project to show progression
    models = Model.query.filter_by(project_id=project_id).order_by(Model.version).all()
    
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
        'status': project.status
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