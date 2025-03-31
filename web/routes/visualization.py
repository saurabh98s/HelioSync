"""
Visualization Routes

This module provides routes for visualizing training metrics and model performance.
"""

from flask import Blueprint, render_template, jsonify, current_app, abort
from flask_login import login_required, current_user
from web.models import Project, Model
from datetime import datetime, timedelta
import random

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

@visualization_bp.route('/project/<int:project_id>')
@login_required
def project_metrics(project_id):
    """Show metrics for a specific project"""
    project = Project.query.get_or_404(project_id)
    if not current_user.is_admin and project.creator_id != current_user.id:
        abort(403)
    
    # Generate mock metrics data
    rounds = project.rounds
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(rounds)]
    
    # Mock accuracy data
    accuracy_data = {
        'global': [random.uniform(0.7, 0.95) for _ in range(rounds)],
        'clients': [[random.uniform(0.6, 0.9) for _ in range(rounds)] for _ in range(project.min_clients)]
    }
    
    # Mock loss data
    loss_data = {
        'global': [random.uniform(0.1, 0.5) for _ in range(rounds)],
        'clients': [[random.uniform(0.2, 0.6) for _ in range(rounds)] for _ in range(project.min_clients)]
    }
    
    # Mock client participation data
    participation_data = {
        'dates': dates,
        'values': [random.randint(project.min_clients, project.min_clients + 2) for _ in range(rounds)]
    }
    
    return render_template('visualization/project_metrics.html',
                         project=project,
                         dates=dates,
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
    
    # Generate mock training history
    epochs = 10
    dates = [(datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i in range(epochs)]
    
    # Mock training metrics
    training_data = {
        'accuracy': [random.uniform(0.6, 0.95) for _ in range(epochs)],
        'loss': [random.uniform(0.1, 0.5) for _ in range(epochs)],
        'val_accuracy': [random.uniform(0.65, 0.98) for _ in range(epochs)],
        'val_loss': [random.uniform(0.08, 0.45) for _ in range(epochs)]
    }
    
    return render_template('visualization/model_metrics.html',
                         project=project,
                         model=model,
                         dates=dates,
                         training_data=training_data)

@visualization_bp.route('/api/project/<int:project_id>/metrics')
@login_required
def get_project_metrics(project_id):
    """API endpoint to get project metrics"""
    project = Project.query.get_or_404(project_id)
    if not current_user.is_admin and project.creator_id != current_user.id:
        abort(403)
    
    # Generate mock real-time metrics
    return jsonify({
        'accuracy': random.uniform(0.7, 0.95),
        'loss': random.uniform(0.1, 0.5),
        'active_clients': random.randint(project.min_clients, project.min_clients + 2),
        'current_round': project.current_round,
        'total_rounds': project.rounds
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