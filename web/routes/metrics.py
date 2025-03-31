"""
Metrics Routes

This module handles routes for displaying training metrics and model performance.
"""

from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required, current_user
from web.models import Project, ProjectClient, Model

metrics_bp = Blueprint('metrics', __name__, url_prefix='/metrics')

@metrics_bp.route('/')
@login_required
def index():
    """Display metrics dashboard."""
    # Get projects for the current user's organization
    if current_user.is_admin:
        projects = Project.query.all()
    else:
        projects = Project.query.filter(
            Project.organizations.any(id=current_user.organization_id)
        ).all()
    
    # Prepare data for each project
    project_data = []
    for project in projects:
        data = {
            'project': project,
            'models': project.models,
            'accuracy_data': [],
            'loss_data': []
        }
        
        # Get metrics from models
        for model in project.models:
            if model.metrics:
                if 'accuracy' in model.metrics:
                    data['accuracy_data'].append({
                        'x': model.version,
                        'y': model.metrics['accuracy']
                    })
                if 'loss' in model.metrics:
                    data['loss_data'].append({
                        'x': model.version,
                        'y': model.metrics['loss']
                    })
        
        project_data.append(data)
    
    return render_template('dashboard/metrics.html', project_data=project_data)

@metrics_bp.route('/api/project/<int:project_id>')
@login_required
def project_metrics(project_id):
    """Get metrics for a specific project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has access to this project
    if not current_user.is_admin and not any(org.id == current_user.organization_id for org in project.organizations):
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Get metrics from models
    metrics = []
    for model in project.models:
        if model.metrics:
            metrics.append({
                'version': model.version,
                'metrics': model.metrics,
                'created_at': model.created_at.isoformat(),
                'clients_count': model.clients_count
            })
    
    return jsonify({
        'project_id': project.id,
        'name': project.name,
        'metrics': metrics
    }) 