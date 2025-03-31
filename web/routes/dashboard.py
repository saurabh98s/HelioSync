"""
Dashboard Routes

This module provides routes for the dashboard interface.
"""

from flask import Blueprint, render_template, flash, redirect, url_for, request, session
from flask_login import login_required, current_user

from web.app import db
from web.models import User, Organization, Project, Client, Model

# Create blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    """Dashboard home page."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Get organization
    organization = current_user.organization
    
    # Get statistics
    stats = {
        'projects': {
            'total': organization.projects.count(),
            'active': Project.query.join(Project.organizations).filter(
                Organization.id == organization.id,
                Project.status == 'running'
            ).count(),
            'completed': Project.query.join(Project.organizations).filter(
                Organization.id == organization.id,
                Project.status == 'completed'
            ).count()
        },
        'clients': {
            'total': organization.clients.count(),
            'active': Client.query.filter_by(
                organization_id=organization.id,
                status='active'
            ).count()
        },
        'models': {
            'total': Model.query.join(Project).join(Project.organizations).filter(
                Organization.id == organization.id
            ).count(),
            'deployed': Model.query.join(Project).join(Project.organizations).filter(
                Organization.id == organization.id,
                Model.is_deployed == True
            ).count()
        }
    }
    
    # Get recent projects
    recent_projects = Project.query.join(Project.organizations).filter(
        Organization.id == organization.id
    ).order_by(Project.created_at.desc()).limit(5).all()
    
    # Get recent models
    recent_models = Model.query.join(Project).join(Project.organizations).filter(
        Organization.id == organization.id
    ).order_by(Model.created_at.desc()).limit(5).all()
    
    return render_template('dashboard/index.html',
                          organization=organization,
                          stats=stats,
                          recent_projects=recent_projects,
                          recent_models=recent_models)

@dashboard_bp.route('/clients')
@login_required
def clients():
    """Client management page."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Get organization
    organization = current_user.organization
    
    # Get clients
    clients = Client.query.filter_by(organization_id=organization.id).order_by(Client.last_heartbeat.desc()).all()
    
    return render_template('dashboard/clients.html',
                          organization=organization,
                          clients=clients)

@dashboard_bp.route('/client/<client_id>')
@login_required
def client_detail(client_id):
    """Client detail page."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Get organization
    organization = current_user.organization
    
    # Get client
    client = Client.query.filter_by(client_id=client_id, organization_id=organization.id).first_or_404()
    
    return render_template('dashboard/client_detail.html',
                          organization=organization,
                          client=client)

@dashboard_bp.route('/organization')
@login_required
def organization():
    """Organization management page."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Get organization
    organization = current_user.organization
    
    # Get users in organization
    users = User.query.filter_by(organization_id=organization.id).all()
    
    # Get API keys
    api_keys = organization.api_keys
    
    # Check if we need to display the new API key from the session
    new_api_key = None
    if 'new_api_key' in session:
        new_api_key = session.pop('new_api_key')
    
    return render_template('dashboard/organization.html',
                          organization=organization,
                          users=users,
                          api_keys=api_keys,
                          new_api_key=new_api_key)

@dashboard_bp.route('/metrics')
@login_required
def metrics():
    """Metrics and visualizations page."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Get organization
    organization = current_user.organization
    
    # Get all projects for the organization
    projects = Project.query.join(Project.organizations).filter(
        Organization.id == organization.id
    ).order_by(Project.created_at.desc()).all()
    
    # Get models for each project
    project_data = []
    for project in projects:
        models = Model.query.filter_by(project_id=project.id).order_by(Model.created_at).all()
        
        # Prepare data for charts
        accuracy_data = [{'x': i+1, 'y': model.accuracy} for i, model in enumerate(models)]
        loss_data = [{'x': i+1, 'y': model.loss} for i, model in enumerate(models)]
        
        project_data.append({
            'project': project,
            'models': models,
            'accuracy_data': accuracy_data,
            'loss_data': loss_data
        })
    
    return render_template('dashboard/metrics.html',
                          organization=organization,
                          project_data=project_data) 