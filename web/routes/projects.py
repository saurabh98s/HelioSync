"""
Project Routes

This module provides routes for project management, including creation, viewing, and training control.
"""

from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, request, abort, send_file
from flask_login import login_required, current_user

from web.app import db
from web.models import Project, Organization, Client, Model, ProjectClient
from web.forms.projects import ProjectForm, ModelDeploymentForm, ClientAssignmentForm
from web.services.fl_manager import start_federated_server
from web.services.model_manager import ModelManager

# Create blueprint
projects_bp = Blueprint('projects', __name__)

@projects_bp.route('/')
@login_required
def index():
    """Project list page."""
    # Check if user belongs to an organization
    if not current_user.organization:
        flash('You need to be part of an organization to access projects.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Get all projects for the user's organization
    projects = current_user.organization.projects
    
    return render_template('projects/index.html', projects=projects)

@projects_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    """Create project page."""
    # Check if user belongs to an organization
    if not current_user.organization:
        flash('You need to be part of an organization to create projects.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    form = ProjectForm()
    
    if form.validate_on_submit():
        project = Project(
            name=form.name.data,
            description=form.description.data,
            dataset_name=form.dataset_name.data,
            framework=form.framework.data,
            min_clients=form.min_clients.data,
            rounds=form.rounds.data,
            creator_id=current_user.id,
            created_at=datetime.utcnow()
        )
        
        # Add to user's organization
        project.organizations.append(current_user.organization)
        
        db.session.add(project)
        db.session.commit()
        
        flash(f'Project "{project.name}" created successfully!', 'success')
        return redirect(url_for('projects.view', project_id=project.id))
    
    return render_template('projects/create.html', form=form)

@projects_bp.route('/<int:project_id>')
@login_required
def view(project_id):
    """Project details page."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user's organization is associated with the project
    if current_user.organization not in project.organizations:
        flash('You do not have access to this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    # Eager load related data to avoid lazy loading issues
    # This ensures proper parameter binding in SQL queries
    project = Project.query.options(
        db.joinedload(Project.project_clients),
        db.joinedload(Project.models)
    ).get_or_404(project_id)
    
    # Create forms for the template
    form = ProjectForm(obj=project)
    
    # Get available clients that can be assigned to this project
    available_clients = Client.query.filter_by(
        organization_id=current_user.organization.id,
        is_connected=True
    ).all()
    
    # Debug output to console
    print(f"Found {len(available_clients)} connected clients for assignment")
    for client in available_clients:
        print(f"  Client: {client.name}, ID: {client.id}, UUID: {client.client_id}, Connected: {client.is_connected}")
    
    client_form = ClientAssignmentForm(available_clients=available_clients)
    
    return render_template('projects/view.html', project=project, form=form, client_form=client_form)

@projects_bp.route('/<int:project_id>/start', methods=['POST'])
@login_required
def start(project_id):
    """Start training for a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user's organization is associated with the project
    if current_user.organization not in project.organizations:
        flash('You do not have access to this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    # Check if project is already running
    if project.status == 'running':
        flash('This project is already running.', 'warning')
        return redirect(url_for('projects.view', project_id=project.id))
    
    # Check if there are enough clients
    if project.active_clients_count < project.min_clients:
        flash(f'Not enough active clients. Need at least {project.min_clients} clients.', 'warning')
        return redirect(url_for('projects.view', project_id=project.id))
    
    # Start the federated learning server
    success = start_federated_server(project)
    
    if success:
        flash('Training started successfully!', 'success')
    else:
        flash('Failed to start training. Please check the logs.', 'danger')
    
    return redirect(url_for('projects.view', project_id=project.id))

@projects_bp.route('/<int:project_id>/models/<int:model_id>')
@login_required
def view_model(project_id, model_id):
    """Model details page."""
    project = Project.query.get_or_404(project_id)
    model = Model.query.get_or_404(model_id)
    
    # Check if model belongs to the project
    if model.project_id != project.id:
        flash('Model does not belong to this project.', 'danger')
        return redirect(url_for('projects.view', project_id=project.id))
    
    # Check if user's organization is associated with the project
    if current_user.organization not in project.organizations:
        flash('You do not have access to this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    deployment_form = ModelDeploymentForm()
    
    return render_template('projects/model.html', project=project, model=model, deployment_form=deployment_form)

@projects_bp.route('/<int:project_id>/models/<int:model_id>/deploy', methods=['POST'])
@login_required
def deploy_model(project_id, model_id):
    """Deploy a model as API or for download."""
    project = Project.query.get_or_404(project_id)
    model = Model.query.get_or_404(model_id)
    
    # Check if model belongs to the project
    if model.project_id != project.id:
        flash('Model does not belong to this project.', 'danger')
        return redirect(url_for('projects.view', project_id=project.id))
    
    # Check if user's organization is associated with the project
    if current_user.organization not in project.organizations:
        flash('You do not have access to this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    form = ModelDeploymentForm()
    
    if form.validate_on_submit():
        deploy_type = form.deploy_type.data
        
        # Deploy the model
        result = ModelManager.deploy_model(model, deploy_type)
        
        if result.get('success'):
            flash(f'Model deployed successfully as {deploy_type.upper()}.', 'success')
        else:
            flash(f'Failed to deploy model: {result.get("error")}', 'danger')
        
        return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))
    
    flash('Invalid form submission.', 'danger')
    return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))

@projects_bp.route('/<int:project_id>/models/<int:model_id>/download')
@login_required
def download_model(project_id, model_id):
    """Download a model file."""
    project = Project.query.get_or_404(project_id)
    model = Model.query.get_or_404(model_id)
    
    # Check if model belongs to the project
    if model.project_id != project.id:
        flash('Model does not belong to this project.', 'danger')
        return redirect(url_for('projects.view', project_id=project.id))
    
    # Check if user's organization is associated with the project
    if current_user.organization not in project.organizations:
        flash('You do not have access to this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    # Check if model is deployed for download
    if not model.is_deployed or model.deployment_info.get('type') != 'download':
        flash('Model is not available for download.', 'warning')
        return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))
    
    # Get the file path from deployment info
    file_path = model.deployment_info.get('download_path')
    
    if not file_path:
        flash('Download path not found.', 'danger')
        return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))
    
    # TODO: Implement the actual file download
    # For now, just redirect back
    flash('Model download not implemented yet.', 'info')
    return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))

@projects_bp.route('/<int:project_id>/stop', methods=['POST'])
@login_required
def stop(project_id):
    """Stop training for a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user's organization is associated with the project
    if current_user.organization not in project.organizations:
        flash('You do not have access to this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    # Check if project is actually running
    if project.status != 'running':
        flash('This project is not currently running.', 'warning')
        return redirect(url_for('projects.view', project_id=project.id))
    
    # TODO: Implement logic to stop the federated learning process
    # Update project status
    project.status = 'stopped'
    db.session.commit()
    
    flash('Training stopped successfully.', 'success')
    return redirect(url_for('projects.view', project_id=project.id))

@projects_bp.route('/<int:project_id>/assign_clients', methods=['POST'])
@login_required
def assign_clients(project_id):
    """Assign clients to a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user's organization is associated with the project
    if current_user.organization not in project.organizations:
        flash('You do not have access to this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    # Get available clients
    available_clients = Client.query.filter_by(
        organization_id=current_user.organization.id,
        is_connected=True
    ).all()
    
    # Debug output to console
    print(f"Found {len(available_clients)} connected clients for assignment")
    for client in available_clients:
        print(f"  Client: {client.name}, ID: {client.id}, UUID: {client.client_id}, Connected: {client.is_connected}")
    
    client_form = ClientAssignmentForm(request.form, available_clients=available_clients)
    
    if client_form.validate_on_submit():
        assigned_count = 0
        already_assigned_count = 0
        
        # Process each selected client
        for client_id in client_form.clients.data:
            client = Client.query.get_or_404(client_id)
            
            # Check if client already assigned to this project
            existing = ProjectClient.query.filter_by(
                project_id=project.id, 
                client_id=client.id
            ).first()
            
            if existing:
                already_assigned_count += 1
            else:
                # Create new project-client association
                project_client = ProjectClient(
                    project_id=project.id,
                    client_id=client.id,
                    joined_at=datetime.utcnow()
                )
                db.session.add(project_client)
                assigned_count += 1
        
        if assigned_count > 0:
            db.session.commit()
            flash(f'{assigned_count} clients assigned to project successfully!', 'success')
        
        if already_assigned_count > 0:
            flash(f'{already_assigned_count} clients were already assigned to this project.', 'warning')
    else:
        for field, errors in client_form.errors.items():
            for error in errors:
                flash(f'Error in {getattr(client_form, field).label.text}: {error}', 'danger')
    
    return redirect(url_for('projects.view', project_id=project.id))

@projects_bp.route('/<int:project_id>/edit', methods=['POST'])
@login_required
def edit(project_id):
    """Edit a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user's organization is associated with the project
    if current_user.organization not in project.organizations:
        flash('You do not have access to this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    form = ProjectForm(request.form)
    
    if form.validate_on_submit():
        # Update basic fields
        project.name = form.name.data
        project.description = form.description.data
        project.dataset_name = form.dataset_name.data
        
        # Only update these if project hasn't started yet
        if project.status == 'pending' or project.status == 'created':
            project.framework = form.framework.data
            project.min_clients = form.min_clients.data
            project.rounds = form.rounds.data
        
        db.session.commit()
        flash('Project updated successfully!', 'success')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'Error in {getattr(form, field).label.text}: {error}', 'danger')
    
    return redirect(url_for('projects.view', project_id=project.id)) 