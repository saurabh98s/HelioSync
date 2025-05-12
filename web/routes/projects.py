"""
Projects Routes

This module handles routes for managing federated learning projects.
"""

from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, request, abort, send_file, jsonify, current_app
from flask_login import login_required, current_user
import os

from web.app import db
from web.models import Project, Organization, Client, Model, ProjectClient
from web.forms.projects import ProjectForm, ModelDeploymentForm, ClientAssignmentForm
from web.services.fl_manager import start_federated_server
from web.services.model_manager import ModelManager

# Create blueprint
projects_bp = Blueprint('projects', __name__, url_prefix='/projects')

@projects_bp.route('/')
@login_required
def index():
    """List all projects."""
    if current_user.is_admin:
        projects = Project.query.all()
    else:
        projects = Project.query.filter(
            Project.organizations.any(id=current_user.organization_id)
        ).all()
    return render_template('dashboard/projects.html', projects=projects)

@projects_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    """Create a new project."""
    form = ProjectForm()
    if form.validate_on_submit():
        project = Project(
            name=form.name.data,
            description=form.description.data,
            dataset_name=form.dataset_name.data,
            framework=form.framework.data,
            min_clients=form.min_clients.data,
            rounds=form.rounds.data,
            creator_id=current_user.id
        )
        project.organizations.append(current_user.organization)
        db.session.add(project)
        db.session.commit()
        flash('Project created successfully.', 'success')
        return redirect(url_for('projects.view', project_id=project.id))
    return render_template('dashboard/project_form.html', form=form, title='Create Project')

@projects_bp.route('/<int:project_id>')
@login_required
def view(project_id):
    """View project details."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has access to this project
    if not current_user.is_admin and not any(org.id == current_user.organization_id for org in project.organizations):
        flash('You do not have permission to view this project.', 'error')
        return redirect(url_for('projects.index'))
    
    return render_template('dashboard/project.html', project=project)

@projects_bp.route('/<int:project_id>/edit', methods=['GET', 'POST'])
@login_required
def edit(project_id):
    """Edit a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has permission to edit
    if not current_user.is_admin and project.creator_id != current_user.id:
        flash('You do not have permission to edit this project.', 'error')
        return redirect(url_for('projects.index'))
    
    form = ProjectForm(obj=project)
    if form.validate_on_submit():
        project.name = form.name.data
        project.description = form.description.data
        project.dataset_name = form.dataset_name.data
        project.framework = form.framework.data
        project.min_clients = form.min_clients.data
        project.rounds = form.rounds.data
        db.session.commit()
        flash('Project updated successfully.', 'success')
        return redirect(url_for('projects.view', project_id=project.id))
    
    return render_template('dashboard/project_form.html', form=form, title='Edit Project')

@projects_bp.route('/<int:project_id>/start', methods=['POST'])
@login_required
def start(project_id):
    """Start a federated learning project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has access to this project
    if not project.can_access(current_user):
        flash('You do not have permission to start this project.', 'error')
        return redirect(url_for('projects.view', project_id=project_id))
    
    # Check if project is already running
    if project.status == 'running':
        flash('Project is already running.', 'warning')
        return redirect(url_for('projects.view', project_id=project_id))
    
    # Check if project is completed
    if project.status == 'completed':
        flash('Cannot restart a completed project. Create a new project instead.', 'warning')
        return redirect(url_for('projects.view', project_id=project_id))
    
    try:
        # Get the FL server instance
        fl_server = current_app.fl_server
        if not fl_server:
            flash('Federated learning server is not initialized.', 'error')
            return redirect(url_for('projects.view', project_id=project_id))
        
        # Initialize the project in the FL server
        success = fl_server.initialize_project(project)
        if not success:
            flash('Failed to initialize project.', 'error')
            return redirect(url_for('projects.view', project_id=project_id))
        
        # Register all assigned clients with the FL server
        assigned_clients_count = 0
        for project_client in project.project_clients:
            client = project_client.client
            if client.is_active():
                # Add client to FL server project
                if client.client_id not in fl_server.clients:
                    # First register client with the server if not already registered
                    fl_server.register_client(
                        client_id=client.client_id,
                        name=client.name,
                        data_size=client.data_size or 0,
                        device_info=client.device_info or '',
                        platform=client.platform or 'unknown',
                        machine=client.machine or 'unknown',
                        python_version=client.python_version or '3.x'
                    )
                
                # Now add client to this specific project
                fl_server.add_client_to_project(client.client_id, project.id)
                assigned_clients_count += 1
                
                # Update project client status
                project_client.status = 'training'
                
        db.session.commit()
        
        if assigned_clients_count < project.min_clients:
            flash(f'Warning: Only {assigned_clients_count} active clients found, but project requires {project.min_clients}. Training may not start.', 'warning')
        
        # Start the federated server
        success = start_federated_server(project)
        if not success:
            flash('Failed to start federated server.', 'error')
            return redirect(url_for('projects.view', project_id=project_id))
        
        # Update project status
        project.status = 'running'
        project.current_round = 0
        db.session.commit()
        
        flash('Project started successfully.', 'success')
        
    except Exception as e:
        current_app.logger.error(f"Error starting project: {str(e)}")
        flash('An error occurred while starting the project.', 'error')
    
    return redirect(url_for('projects.view', project_id=project_id))

@projects_bp.route('/<int:project_id>/stop', methods=['POST'])
@login_required
def stop(project_id):
    """Stop a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has permission to stop
    if not current_user.is_admin and project.creator_id != current_user.id:
        flash('You do not have permission to stop this project.', 'error')
        return redirect(url_for('projects.index'))
    
    project.status = 'stopped'
    db.session.commit()
    
    flash('Project stopped successfully.', 'success')
    return redirect(url_for('projects.view', project_id=project.id))

@projects_bp.route('/<int:project_id>/clients')
@login_required
def clients(project_id):
    """List clients for a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has access to this project
    if not current_user.is_admin and not any(org.id == current_user.organization_id for org in project.organizations):
        flash('You do not have permission to view this project.', 'error')
        return redirect(url_for('projects.index'))
    
    # Get available clients that are not already assigned to this project
    assigned_client_ids = [pc.client_id for pc in project.project_clients]
    available_clients = Client.query.filter(
        Client.id.notin_(assigned_client_ids)
    ).all()
    
    return render_template('dashboard/project_clients.html', 
                         project=project,
                         available_clients=available_clients)

@projects_bp.route('/<int:project_id>/clients/assign', methods=['POST'])
@login_required
def assign_clients(project_id):
    """Assign clients to a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has access to this project
    if not current_user.is_admin and not any(org.id == current_user.organization_id for org in project.organizations):
        flash('You do not have permission to modify this project.', 'error')
        return redirect(url_for('projects.index'))
    
    client_ids = request.form.getlist('client_ids')
    if not client_ids:
        flash('No clients selected.', 'warning')
        return redirect(url_for('projects.clients', project_id=project.id))
    
    for client_id in client_ids:
        client = Client.query.get(client_id)
        if client and client not in [pc.client for pc in project.project_clients]:
            project_client = ProjectClient(
                project_id=project.id,
                client_id=client.id,
                status='pending'
            )
            db.session.add(project_client)
    
    db.session.commit()
    flash('Clients assigned successfully.', 'success')
    return redirect(url_for('projects.clients', project_id=project.id))

@projects_bp.route('/<int:project_id>/clients/<int:client_id>/remove', methods=['POST'])
@login_required
def remove_client(project_id, client_id):
    """Remove a client from a project."""
    project = Project.query.get_or_404(project_id)
    client = Client.query.get_or_404(client_id)
    
    # Check if user has access to this project
    if not current_user.is_admin and not any(org.id == current_user.organization_id for org in project.organizations):
        flash('You do not have permission to modify this project.', 'error')
        return redirect(url_for('projects.index'))
    
    # Find and remove the project-client association
    project_client = ProjectClient.query.filter_by(
        project_id=project.id,
        client_id=client.id
    ).first()
    
    if project_client:
        db.session.delete(project_client)
        db.session.commit()
        flash('Client removed successfully.', 'success')
    else:
        flash('Client is not assigned to this project.', 'warning')
    
    return redirect(url_for('projects.clients', project_id=project.id))

@projects_bp.route('/<int:project_id>/models')
@login_required
def models(project_id):
    """List models for a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has access to this project
    if not current_user.is_admin and not any(org.id == current_user.organization_id for org in project.organizations):
        flash('You do not have permission to view this project.', 'error')
        return redirect(url_for('projects.index'))
    
    # Get all models for this project
    all_models = Model.query.filter_by(project_id=project.id).order_by(Model.version.desc()).all()
    
    # Get the final (global) model if it exists
    global_model = Model.query.filter_by(project_id=project.id, is_final=True).first()
    
    # Get client models based on project_clients
    client_models = []
    for pc in project.project_clients:
        # Get client model info from projects_client metrics
        if pc.metrics:
            client_model = {
                'client': pc.client,
                'status': pc.status,
                'accuracy': pc.metrics.get('accuracy', 0),
                'loss': pc.metrics.get('loss', 0),
                'updated_at': pc.last_update or datetime.utcnow()
            }
            client_models.append(client_model)
    
    # Don't create sample models - only show real ones
    if not all_models:
        # No models available yet
        global_model = None
    
    # Attach models as properties to project for template access
    project.all_models = all_models
    project.global_model = global_model
    project.client_models = client_models
    
    return render_template('dashboard/project_models.html', project=project)

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

@projects_bp.route('/<int:project_id>/models/<int:model_id>/deploy', methods=['GET', 'POST'])
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
    
    # If this is a GET request, show the deployment form
    if request.method == 'GET':
        return render_template('projects/deploy.html', project=project, model=model, form=form)
    
    # If this is a POST request, process the deployment
    # Get the deployment type from the form
    # We're handling both regular form submissions and direct inputs with hidden fields
    current_app.logger.info(f"Form data: {request.form}")
    
    deploy_type = None
    
    # First check if the form has validated
    if form.validate_on_submit():
        deploy_type = form.deploy_type.data
        current_app.logger.info(f"Form validated: deploy_type = {deploy_type}")
    else:
        # If not, check if the deploy_type is in the request.form directly
        deploy_type = request.form.get('deploy_type')
        current_app.logger.info(f"Form did not validate, using direct deploy_type = {deploy_type}")
        
        # Check for form errors
        if form.errors:
            current_app.logger.warning(f"Form errors: {form.errors}")
    
    # Make sure we have a valid deploy_type
    if not deploy_type or deploy_type not in ['api', 'download', 'huggingface']:
        flash('Invalid deployment type.', 'danger')
        return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))
    
    # Deploy the model
    current_app.logger.info(f"Deploying model {model.id} as {deploy_type}")
    
    if deploy_type == 'huggingface':
        # Deploy to Hugging Face Hub
        result = ModelManager.deploy_to_huggingface(model)
        current_app.logger.info(f"Hugging Face deployment result: {result}")
    else:
        # Deploy using standard methods
        result = ModelManager.deploy_model(model, deploy_type)
        current_app.logger.info(f"Standard deployment result: {result}")
    
    if result.get('success'):
        flash(f'Model deployed successfully as {deploy_type.upper()}.', 'success')
    else:
        flash(f'Failed to deploy model: {result.get("error")}', 'danger')
    
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
    
    # Check if model has a path
    if not model.path or not os.path.exists(model.path):
        flash('Model file not found.', 'warning')
        return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))
    
    # If model is not deployed, deploy it for download
    if not model.is_deployed:
        result = ModelManager.deploy_model(model, deploy_type='download')
        if not result.get('success', False):
            flash(f'Failed to prepare model for download: {result.get("error", "Unknown error")}', 'danger')
            return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))
    
    # If already deployed but not for download, redeploy
    if model.is_deployed and model.deployment_info.get('type') != 'download':
        result = ModelManager.deploy_model(model, deploy_type='download')
        if not result.get('success', False):
            flash(f'Failed to prepare model for download: {result.get("error", "Unknown error")}', 'danger')
            return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))
    
    # Get the download path from deployment info
    download_path = model.deployment_info.get('download_path')
    if not download_path or not os.path.exists(download_path):
        flash('Download file not found.', 'danger')
        return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))
    
    # Get the filename from the path
    filename = os.path.basename(download_path)
    
    # Serve the file
    try:
        return send_file(download_path, as_attachment=True, download_name=filename)
    except Exception as e:
        current_app.logger.error(f"Error sending model file: {str(e)}")
        flash('Error downloading model file.', 'danger')
        return redirect(url_for('projects.view_model', project_id=project.id, model_id=model.id))

@projects_bp.route('/<int:project_id>/delete', methods=['POST'])
@login_required
def delete(project_id):
    """Delete a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has permission to delete
    if not current_user.is_admin and project.creator_id != current_user.id:
        flash('You do not have permission to delete this project.', 'error')
        return redirect(url_for('projects.index'))
    
    # Check if project is running
    if project.status == 'running':
        flash('Cannot delete a running project. Please stop it first.', 'error')
        return redirect(url_for('projects.index'))
    
    try:
        # Delete associated models and project clients
        Model.query.filter_by(project_id=project.id).delete()
        ProjectClient.query.filter_by(project_id=project.id).delete()
        
        # Remove project from organizations
        project.organizations = []
        
        # Delete the project
        db.session.delete(project)
        db.session.commit()
        
        flash('Project deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting project: {str(e)}', 'error')
    
    return redirect(url_for('projects.index')) 