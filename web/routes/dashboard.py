"""
Dashboard Routes

This module handles routes for the main dashboard.
"""

from flask import Blueprint, render_template, jsonify, flash, redirect, url_for, request, session
from flask_login import login_required, current_user
from flask_wtf import FlaskForm

from web.app import db
from web.models import User, Organization, Project, Client, Model, ApiKey
from sqlalchemy import func

# Create blueprint
dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@dashboard_bp.route('/')
@login_required
def index():
    """Display the main dashboard."""
    # Get user's organization
    org = current_user.organization
    
    # Calculate statistics
    stats = {
        'projects': {
            'total': Project.query.join(Project.organizations).filter(Organization.id == org.id).count(),
            'active': Project.query.join(Project.organizations).filter(Organization.id == org.id, Project.status == 'active').count(),
            'completed': Project.query.join(Project.organizations).filter(Organization.id == org.id, Project.status == 'completed').count()
        },
        'clients': {
            'total': Client.query.filter_by(organization_id=org.id).count(),
            'active': Client.query.filter_by(organization_id=org.id, is_connected=True).count(),
            'inactive': Client.query.filter_by(organization_id=org.id, is_connected=False).count()
        },
        'models': {
            'total': Model.query.join(Project).join(Project.organizations).filter(Organization.id == org.id).count(),
            'trained': Model.query.join(Project).join(Project.organizations).filter(Organization.id == org.id, Model.is_final == True).count(),
            'training': Model.query.join(Project).join(Project.organizations).filter(Organization.id == org.id, Model.is_final == False).count()
        }
    }
    
    # Get recent projects
    recent_projects = Project.query.join(Project.organizations)\
        .filter(Organization.id == org.id)\
        .order_by(Project.created_at.desc())\
        .limit(5)\
        .all()
    
    # Get recent clients
    recent_clients = Client.query.filter_by(organization_id=org.id)\
        .order_by(Client.created_at.desc())\
        .limit(5)\
        .all()
    
    return render_template('dashboard/index.html',
                         stats=stats,
                         recent_projects=recent_projects,
                         recent_clients=recent_clients)

@dashboard_bp.route('/stats')
@login_required
def get_stats():
    """Get dashboard statistics."""
    # Get user's organization
    org = current_user.organization
    
    # Calculate statistics
    stats = {
        'projects': {
            'total': Project.query.join(Project.organizations).filter(Organization.id == org.id).count(),
            'active': Project.query.join(Project.organizations).filter(Organization.id == org.id, Project.status == 'active').count(),
            'completed': Project.query.join(Project.organizations).filter(Organization.id == org.id, Project.status == 'completed').count()
        },
        'clients': {
            'total': Client.query.filter_by(organization_id=org.id).count(),
            'active': Client.query.filter_by(organization_id=org.id, is_connected=True).count(),
            'inactive': Client.query.filter_by(organization_id=org.id, is_connected=False).count()
        },
        'models': {
            'total': Model.query.join(Project).join(Project.organizations).filter(Organization.id == org.id).count(),
            'trained': Model.query.join(Project).join(Project.organizations).filter(Organization.id == org.id, Model.is_final == True).count(),
            'training': Model.query.join(Project).join(Project.organizations).filter(Organization.id == org.id, Model.is_final == False).count()
        }
    }
    
    return jsonify(stats)

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
    
    # Create form for CSRF token
    form = FlaskForm()
    
    return render_template('dashboard/organization.html',
                          organization=organization,
                          users=users,
                          api_keys=api_keys,
                          new_api_key=new_api_key,
                          form=form)

@dashboard_bp.route('/create_api_key', methods=['POST'])
@login_required
def create_api_key():
    """Create a new API key for the organization."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Validate CSRF token
    form = FlaskForm()
    if not form.validate_on_submit():
        flash('Invalid request. Please try again.', 'error')
        return redirect(url_for('dashboard.organization'))
    
    # Get organization
    organization = current_user.organization
    
    try:
        # Generate a new API key
        import secrets
        from datetime import datetime, timedelta
        
        # Generate a random key (this will be the actual key used)
        api_key_value = secrets.token_hex(32)
        
        # Check if expiration date is set
        if request.form.get('set_expiration') == '1' and request.form.get('expiration_date'):
            try:
                expiration_date = datetime.strptime(request.form.get('expiration_date'), '%Y-%m-%d')
            except ValueError:
                flash('Invalid expiration date format.', 'error')
                return redirect(url_for('dashboard.organization'))
        else:
            # Default expiration is 1 year from now
            expiration_date = datetime.utcnow() + timedelta(days=365)
        
        # Create new API key
        api_key = ApiKey(
            key=api_key_value,  # Store the raw key directly
            organization=organization,
            created_at=datetime.utcnow(),
            expires_at=expiration_date,
            is_active=True
        )
        
        # Save to database
        db.session.add(api_key)
        db.session.commit()
        
        # Store the key in session to display it once
        session['new_api_key'] = api_key_value
        
        flash('New API key created successfully.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating API key: {str(e)}', 'error')
    
    return redirect(url_for('dashboard.organization'))

@dashboard_bp.route('/revoke_api_key', methods=['POST'])
@login_required
def revoke_api_key():
    """Revoke an existing API key."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Validate CSRF token
    form = FlaskForm()
    if not form.validate_on_submit():
        flash('Invalid request. Please try again.', 'error')
        return redirect(url_for('dashboard.organization'))
    
    try:
        # Get organization
        organization = current_user.organization
        
        # Get key ID from form
        key_id = request.form.get('key_id')
        if not key_id:
            flash('Invalid request.', 'error')
            return redirect(url_for('dashboard.organization'))
        
        # Find the key
        api_key = ApiKey.query.filter_by(id=key_id, organization_id=organization.id).first()
        if not api_key:
            flash('API key not found.', 'error')
            return redirect(url_for('dashboard.organization'))
        
        # Revoke the key
        api_key.is_active = False
        db.session.commit()
        
        flash('API key revoked successfully.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error revoking API key: {str(e)}', 'error')
    
    return redirect(url_for('dashboard.organization'))

@dashboard_bp.route('/update_organization', methods=['POST'])
@login_required
def update_organization():
    """Update organization details."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Validate CSRF token
    form = FlaskForm()
    if not form.validate_on_submit():
        flash('Invalid request. Please try again.', 'error')
        return redirect(url_for('dashboard.organization'))
    
    # Get organization
    organization = current_user.organization
    
    # Update organization details
    organization.name = request.form.get('name', organization.name)
    organization.description = request.form.get('description', organization.description)
    
    db.session.commit()
    
    flash('Organization updated successfully.', 'success')
    return redirect(url_for('dashboard.organization'))

@dashboard_bp.route('/invite_user', methods=['POST'])
@login_required
def invite_user():
    """Invite a user to the organization."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Get organization
    organization = current_user.organization
    
    # Get email from form
    email = request.form.get('email')
    if not email:
        flash('Email address is required.', 'error')
        return redirect(url_for('dashboard.organization'))
    
    # Check if user already exists
    user = User.query.filter_by(email=email).first()
    if user:
        # If user already belongs to this organization
        if user.organization_id == organization.id:
            flash('User is already a member of this organization.', 'warning')
            return redirect(url_for('dashboard.organization'))
        
        # TODO: Send invitation email
        flash(f'Invitation sent to {email}.', 'success')
    else:
        # TODO: Create invitation record and send email
        flash(f'Invitation sent to {email}.', 'success')
    
    return redirect(url_for('dashboard.organization'))

@dashboard_bp.route('/remove_user', methods=['POST'])
@login_required
def remove_user():
    """Remove a user from the organization."""
    # If user doesn't belong to an organization, redirect to create organization page
    if not current_user.organization:
        flash('You need to create or join an organization first.', 'warning')
        return redirect(url_for('auth.create_org'))
    
    # Get organization
    organization = current_user.organization
    
    # Get user ID from form
    user_id = request.form.get('user_id')
    if not user_id:
        flash('Invalid request.', 'error')
        return redirect(url_for('dashboard.organization'))
    
    # Cannot remove yourself
    if int(user_id) == current_user.id:
        flash('You cannot remove yourself from the organization.', 'error')
        return redirect(url_for('dashboard.organization'))
    
    # Find the user
    user = User.query.filter_by(id=user_id, organization_id=organization.id).first()
    if not user:
        flash('User not found in organization.', 'error')
        return redirect(url_for('dashboard.organization'))
    
    # Remove the user from the organization
    user.organization_id = None
    db.session.commit()
    
    flash(f'User {user.username} removed from the organization.', 'success')
    return redirect(url_for('dashboard.organization'))

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
        accuracy_data = [{'x': i+1, 'y': model.metrics.get('accuracy', 0) if model.metrics else 0} for i, model in enumerate(models)]
        loss_data = [{'x': i+1, 'y': model.metrics.get('loss', 0) if model.metrics else 0} for i, model in enumerate(models)]
        
        project_data.append({
            'project': project,
            'models': models,
            'accuracy_data': accuracy_data,
            'loss_data': loss_data
        })
    
    return render_template('dashboard/metrics.html',
                          organization=organization,
                          project_data=project_data) 