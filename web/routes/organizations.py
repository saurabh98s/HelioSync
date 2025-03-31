"""
Organizations Routes

This module handles routes for managing organizations.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from web.models import db, Organization, User, ApiKey
from web.forms import OrganizationForm
import secrets

organizations_bp = Blueprint('organizations', __name__, url_prefix='/organizations')

@organizations_bp.route('/')
@login_required
def index():
    """List organizations."""
    if current_user.is_admin:
        organizations = Organization.query.all()
    else:
        organizations = [current_user.organization]
    return render_template('dashboard/organization.html', organizations=organizations)

@organizations_bp.route('/<int:org_id>')
@login_required
def view(org_id):
    """View organization details."""
    org = Organization.query.get_or_404(org_id)
    
    # Check if user has access to this organization
    if not current_user.is_admin and current_user.organization_id != org.id:
        flash('You do not have permission to view this organization.', 'error')
        return redirect(url_for('organizations.index'))
    
    return render_template('dashboard/organization.html', organization=org)

@organizations_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    """Create a new organization."""
    if not current_user.is_admin:
        flash('You do not have permission to create organizations.', 'error')
        return redirect(url_for('organizations.index'))
    
    form = OrganizationForm()
    if form.validate_on_submit():
        org = Organization(
            name=form.name.data,
            description=form.description.data,
            creator_id=current_user.id
        )
        db.session.add(org)
        db.session.commit()
        flash('Organization created successfully.', 'success')
        return redirect(url_for('organizations.view', org_id=org.id))
    
    return render_template('dashboard/organization_form.html', form=form, title='Create Organization')

@organizations_bp.route('/<int:org_id>/edit', methods=['GET', 'POST'])
@login_required
def edit(org_id):
    """Edit an organization."""
    org = Organization.query.get_or_404(org_id)
    
    # Check if user has permission to edit
    if not current_user.is_admin and not current_user.is_org_admin:
        flash('You do not have permission to edit this organization.', 'error')
        return redirect(url_for('organizations.index'))
    
    form = OrganizationForm(obj=org)
    if form.validate_on_submit():
        org.name = form.name.data
        org.description = form.description.data
        db.session.commit()
        flash('Organization updated successfully.', 'success')
        return redirect(url_for('organizations.view', org_id=org.id))
    
    return render_template('dashboard/organization_form.html', form=form, title='Edit Organization')

@organizations_bp.route('/<int:org_id>/api-keys')
@login_required
def api_keys(org_id):
    """List API keys for an organization."""
    org = Organization.query.get_or_404(org_id)
    
    # Check if user has access to this organization
    if not current_user.is_admin and current_user.organization_id != org.id:
        flash('You do not have permission to view API keys for this organization.', 'error')
        return redirect(url_for('organizations.index'))
    
    return render_template('dashboard/api_keys.html', organization=org)

@organizations_bp.route('/<int:org_id>/api-keys/create', methods=['POST'])
@login_required
def create_api_key(org_id):
    """Create a new API key for an organization."""
    org = Organization.query.get_or_404(org_id)
    
    # Check if user has permission to create API keys
    if not current_user.is_admin and current_user.organization_id != org.id:
        flash('You do not have permission to create API keys for this organization.', 'error')
        return redirect(url_for('organizations.index'))
    
    # Generate a new API key
    key = secrets.token_hex(32)
    api_key = ApiKey(
        key=key,
        organization_id=org.id
    )
    db.session.add(api_key)
    db.session.commit()
    
    # Store the raw key in session for one-time display
    session['new_api_key'] = key
    
    flash('API key created successfully.', 'success')
    return redirect(url_for('organizations.api_keys', org_id=org.id))

@organizations_bp.route('/<int:org_id>/api-keys/<int:key_id>/revoke', methods=['POST'])
@login_required
def revoke_api_key(org_id, key_id):
    """Revoke an API key."""
    org = Organization.query.get_or_404(org_id)
    api_key = ApiKey.query.get_or_404(key_id)
    
    # Check if user has permission to revoke API keys
    if not current_user.is_admin and current_user.organization_id != org.id:
        flash('You do not have permission to revoke API keys for this organization.', 'error')
        return redirect(url_for('organizations.index'))
    
    # Check if key belongs to the organization
    if api_key.organization_id != org.id:
        flash('Invalid API key.', 'error')
        return redirect(url_for('organizations.api_keys', org_id=org.id))
    
    api_key.is_active = False
    db.session.commit()
    
    flash('API key revoked successfully.', 'success')
    return redirect(url_for('organizations.api_keys', org_id=org.id)) 