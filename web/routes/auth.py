"""
Authentication Routes

This module provides routes for user authentication and organization management.
"""

import os
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, request, session, current_app
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash

from web.models import User, Organization, ApiKey
from web.forms.auth import LoginForm, RegistrationForm, OrganizationForm
from web.app import db

# Create blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'error')
            return redirect(url_for('auth.login'))
        
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('dashboard.index')
        return redirect(next_page)
    
    return render_template('auth/login.html', form=form)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    # Redirect if user is already logged in
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    form = RegistrationForm()
    
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=generate_password_hash(form.password.data),
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@auth_bp.route('/create-organization', methods=['GET', 'POST'])
@login_required
def create_org():
    """Create organization page."""
    # Redirect if user already belongs to an organization
    if current_user.organization:
        flash('You already belong to an organization.', 'info')
        return redirect(url_for('dashboard.organization'))
    
    form = OrganizationForm()
    
    if form.validate_on_submit():
        # Step 1: Save current_user ID before creating organization
        user_id = current_user.id
        
        # Step 2: Create the organization with reference to user ID, not the object
        organization = Organization(
            name=form.name.data,
            description=form.description.data,
            creator_id=user_id,  # Use ID instead of object reference
            created_at=datetime.utcnow()
        )
        
        db.session.add(organization)
        db.session.commit()
        
        # Step 3: Update user's organization after organization is created
        current_user.organization_id = organization.id
        db.session.commit()
        
        flash('Organization created successfully!', 'success')
        return redirect(url_for('dashboard.organization'))
    
    return render_template('auth/create_organization.html', form=form)

@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page."""
    return render_template('auth/profile.html', user=current_user) 