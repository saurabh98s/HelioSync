"""
Authentication Service

This module manages user authentication and organization management.
"""

import os
import sys
import uuid
import hashlib
from datetime import datetime, timedelta
from flask import current_app, session
from werkzeug.security import generate_password_hash, check_password_hash

# Add parent directory to path so we can import from the main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import Flask app and models
from web.app import db
from web.models import User, Organization, ApiKey

def register_user(username, email, password, organization_name=None, organization_description=None):
    """Register a new user and optionally create an organization.
    
    Args:
        username: The username.
        email: The email address.
        password: The password.
        organization_name: Optional organization name.
        organization_description: Optional organization description.
        
    Returns:
        User: The created user object or None if failed.
    """
    try:
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            return None, "Username already exists"
        
        if User.query.filter_by(email=email).first():
            return None, "Email already exists"
        
        # Create user
        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password),
            created_at=datetime.utcnow()
        )
        
        db.session.add(user)
        
        # Create organization if provided
        if organization_name:
            # Check if organization name is already taken
            if Organization.query.filter_by(name=organization_name).first():
                return None, "Organization name already exists"
            
            organization = Organization(
                name=organization_name,
                description=organization_description or "",
                created_at=datetime.utcnow(),
                creator=user
            )
            
            db.session.add(organization)
            user.organization = organization
            
            # Create an API key for the organization
            create_api_key(organization)
        
        db.session.commit()
        return user, "User registered successfully"
    
    except Exception as e:
        db.session.rollback()
        return None, f"Error registering user: {str(e)}"

def authenticate_user(username, password):
    """Authenticate a user.
    
    Args:
        username: The username.
        password: The password.
        
    Returns:
        User: The user object if authentication succeeds, None otherwise.
    """
    user = User.query.filter_by(username=username).first()
    
    if user and check_password_hash(user.password, password):
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        return user
    
    return None

def create_organization(name, description, user_id):
    """Create a new organization.
    
    Args:
        name: The organization name.
        description: The organization description.
        user_id: The ID of the user creating the organization.
        
    Returns:
        Organization: The created organization object or None if failed.
    """
    try:
        user = User.query.get(user_id)
        if not user:
            return None, "User not found"
        
        # Check if user already belongs to an organization
        if user.organization_id:
            return None, "User already belongs to an organization"
        
        # Check if organization name is already taken
        if Organization.query.filter_by(name=name).first():
            return None, "Organization name already exists"
        
        organization = Organization(
            name=name,
            description=description or "",
            created_at=datetime.utcnow(),
            creator=user
        )
        
        db.session.add(organization)
        user.organization = organization
        
        # Create an API key for the organization
        create_api_key(organization)
        
        db.session.commit()
        return organization, "Organization created successfully"
    
    except Exception as e:
        db.session.rollback()
        return None, f"Error creating organization: {str(e)}"

def create_api_key(organization):
    """Create a new API key for an organization.
    
    Args:
        organization: The organization object.
        
    Returns:
        ApiKey: The created API key object.
    """
    # Generate a random API key
    api_key = f"{organization.id}_{uuid.uuid4().hex}"
    hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Create API key object
    api_key_obj = ApiKey(
        key=hashed_key,
        organization=organization,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=365)  # 1 year expiry
    )
    
    db.session.add(api_key_obj)
    
    # Return the plain API key for one-time display to the user
    return api_key

def validate_api_key(api_key):
    """Validate an API key.
    
    Args:
        api_key: The API key to validate.
        
    Returns:
        Organization: The organization if the key is valid, None otherwise.
    """
    if not api_key:
        return None
    
    # Hash the API key for comparison
    hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Find the API key in the database
    api_key_obj = ApiKey.query.filter_by(key=hashed_key).first()
    
    if api_key_obj and api_key_obj.expires_at >= datetime.utcnow() and not api_key_obj.revoked:
        return api_key_obj.organization
    
    return None

def add_user_to_organization(user_id, organization_id, admin=False):
    """Add a user to an organization.
    
    Args:
        user_id: The ID of the user.
        organization_id: The ID of the organization.
        admin: Whether the user should be an admin.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        user = User.query.get(user_id)
        organization = Organization.query.get(organization_id)
        
        if not user or not organization:
            return False, "User or organization not found"
        
        # Check if user already belongs to an organization
        if user.organization_id:
            return False, "User already belongs to an organization"
        
        user.organization = organization
        user.is_org_admin = admin
        
        db.session.commit()
        return True, "User added to organization successfully"
    
    except Exception as e:
        db.session.rollback()
        return False, f"Error adding user to organization: {str(e)}" 