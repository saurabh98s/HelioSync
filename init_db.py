#!/usr/bin/env python3
"""
Database Initialization Script

This script initializes the database for the Federated Learning platform,
creating the necessary tables and an admin user.
"""

import os
import sys
from datetime import datetime
from getpass import getpass
from werkzeug.security import generate_password_hash

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from web.config import Config
from web.app import db, create_app
from web.models import User, Organization, ApiKey

def init_db():
    """Initialize the database with tables and create an admin user."""
    app = create_app(Config)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Check if admin user already exists
        admin = User.query.filter_by(username='admin').first()
        if admin:
            print("Admin user already exists.")
            return
        
        # Create admin user
        print("\n=== Creating Admin User ===")
        password = getpass("Enter admin password: ")
        confirm_password = getpass("Confirm admin password: ")
        
        if password != confirm_password:
            print("Passwords do not match. Aborting.")
            return
        
        admin = User(
            username='admin',
            email='admin@example.com',
            password=generate_password_hash(password),
            is_active=True,
            is_admin=True,
            is_org_admin=True,
            created_at=datetime.utcnow()
        )
        
        # Add and commit the admin user first (without organization)
        db.session.add(admin)
        db.session.commit()
        
        # Create default organization
        print("\n=== Creating Default Organization ===")
        org_name = input("Enter organization name [Federated Learning Admin]: ") or "Federated Learning Admin"
        org_description = input("Enter organization description [System administration organization]: ") or "System administration organization"
        
        org = Organization(
            name=org_name,
            description=org_description,
            created_at=datetime.utcnow(),
            creator_id=admin.id  # Use the ID instead of the object to avoid circular reference
        )
        
        # Add and commit the organization
        db.session.add(org)
        db.session.commit()
        
        # Now update the admin user with the organization reference
        admin.organization_id = org.id
        db.session.commit()
        
        # Generate API key
        api_key = ApiKey(
            key=os.urandom(24).hex(),
            organization_id=org.id,  # Use ID instead of object
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow().replace(year=datetime.utcnow().year + 1)
        )
        
        # Add API key
        db.session.add(api_key)
        db.session.commit()
        
        print("\n=== Database Initialization Complete ===")
        print(f"Admin user created: admin@example.com")
        print(f"Organization created: {org.name}")
        print(f"API Key: {api_key.key}")
        print("\nIMPORTANT: Save this API key as it will not be shown again!")
        print("\nYou can now run the application with: python run.py")

if __name__ == "__main__":
    init_db() 