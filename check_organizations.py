#!/usr/bin/env python3
"""
Check organization assignments for clients.
"""

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath('.'))

from web.app import create_app, db
from web.models import Client, Organization
from web.config import DevelopmentConfig

# Create app context
app = create_app(DevelopmentConfig)

with app.app_context():
    # Get all organizations
    organizations = Organization.query.all()
    print("\nOrganizations in database:")
    print("-" * 80)
    for org in organizations:
        print(f"ID: {org.id}, Name: {org.name}")
    
    print("\nClients and their organization assignments:")
    print("-" * 80)
    clients = Client.query.all()
    for client in clients:
        org = Organization.query.get(client.organization_id)
        org_name = org.name if org else "NO ORGANIZATION"
        print(f"Client {client.id}: org_id={client.organization_id}, name={client.name}, connected={client.is_connected}, org_name={org_name}") 