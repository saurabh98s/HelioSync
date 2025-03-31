#!/usr/bin/env python3
"""
Fix client organization assignments.
"""

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath('.'))

from web.app import create_app, db
from web.models import Client, Organization, User
from web.config import DevelopmentConfig

# Create app context
app = create_app(DevelopmentConfig)

with app.app_context():
    # Get the SJSU organization
    org = Organization.query.filter_by(name='SJSU').first()
    if not org:
        print("Error: Could not find SJSU organization")
        sys.exit(1)
    
    print(f"\nFound organization: {org.name} (ID: {org.id})")
    
    # Get all clients
    clients = Client.query.all()
    print(f"\nFound {len(clients)} clients")
    
    # Update each client's organization
    updated = 0
    for client in clients:
        if client.organization_id != org.id:
            print(f"Updating client {client.id} ({client.name}) from org {client.organization_id} to {org.id}")
            client.organization_id = org.id
            db.session.add(client)
            updated += 1
    
    if updated > 0:
        db.session.commit()
        print(f"\nUpdated {updated} clients to organization {org.name}")
    else:
        print("\nNo clients needed updating") 