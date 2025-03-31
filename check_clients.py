#!/usr/bin/env python3
"""
Check and fix client connections in the database.

This script:
1. Lists all clients in the database
2. Sets is_connected=True for all active clients
3. Shows which clients should be visible in the assignment dialog
"""

from datetime import datetime, timedelta
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
    # Get current time
    now = datetime.utcnow()
    
    # Get all clients
    clients = Client.query.all()
    
    print(f"Found {len(clients)} clients in the database:")
    print("-" * 80)
    print(f"{'ID':<5} {'Name':<30} {'Connected':<10} {'Active':<10} {'Last Heartbeat':<25}")
    print("-" * 80)
    
    # Check each client
    for client in clients:
        # Check if client is active (heartbeat in last 5 minutes)
        is_active = False
        last_heartbeat = "Never"
        
        if client.last_heartbeat:
            time_diff = (now - client.last_heartbeat).total_seconds()
            is_active = time_diff < 300  # 5 minutes
            last_heartbeat = client.last_heartbeat.strftime("%Y-%m-%d %H:%M:%S")
            
            # Update is_connected status to match activity
            if client.is_connected != is_active:
                client.is_connected = is_active
                db.session.add(client)
        
        print(f"{client.id:<5} {client.name:<30} {str(client.is_connected):<10} {str(is_active):<10} {last_heartbeat:<25}")
    
    # Commit any changes
    db.session.commit()
    
    # Now show which clients should be visible for assignment
    print("\nClients available for assignment (connected and active):")
    print("-" * 80)
    
    active_clients = Client.query.filter_by(is_connected=True).all()
    if active_clients:
        for client in active_clients:
            print(f"ID: {client.id}, Name: {client.name}, UUID: {client.client_id}")
    else:
        print("No active clients found that can be assigned to projects.")
    
    print("\nTo fix this issue:")
    print("1. Make sure your client is running and sending heartbeats")
    print("2. Check that it has a recent last_heartbeat value (within last 5 minutes)")
    print("3. Verify the is_connected flag is set to True")
    print("4. Refresh the project page and try assigning clients again") 