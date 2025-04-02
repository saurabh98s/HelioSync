#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import create_app, db
from web.models import Project

def update_project_state(project_id, new_state):
    """
    Update the state of a project in the database.
    
    Args:
        project_id (int): The ID of the project to update
        new_state (str): The new state to set ('created', 'running', 'completed', 'failed')
    """
    app = create_app()
    
    with app.app_context():
        try:
            project = Project.query.get(project_id)
            if not project:
                print(f"Error: Project with ID {project_id} not found")
                return False
                
            if new_state not in ['created', 'running', 'completed', 'failed']:
                print(f"Error: Invalid state '{new_state}'. Must be one of: created, running, completed, failed")
                return False
                
            project.status = new_state
            db.session.commit()
            print(f"Successfully updated project {project.name} (ID: {project_id}) to state: {new_state}")
            return True
            
        except Exception as e:
            print(f"Error updating project state: {str(e)}")
            db.session.rollback()
            return False

def list_projects():
    """List all projects and their current states."""
    app = create_app()
    
    with app.app_context():
        projects = Project.query.all()
        if not projects:
            print("No projects found in the database")
            return
            
        print("\nCurrent Projects:")
        print("-" * 80)
        print(f"{'ID':<5} {'Name':<30} {'Status':<15} {'Created At'}")
        print("-" * 80)
        
        for project in projects:
            print(f"{project.id:<5} {project.name[:30]:<30} {project.status:<15} {project.created_at.strftime('%Y-%m-%d %H:%M')}")
        print("-" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  List all projects: python update_project_state.py list")
        print("  Update project state: python update_project_state.py update <project_id> <new_state>")
        print("\nValid states: created, running, completed, failed")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "list":
        list_projects()
    elif command == "update":
        if len(sys.argv) != 4:
            print("Error: Missing arguments")
            print("Usage: python update_project_state.py update <project_id> <new_state>")
            sys.exit(1)
            
        try:
            project_id = int(sys.argv[2])
            new_state = sys.argv[3]
            update_project_state(project_id, new_state)
        except ValueError:
            print("Error: Project ID must be a number")
            sys.exit(1)
    else:
        print(f"Error: Unknown command '{command}'")
        sys.exit(1)

if __name__ == "__main__":
    main() 