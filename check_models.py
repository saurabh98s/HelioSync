#!/usr/bin/env python3
"""
Check models in the database
"""
from web.app import create_app
from web.models import Model, Project

def main():
    app = create_app()
    with app.app_context():
        print("=== Projects ===")
        projects = Project.query.all()
        for project in projects:
            print(f"Project {project.id}: {project.name} ({project.status})")
            print(f"  - Current round: {project.current_round}")
            print(f"  - Total rounds: {project.rounds}")
        
        print("\n=== Models ===")
        models = Model.query.all()
        if not models:
            print("No models found in the database.")
        else:
            for model in models:
                print(f"Model {model.id}:")
                print(f"  - Project: {model.project_id}")
                print(f"  - Version: {model.version}")
                print(f"  - Is Final: {model.is_final}")
                print(f"  - Is Sample: {getattr(model, 'is_sample', 'Field not found')}")
                print(f"  - Metrics: {model.metrics}")
                print(f"  - Path: {model.path}")
                print(f"  - Created: {model.created_at}")
                print("---")

if __name__ == "__main__":
    main() 