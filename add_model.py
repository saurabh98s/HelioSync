#!/usr/bin/env python3
"""
Add a sample real model to the database
"""
from datetime import datetime, timedelta
from web.app import create_app, db
from web.models import Model, Project

def main():
    app = create_app()
    with app.app_context():
        # Get the MNIST project
        project = Project.query.filter_by(name="MNIST Project").first()
        if not project:
            print("MNIST Project not found")
            return
        
        # Check if we already have models
        existing_models = Model.query.filter_by(project_id=project.id).count()
        next_version = existing_models + 1
        
        # Create a model without using is_sample parameter - higher accuracy for version 2
        model = Model(
            project_id=project.id,
            version=next_version,
            path="/path/to/model/v" + str(next_version),
            metrics={
                "accuracy": 0.9348 + (next_version * 0.01),  # Higher accuracy for each version
                "loss": 0.3093 - (next_version * 0.05),      # Lower loss for each version
                "val_accuracy": 0.9518 + (next_version * 0.005),
                "val_loss": 0.1797 - (next_version * 0.02),
                "round": next_version - 1,
                "clients": next_version + 1 
            },
            created_at=datetime.utcnow() - timedelta(days=10-next_version),  # Different dates to show progression
            is_final=(next_version == 5),  # Version 5 will be final
            clients_count=next_version + 1
        )
        
        # Add the model to the session first
        db.session.add(model)
        
        # Use SQL to set the is_sample field directly
        db.session.flush()
        db.session.execute("UPDATE models SET is_sample = 0 WHERE id = :id", {"id": model.id})
        
        # Commit the transaction
        db.session.commit()
        
        print(f"Added model version {model.version} for project {project.name}")
        print(f"Model ID: {model.id}")

if __name__ == "__main__":
    main() 