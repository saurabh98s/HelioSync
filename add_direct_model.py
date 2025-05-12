#!/usr/bin/env python3
"""
Add a model directly to the database.
This avoids the API and connection issues.
"""
import os
import json
import sqlite3
from datetime import datetime, timedelta
import random

# Configuration
PROJECT_ID = 1  # MNIST Project
START_VERSION = 6  # Start version
NUM_MODELS = 3  # Number of models to add
CLIENT_COUNT = 3  # Base number of clients

def main():
    """Add models directly to the database"""
    print("=== Adding Models Directly to Database ===")
    
    # Path to the database file
    db_path = 'instance/app.db'
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return False
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the current max version for this project
    cursor.execute("""
        SELECT MAX(version) FROM models
        WHERE project_id = ?
    """, (PROJECT_ID,))
    
    max_version_row = cursor.fetchone()
    current_max_version = max_version_row[0] if max_version_row and max_version_row[0] else 0
    
    print(f"Current max version: {current_max_version}")
    
    # Use the higher of START_VERSION or current_max_version + 1
    start_at_version = max(START_VERSION, current_max_version + 1)
    
    for i in range(NUM_MODELS):
        version = start_at_version + i
        
        # Generate improving metrics for each model
        base_accuracy = 0.95
        base_loss = 0.12
        
        # Each version gets slightly better
        model_accuracy = min(0.99, base_accuracy + (i * 0.01) + (random.random() * 0.005))
        model_loss = max(0.01, base_loss - (i * 0.02) - (random.random() * 0.01))
        
        # Generate random metrics for the model
        metrics = {
            "accuracy": model_accuracy,
            "loss": model_loss,
            "val_accuracy": min(0.99, model_accuracy + 0.01),
            "val_loss": max(0.01, model_loss - 0.01),
            "round": version - 1,
            "clients": CLIENT_COUNT + i  # More clients join over time
        }
        
        try:
            # Check if the model version already exists
            cursor.execute("""
                SELECT id FROM models
                WHERE project_id = ? AND version = ?
            """, (PROJECT_ID, version))
            
            existing_model = cursor.fetchone()
            
            if existing_model:
                print(f"Model version {version} already exists for project {PROJECT_ID}, skipping")
                continue
            
            # Create a path for the model
            model_path = f"/path/to/model/v{version}"
            
            # The last model should be marked as final
            is_final = (i == NUM_MODELS - 1)
            
            # Create date with progression - newer models are more recent
            # Using space instead of T in datetime format to avoid SQLAlchemy parsing issues
            created_at = (datetime.utcnow() - timedelta(days=NUM_MODELS-i)).strftime('%Y-%m-%d %H:%M:%S.%f')
            
            # Insert the model into the database
            cursor.execute("""
                INSERT INTO models (
                    project_id, version, path, metrics, created_at,
                    is_final, is_deployed, clients_count, is_sample
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                PROJECT_ID,                            # project_id
                version,                               # version
                model_path,                            # path
                json.dumps(metrics),                   # metrics (as JSON string)
                created_at,                            # created_at
                is_final,                              # is_final
                False,                                 # is_deployed
                CLIENT_COUNT + i,                      # clients_count
                False                                  # is_sample
            ))
            
            print(f"Added model version {version} for project {PROJECT_ID}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Clients: {CLIENT_COUNT + i}")
            print(f"  Created: {created_at}")
            print(f"  Is Final: {is_final}")
            print()
            
        except Exception as e:
            print(f"Error adding model version {version}: {str(e)}")
            conn.rollback()
            return False
    
    # Commit all changes at the end
    conn.commit()
    print(f"Successfully added {NUM_MODELS} models to the database")
    
    # Close the connection
    conn.close()
    return True

if __name__ == "__main__":
    main() 