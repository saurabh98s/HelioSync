#!/usr/bin/env python3
"""
Create model directories needed for the application
"""
import os
from web.app import create_app

def create_model_directories():
    """Ensure all required model directories exist"""
    app = create_app()
    
    with app.app_context():
        # Get the FL_MODEL_PATH from config, default to uploads/models
        model_base_path = app.config.get('FL_MODEL_PATH', 'uploads/models')
        
        # Create the base directory
        os.makedirs(model_base_path, exist_ok=True)
        print(f"Created base model directory: {model_base_path}")
        
        # Create subdirectories
        subdirs = ['downloads', 'models', 'global', 'client']
        for subdir in subdirs:
            path = os.path.join(model_base_path, subdir)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        
        print("All model directories created successfully.")

if __name__ == "__main__":
    create_model_directories() 