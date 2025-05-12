#!/usr/bin/env python3
"""
Fix Model Display Issues

This script:
1. Updates the project_models.html template to correctly access model.metrics instead of attributes
2. Modifies the projects.py file to not create sample models and properly access metrics
"""
import os
import sys
from pathlib import Path

def fix_template():
    """Fix the project_models.html template"""
    template_path = Path('web/templates/dashboard/project_models.html')
    
    if not template_path.exists():
        print(f"Error: Template file not found at {template_path}")
        return False
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Replace direct attribute access with metrics dictionary access
    updated_content = content.replace(
        '<p><strong>Accuracy:</strong> {{ "%.2f"|format(project.global_model.accuracy * 100) }}%</p>',
        '<p><strong>Accuracy:</strong> {{ "%.2f"|format(project.global_model.metrics.get("accuracy", 0) * 100) }}%</p>'
    )
    
    updated_content = updated_content.replace(
        '<p><strong>Loss:</strong> {{ "%.4f"|format(project.global_model.loss) }}</p>',
        '<p><strong>Loss:</strong> {{ "%.4f"|format(project.global_model.metrics.get("loss", 0)) }}</p>'
    )
    
    # Update client model section for metrics access
    updated_content = updated_content.replace(
        '<td>{{ "%.2f"|format(model.accuracy * 100) }}%</td>',
        '<td>{{ "%.2f"|format(model.metrics.get("accuracy", 0) * 100) }}%</td>'
    )
    
    updated_content = updated_content.replace(
        '<td>{{ "%.4f"|format(model.loss) }}</td>',
        '<td>{{ "%.4f"|format(model.metrics.get("loss", 0)) }}</td>'
    )
    
    # Update for status field if it doesn't exist
    updated_content = updated_content.replace(
        '<span class="badge bg-{{ \'success\' if project.global_model.status == \'completed\' else \'warning\' if project.global_model.status == \'training\' else \'secondary\' }}">',
        '<span class="badge bg-{{ \'success\' if project.global_model.is_final else \'secondary\' }}">'
    )
    
    updated_content = updated_content.replace(
        '{{ project.global_model.status }}',
        '{{ "Completed" if project.global_model.is_final else "Training" }}'
    )
    
    with open(template_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated template: {template_path}")
    return True

def fix_projects_py():
    """Modify the projects.py file to not create sample models"""
    projects_py_path = Path('web/routes/projects.py')
    
    if not projects_py_path.exists():
        print(f"Error: Projects route file not found at {projects_py_path}")
        return False
    
    with open(projects_py_path, 'r') as f:
        content = f.read()
    
    # Replace the code that creates sample models
    old_code = """    # If no models exist yet but there's sample data in metrics, create sample models
    if not all_models:
        # Import here to avoid circular imports
        from web.routes.visualization import create_sample_models
        create_sample_models(project)
        # Reload models
        all_models = Model.query.filter_by(project_id=project.id).order_by(Model.version.desc()).all()
        global_model = Model.query.filter_by(project_id=project.id, is_final=True).first()"""
    
    new_code = """    # Don't create sample models - only show real ones
    if not all_models:
        # No models available yet
        global_model = None"""
    
    updated_content = content.replace(old_code, new_code)
    
    with open(projects_py_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated projects route: {projects_py_path}")
    return True

def run_sql_fix():
    """Run SQL commands to directly fix the database"""
    import sqlite3
    
    try:
        # Path to the database file
        db_path = 'instance/app.db'
        
        if not os.path.exists(db_path):
            print(f"Error: Database file not found at {db_path}")
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all model rows
        cursor.execute("SELECT id FROM models")
        model_ids = cursor.fetchall()
        
        if not model_ids:
            print("No models found in the database.")
            conn.close()
            return True
        
        # Check if is_sample column exists
        cursor.execute("PRAGMA table_info(models)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # Add is_sample column if it doesn't exist
        if 'is_sample' not in column_names:
            print("Adding is_sample column to models table...")
            cursor.execute("ALTER TABLE models ADD COLUMN is_sample BOOLEAN DEFAULT 0 NOT NULL")
        
        # Delete any models with sample data
        cursor.execute("DELETE FROM models WHERE is_sample = 1")
        
        # Set all remaining models to not be samples
        cursor.execute("UPDATE models SET is_sample = 0")
        
        conn.commit()
        print(f"Fixed {len(model_ids)} models in the database")
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error fixing database: {str(e)}")
        return False

def main():
    """Main function to run all fixes"""
    print("=== Fixing Federated Learning Model Display Issues ===")
    
    # Fix the template
    template_success = fix_template()
    
    # Fix the projects.py file
    projects_success = fix_projects_py()
    
    # Direct SQL fix for the database
    sql_success = run_sql_fix()
    
    # Print summary
    print("\n=== Results ===")
    print(f"Template fixes: {'SUCCESS' if template_success else 'FAILED'}")
    print(f"Projects route fixes: {'SUCCESS' if projects_success else 'FAILED'}")
    print(f"Database fixes: {'SUCCESS' if sql_success else 'FAILED'}")
    print("\nPlease refresh your browser to see the changes.")

if __name__ == "__main__":
    main() 