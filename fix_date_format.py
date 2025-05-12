#!/usr/bin/env python3
"""
Fix the datetime format in the models table
"""
import sqlite3
import os
from datetime import datetime

def main():
    """Fix datetime format in models table"""
    print("=== Fixing DateTime Format in Models Table ===")
    
    # Path to the database file
    db_path = 'instance/app.db'
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return False
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all models
    cursor.execute("SELECT id, created_at FROM models")
    models = cursor.fetchall()
    
    if not models:
        print("No models found in the database.")
        return True
    
    print(f"Found {len(models)} models to fix")
    fixed_count = 0
    
    for model_id, created_at in models:
        try:
            # Try to parse the date
            if created_at and 'T' in created_at:
                # The problematic format is like '2025-04-02T10:23:43.923070'
                # Convert to SQLite-compatible format: '2025-04-02 10:23:43.923070'
                new_format = created_at.replace('T', ' ')
                
                # Update the database
                cursor.execute(
                    "UPDATE models SET created_at = ? WHERE id = ?",
                    (new_format, model_id)
                )
                fixed_count += 1
                print(f"Fixed model ID {model_id}: {created_at} -> {new_format}")
        except Exception as e:
            print(f"Error fixing model {model_id}: {str(e)}")
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print(f"Successfully fixed {fixed_count} of {len(models)} models")
    return True

if __name__ == "__main__":
    main() 