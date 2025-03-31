"""
Script to update the database schema.
"""

from web.app import create_app, db
from web.models import Client
import sqlite3

def column_exists(conn, table, column):
    """Check if a column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cursor.fetchall()]
    return column in columns

def update_database():
    """Update the database schema."""
    app = create_app()
    with app.app_context():
        # Get database path from app config
        db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Add new columns if they don't exist
            new_columns = [
                ('platform', 'VARCHAR(50)'),
                ('machine', 'VARCHAR(50)'),
                ('python_version', 'VARCHAR(20)'),
                ('data_size', 'INTEGER DEFAULT 0')
            ]
            
            for column, dtype in new_columns:
                if not column_exists(conn, 'clients', column):
                    cursor.execute(f"ALTER TABLE clients ADD COLUMN {column} {dtype}")
            
            conn.commit()
            print("Database schema updated successfully!")
            
        except Exception as e:
            print(f"Error updating database: {str(e)}")
            conn.rollback()
        finally:
            conn.close()

if __name__ == '__main__':
    update_database() 