#!/usr/bin/env python3
"""
Add is_sample column to models table
"""
from web.app import create_app
from flask import current_app
import sqlite3

def main():
    app = create_app()
    with app.app_context():
        db_path = current_app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        print(f"Database path: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(models)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if 'is_sample' not in column_names:
            print("Adding is_sample column to models table...")
            cursor.execute("ALTER TABLE models ADD COLUMN is_sample BOOLEAN DEFAULT 0 NOT NULL")
            conn.commit()
            print("Column added successfully!")
        else:
            print("is_sample column already exists")
        
        conn.close()

if __name__ == "__main__":
    main() 