#!/usr/bin/env python3
"""
Update Database Schema

This script updates the database schema to include any new fields.
"""

from web.app import create_app, db

def update_schema():
    """Update database schema."""
    app = create_app()
    with app.app_context():
        db.create_all()
        print("Database schema updated successfully.")

if __name__ == "__main__":
    update_schema() 