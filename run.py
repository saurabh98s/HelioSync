#!/usr/bin/env python3
"""
Run Script

This script runs the Flask application.
"""

from web.app import create_app

if __name__ == '__main__':
    app = create_app('development')
    app.run(debug=True) 