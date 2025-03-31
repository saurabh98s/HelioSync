#!/usr/bin/env python3
"""
Flask Application

This module initializes the Flask application and its extensions.
"""

import os
from datetime import datetime
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config)
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    
    # Configure login
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    
    # Register blueprints
    from web.routes.auth import auth_bp
    from web.routes.dashboard import dashboard_bp
    from web.routes.projects import projects_bp
    from web.routes.api import api_bp
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(projects_bp, url_prefix='/projects')
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Register error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('errors/500.html'), 500
    
    # Register Jinja filters
    @app.template_filter('formatdatetime')
    def format_datetime(value, format='%Y-%m-%d %H:%M:%S'):
        if value is None:
            return ""
        return value.strftime(format)
    
    # Index route
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Add context processor for templates
    @app.context_processor
    def inject_now():
        return {'now': datetime.utcnow()}
    
    return app

@login_manager.user_loader
def load_user(user_id):
    """Load a user given the user ID."""
    return User.query.get(int(user_id)) 