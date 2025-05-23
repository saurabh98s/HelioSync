#!/usr/bin/env python3
"""
Flask Application

This module initializes the Flask application and its extensions.
"""

import os
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from flask_login import login_required, current_user

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import extensions
from web.extensions import db, login_manager, migrate, csrf

# Import models
from web.models import User, Organization

def init_db(app):
    """Initialize the database."""
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Check if admin user exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            # Create default organization
            org = Organization.query.filter_by(name='SJSU').first()
            if not org:
                org = Organization(name='SJSU')
                db.session.add(org)
                db.session.commit()
            
            # Create admin user
            admin = User(
                username='admin',
                email='admin@example.com',
                is_active=True,
                is_admin=True,
                is_org_admin=True,
                organization_id=org.id
            )
            admin.set_password('admin')  # This will hash the password
            db.session.add(admin)
            db.session.commit()
            print('Created admin user')

def init_fl_server(app):
    """Initialize the federated learning server."""
    try:
        # Import here to avoid circular imports
        from web.services.fl_manager import FederatedLearningServer
        
        # Create the server instance
        fl_server = FederatedLearningServer()
        
        # Store it both as app.fl_server and in config for backward compatibility
        app.fl_server = fl_server
        app.config['FL_SERVER'] = fl_server
        
        # Also set FL_MANAGER for easier access from routes
        app.config['FL_MANAGER'] = fl_server
        
        app.logger.info("Federated Learning Server initialized successfully")
    except Exception as e:
        app.logger.error(f"Error initializing Federated Learning Server: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        
        # Set to None to avoid attribute errors
        app.fl_server = None
        app.config['FL_SERVER'] = None

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    
    # Default configuration
    app.config.update(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(app.instance_path, 'app.db'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=os.path.join(app.instance_path, 'uploads'),
        FL_MODEL_PATH=os.path.join(app.instance_path, 'models'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    )

    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['FL_MODEL_PATH'], exist_ok=True)
    except OSError:
        pass

    # Load configuration from the config.env file through our Config class
    from web.config import Config
    app.config.from_object(Config)
    
    # Set HUGGINGFACE_TOKEN in the config if available in environment
    huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')
    if huggingface_token:
        app.config['HUGGINGFACE_TOKEN'] = huggingface_token
        app.logger.info("Hugging Face token loaded from environment")

    # Load additional configuration if provided
    if config is not None:
        if isinstance(config, str):
            # Load config from config file
            if config == 'development':
                app.config.update(
                    DEBUG=True,
                    TESTING=False,
                    SECRET_KEY='dev-key-please-change',
                )
            elif config == 'testing':
                app.config.update(
                    DEBUG=False,
                    TESTING=True,
                    SECRET_KEY='test-key',
                    SQLALCHEMY_DATABASE_URI='sqlite:///:memory:'
                )
            elif config == 'production':
                app.config.update(
                    DEBUG=False,
                    TESTING=False,
                    SECRET_KEY=os.environ.get('SECRET_KEY', 'prod-key-please-change'),
                )
        else:
            # Direct dictionary update
            app.config.update(config)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    
    # Initialize federated learning server after db is set up
    with app.app_context():
        init_fl_server(app)
    
    # Log if HUGGINGFACE_TOKEN is available
    if app.config.get('HUGGINGFACE_TOKEN'):
        app.logger.info("Hugging Face Token is configured and available for model deployment")
    else:
        app.logger.warning("Hugging Face Token not found in configuration - model deployment to Hugging Face Hub will not work")
    
    # Configure login
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    
    # Register blueprints
    from web.routes.auth import auth_bp
    from web.routes.dashboard import dashboard_bp
    from web.routes.projects import projects_bp
    from web.routes.api import api_bp
    from web.routes.metrics import metrics_bp
    from web.routes.visualization import visualization_bp
    from web.routes.organizations import organizations_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(visualization_bp)
    app.register_blueprint(organizations_bp)
    
    # Exempt API routes from CSRF protection
    csrf.exempt(api_bp)
    
    # Register error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
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
        if current_user.is_authenticated:
            return redirect(url_for('dashboard.index'))
        return render_template('index.html')
    
    # Add context processor for templates
    @app.context_processor
    def inject_now():
        return {'now': datetime.utcnow()}
    
    # Initialize database
    init_db(app)
    
    return app

@login_manager.user_loader
def load_user(user_id):
    """Load a user given the user ID."""
    return User.query.get(int(user_id))

def main():
    """Main function to run the application."""
    app = create_app('development')
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main() 