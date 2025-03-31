"""
Configuration

This module defines configuration settings for the Flask application.
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class."""
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # SQLAlchemy configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Federated learning configuration
    FL_SERVER_HOST = os.environ.get('FL_SERVER_HOST') or 'localhost'
    FL_SERVER_PORT = int(os.environ.get('FL_SERVER_PORT') or 8080)
    FL_SERVER_TOKEN = os.environ.get('FL_SERVER_TOKEN') or 'server_token'
    FL_MODEL_PATH = os.environ.get('FL_MODEL_PATH') or os.path.join(os.getcwd(), 'models')
    
    # Deployment configuration
    DEPLOYMENT_PATH = os.environ.get('DEPLOYMENT_PATH') or os.path.join(os.getcwd(), 'deployments')
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or os.path.join(os.getcwd(), 'logs', 'app.log')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_ECHO = True


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Use a more robust database in production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    
    # Ensure SSL if in production
    if os.environ.get('PRODUCTION'):
        SESSION_COOKIE_SECURE = True
        REMEMBER_COOKIE_SECURE = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 