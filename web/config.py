"""
Configuration Module

This module loads environment variables from the config.env file
and makes them available to the Flask application.
"""

import os
from dotenv import load_dotenv
from datetime import timedelta

# Load environment variables from config.env file
config_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
if os.path.exists(config_env_path):
    load_dotenv(config_env_path)
    print(f"Loaded environment variables from {config_env_path}")
else:
    print(f"Warning: config.env file not found at {config_env_path}")

# Define configuration variables
class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-please-change')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
    
    # Server settings
    SERVER_HOST = os.environ.get('SERVER_HOST', 'localhost')
    SERVER_PORT = int(os.environ.get('SERVER_PORT', 5000))
    MIN_CLIENTS = int(os.environ.get('MIN_CLIENTS', 2))
    ROUNDS = int(os.environ.get('ROUNDS', 5))
    
    # API Key for client authentication
    API_KEY = os.environ.get('API_KEY', 'federated_learning_key')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Flask configuration
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Federated learning configuration
    FL_SERVER_HOST = os.environ.get('FL_SERVER_HOST') or 'localhost'
    FL_SERVER_PORT = int(os.environ.get('FL_SERVER_PORT') or 8080)
    FL_SERVER_TOKEN = os.environ.get('FL_SERVER_TOKEN') or 'server_token'
    FL_MODEL_PATH = os.environ.get('FL_MODEL_PATH') or os.path.join(os.getcwd(), 'models')
    
    # Deployment configuration
    DEPLOYMENT_PATH = os.environ.get('DEPLOYMENT_PATH') or os.path.join(os.getcwd(), 'deployments')
    
    # Logging configuration
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