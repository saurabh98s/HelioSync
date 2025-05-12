"""
Database Models

This module defines the database models for the Federated Learning web interface.
"""

from datetime import datetime, timedelta
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import hashlib

from web.extensions import db, login_manager

# Association tables
project_organizations = db.Table(
    'project_organizations',
    db.Column('project_id', db.Integer, db.ForeignKey('projects.id'), primary_key=True),
    db.Column('organization_id', db.Integer, db.ForeignKey('organizations.id'), primary_key=True)
)

class User(UserMixin, db.Model):
    """User model for authentication and access control."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    is_org_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    organization_id = db.Column(db.Integer, db.ForeignKey('organizations.id'), nullable=True)

    # Relationships
    organization = db.relationship('Organization', back_populates='users')

    def set_password(self, password):
        """Set the password using hashing."""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """Check if the provided password matches the stored hash."""
        return check_password_hash(self.password, password)

    def __repr__(self):
        return f'<User {self.username}>'


class Organization(db.Model):
    """Organization model for grouping users and projects."""
    __tablename__ = 'organizations'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    creator_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationships
    users = db.relationship('User', back_populates='organization', foreign_keys=[User.organization_id])
    creator = db.relationship('User', foreign_keys=[creator_id])
    api_keys = db.relationship('ApiKey', back_populates='organization', cascade='all, delete-orphan')
    clients = db.relationship('Client', back_populates='organization', cascade='all, delete-orphan')
    projects = db.relationship('Project', secondary=project_organizations, back_populates='organizations')

    def __repr__(self):
        return f'<Organization {self.name}>'


class ApiKey(db.Model):
    """API key model for client authentication."""
    __tablename__ = 'api_keys'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False)
    organization_id = db.Column(db.Integer, db.ForeignKey('organizations.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)

    # Relationships
    organization = db.relationship('Organization', back_populates='api_keys')

    def __init__(self, **kwargs):
        """Initialize a new API key."""
        super(ApiKey, self).__init__(**kwargs)
        
        # Set default expiration if not provided
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(days=365)
        
        # Set default created_at if not provided
        if not self.created_at:
            self.created_at = datetime.utcnow()
        
        # Set default is_active if not provided
        if self.is_active is None:
            self.is_active = True

    def is_valid(self):
        """Check if the API key is valid (not expired and active)."""
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        return True

    @classmethod
    def verify_key(cls, key):
        """Verify an API key and return the associated organization."""
        if not key:
            return None
        
        # Find the API key in the database directly (it's already hashed)
        api_key = cls.query.filter_by(key=key).first()
        
        if api_key and api_key.is_valid():
            return api_key.organization
        return None

    def __repr__(self):
        """String representation of the API key."""
        return f'<ApiKey {self.key[:8]}...>'


class Project(db.Model):
    """Project model for federated learning projects."""
    __tablename__ = 'projects'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=True)
    dataset_name = db.Column(db.String(50), nullable=False)
    framework = db.Column(db.String(20), nullable=False)  # TensorFlow, PyTorch, etc.
    min_clients = db.Column(db.Integer, default=2)
    rounds = db.Column(db.Integer, default=5)
    current_round = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='created')  # created, running, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    creator_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationships
    creator = db.relationship('User', foreign_keys=[creator_id])
    organizations = db.relationship('Organization', secondary=project_organizations, back_populates='projects')
    project_clients = db.relationship('ProjectClient', back_populates='project', cascade='all, delete-orphan')
    models = db.relationship('Model', back_populates='project', cascade='all, delete-orphan')

    @property
    def clients(self):
        """Get all clients associated with this project."""
        return [pc.client for pc in self.project_clients]

    @property
    def active_clients_count(self):
        """Count active clients (those that recently sent a heartbeat)."""
        now = datetime.utcnow()
        return sum(1 for pc in self.project_clients if pc.client.is_active(now))

    @property
    def latest_model(self):
        """Get the latest model version."""
        if not self.models:
            return None
        return sorted(self.models, key=lambda m: m.version, reverse=True)[0]

    def __repr__(self):
        return f'<Project {self.name}>'


class Client(db.Model):
    """Client model for federated learning clients."""
    __tablename__ = 'clients'

    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.String(36), unique=True, nullable=False)  # UUID
    name = db.Column(db.String(100), nullable=False)
    ip_address = db.Column(db.String(50), nullable=True)
    device_info = db.Column(db.String(200), nullable=True)
    platform = db.Column(db.String(50), nullable=True)  # Operating system platform
    machine = db.Column(db.String(50), nullable=True)  # Machine architecture
    python_version = db.Column(db.String(20), nullable=True)  # Python version
    data_size = db.Column(db.Integer, default=0)  # Size of client's dataset
    is_connected = db.Column(db.Boolean, default=False)
    last_heartbeat = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    organization_id = db.Column(db.Integer, db.ForeignKey('organizations.id'), nullable=False)

    # Relationships
    organization = db.relationship('Organization', back_populates='clients')
    project_clients = db.relationship('ProjectClient', back_populates='client', cascade='all, delete-orphan')

    def is_active(self, now=None):
        """Check if the client is considered active based on heartbeat."""
        if now is None:
            now = datetime.utcnow()
        if not self.last_heartbeat:
            return False
        # Consider a client active if it sent a heartbeat in the last 5 minutes
        return (now - self.last_heartbeat).total_seconds() < 300  # 5 minutes

    @property
    def projects(self):
        """Get all projects this client is part of."""
        return [pc.project for pc in self.project_clients]

    def __repr__(self):
        return f'<Client {self.name}>'


class ProjectClient(db.Model):
    """Association model between projects and clients with additional data."""
    __tablename__ = 'project_clients'

    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), primary_key=True)
    client_id = db.Column(db.Integer, db.ForeignKey('clients.id'), primary_key=True)
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_update = db.Column(db.DateTime, nullable=True)
    local_epochs = db.Column(db.Integer, default=0)
    training_samples = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='joined')  # joined, training, completed, failed
    metrics = db.Column(db.JSON, nullable=True)  # Training metrics like loss, accuracy

    # Relationships
    project = db.relationship('Project', back_populates='project_clients')
    client = db.relationship('Client', back_populates='project_clients')

    def __repr__(self):
        return f'<ProjectClient {self.project_id}:{self.client_id}>'


class Model(db.Model):
    """Model for trained federated learning models."""
    __tablename__ = 'models'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    version = db.Column(db.Integer, default=1)
    path = db.Column(db.String(255), nullable=True)  # Path to the saved model file
    metrics = db.Column(db.JSON, nullable=True)  # Model metrics (accuracy, loss, etc.)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_final = db.Column(db.Boolean, default=False)  # Is this the final model of the project
    is_deployed = db.Column(db.Boolean, default=False)  # Is this model deployed
    deployment_info = db.Column(db.JSON, nullable=True)  # Information about deployment
    clients_count = db.Column(db.Integer, default=0)  # Number of clients that contributed to this model
    is_sample = db.Column(db.Boolean, default=False)  # Whether this is a sample model for demonstration

    # Relationships
    project = db.relationship('Project', back_populates='models')

    def __repr__(self):
        return f'<Model {self.project_id}:{self.version}>'


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    return User.query.get(int(user_id)) 