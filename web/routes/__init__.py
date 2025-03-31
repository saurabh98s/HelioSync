"""
Routes package for the Federated Learning web interface.
"""

from web.routes.auth import auth_bp
from web.routes.dashboard import dashboard_bp
from web.routes.projects import projects_bp
from web.routes.api import api_bp
from web.routes.metrics import metrics_bp
from web.routes.visualization import visualization_bp
from web.routes.organizations import organizations_bp

__all__ = [
    'auth_bp',
    'dashboard_bp',
    'projects_bp',
    'api_bp',
    'metrics_bp',
    'visualization_bp',
    'organizations_bp'
] 