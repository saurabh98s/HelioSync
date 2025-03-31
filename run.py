#!/usr/bin/env python3
"""
Federated Learning Platform

This is the main entry point for running the Federated Learning web platform.
"""

import os
import argparse
from web.app import create_app
from web.config import Config, DevelopmentConfig, ProductionConfig

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Federated Learning Platform')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    parser.add_argument('--config', type=str, default='development',
                        choices=['development', 'production'],
                        help='Configuration to use')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    return parser.parse_args()

def main():
    """Run the Flask application."""
    args = parse_args()
    
    # Select the configuration based on the command-line argument
    if args.config == 'development':
        config = DevelopmentConfig
    elif args.config == 'production':
        config = ProductionConfig
    else:
        config = Config
    
    # Override debug mode if specified
    if args.debug:
        os.environ['FLASK_DEBUG'] = '1'
    
    # Create and run the app
    app = create_app(config)
    
    print(f"Starting Federated Learning Platform on {args.host}:{args.port}")
    print(f"Configuration: {args.config}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == "__main__":
    main() 