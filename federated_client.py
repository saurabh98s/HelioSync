"""
Federated Learning Client

A client implementation for communicating with the federated learning server.
This file contains all functionality needed for client registration and API communication.
"""

import requests
import argparse
import uuid
import socket
import platform
import time
import threading
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('federated_client')

class FederatedClient:
    """Client for connecting to a federated learning server."""
    
    def __init__(self, server_url, api_key, client_name=None):
        """
        Initialize a new federated learning client.
        
        Args:
            server_url (str): URL of the federated learning server
            api_key (str): API key for authentication
            client_name (str, optional): Name for this client. Defaults to hostname.
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.client_id = str(uuid.uuid4())
        self.client_name = client_name or f"Client-{socket.gethostname()}"
        self.device_info = f"{platform.system()} {platform.release()} - {platform.machine()}"
        
        # Set up headers for API requests
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        }
        
        # Internal state
        self.running = False
        self.connected = False
        self.heartbeat_thread = None
        self.current_task = None
    
    def connect(self):
        """
        Connect to the server by registering the client.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        logger.info(f"Client details:")
        logger.info(f"- Name: {self.client_name}")
        logger.info(f"- Client ID: {self.client_id}")
        logger.info(f"- Device: {self.device_info}")
        logger.info(f"- Server URL: {self.server_url}")
        logger.info(f"- API Key: {self.api_key[:8]}...{self.api_key[-8:] if len(self.api_key) > 16 else ''}")
        
        # Registration data
        data = {
            'client_id': self.client_id,
            'name': self.client_name,
            'device_info': self.device_info
        }
        
        # Try all possible registration endpoints
        endpoints = [
            '/api/register_client',
            '/api/client/register', 
            '/api/clients/register',
            '/api/register'
        ]
        
        for endpoint in endpoints:
            try:
                url = f"{self.server_url}{endpoint}"
                logger.info(f"Trying registration endpoint: {url}")
                
                response = requests.post(url, json=data, headers=self.headers)
                
                if response.status_code == 200:
                    logger.info(f"Successfully registered with server using {endpoint}")
                    self.connected = True
                    return True
                else:
                    logger.warning(f"Failed registration attempt with {endpoint}. Status: {response.status_code}")
            except Exception as e:
                logger.error(f"Error with endpoint {endpoint}: {str(e)}")
        
        logger.error("All registration attempts failed")
        return False
    
    def send_heartbeat(self):
        """
        Send a heartbeat to the server.
        
        Returns:
            bool: True if heartbeat successful, False otherwise
        """
        if not self.connected:
            logger.warning("Can't send heartbeat - not connected")
            return False
        
        # Try all possible heartbeat endpoints
        endpoints = [
            '/api/client/heartbeat',
            '/api/clients/heartbeat',
            f'/api/clients/{self.client_id}/heartbeat'
        ]
        
        data = {
            'client_id': self.client_id
        }
        
        for endpoint in endpoints:
            try:
                url = f"{self.server_url}{endpoint}"
                response = requests.post(url, json=data, headers=self.headers)
                
                if response.status_code == 200:
                    logger.debug(f"Heartbeat sent using {endpoint}")
                    return True
            except:
                continue
        
        logger.warning("All heartbeat attempts failed")
        return False
    
    def heartbeat_worker(self):
        """Worker thread that sends periodic heartbeats to the server."""
        while self.running:
            if self.connected:
                if self.send_heartbeat():
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Heartbeat sent")
                else:
                    logger.warning("Failed to send heartbeat")
            time.sleep(30)  # Send heartbeat every 30 seconds
    
    def check_for_tasks(self):
        """
        Check if the server has tasks for this client.
        
        Returns:
            bool: True if a task was received, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            url = f"{self.server_url}/api/clients/tasks"
            params = {'client_id': self.client_id}
            
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('has_task', False):
                    task = data.get('task', {})
                    self.current_task = task
                    logger.info(f"Received task: {task}")
                    return True
                else:
                    logger.debug("No tasks available")
            else:
                logger.warning(f"Failed to check for tasks. Status: {response.status_code}")
            
            return False
        except Exception as e:
            logger.error(f"Error checking for tasks: {str(e)}")
            return False
    
    def download_model(self, task_id):
        """
        Download model weights for a specific task.
        
        Args:
            task_id (str): Task ID to download model for
            
        Returns:
            str: Path to downloaded model file or None if failed
        """
        try:
            url = f"{self.server_url}/api/tasks/{task_id}/model"
            params = {'client_id': self.client_id}
            
            response = requests.get(url, params=params, headers=self.headers, stream=True)
            
            if response.status_code == 200:
                filename = f"model_{task_id}.h5"
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Model downloaded to {filename}")
                return filename
            else:
                logger.error(f"Failed to download model. Status: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return None
    
    def upload_model(self, task_id, model_path, metrics=None):
        """
        Upload trained model weights.
        
        Args:
            task_id (str): Task ID the model belongs to
            model_path (str): Path to the model file
            metrics (dict, optional): Training metrics to report
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            url = f"{self.server_url}/api/tasks/{task_id}/model"
            
            # Prepare form data
            files = {'model': open(model_path, 'rb')}
            data = {'client_id': self.client_id}
            
            if metrics:
                data['metrics'] = json.dumps(metrics)
            
            # The headers need to be removed for multipart/form-data
            headers = {'X-API-Key': self.api_key}
            
            response = requests.post(url, files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"Model uploaded successfully")
                return True
            else:
                logger.error(f"Failed to upload model. Status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            return False
    
    def start(self):
        """
        Start the client and connect to the server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        logger.info(f"Starting federated learning client: {self.client_name}")
        self.running = True
        
        # Connect to the server
        if not self.connect():
            logger.error("Failed to connect to server")
            self.running = False
            return False
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_worker)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        logger.info("Client started successfully")
        return True
    
    def stop(self):
        """Stop the client and disconnect from the server."""
        logger.info("Stopping client...")
        self.running = False
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
        
        logger.info("Client stopped")
    
    def run(self):
        """
        Run the client and wait for tasks from the server.
        
        This method blocks until the client is stopped.
        """
        try:
            logger.info("Client running and waiting for tasks. Press Ctrl+C to stop.")
            
            while self.running:
                if self.connected:
                    self.check_for_tasks()
                time.sleep(10)  # Check for tasks every 10 seconds
                
        except KeyboardInterrupt:
            logger.info("Client interrupted by user")
        finally:
            self.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--server', type=str, default='http://127.0.0.1:5000', 
                      help='Server URL')
    parser.add_argument('--api_key', type=str, 
                      default='7039844b0472d0c6bdf1d4db1c6aa5d46c8be09bf872b6d9',
                      help='API key for authentication')
    parser.add_argument('--name', type=str, default=None,
                      help='Client name (defaults to hostname)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create client
    client = FederatedClient(
        server_url=args.server,
        api_key=args.api_key,
        client_name=args.name
    )
    
    if client.start():
        client.run() 