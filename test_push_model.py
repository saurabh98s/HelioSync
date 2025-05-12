#!/usr/bin/env python3
"""
Test pushing a model to the server.

This script will:
1. Create a simple TensorFlow model
2. Push it to the federated learning server 
3. Confirm the model was processed correctly
"""
import os
import sys
import json
import requests
import numpy as np
import sqlite3
from datetime import datetime
import time

SERVER_URL = "http://localhost:5000"
API_ENDPOINT = f"{SERVER_URL}/api/clients/client1/update"
REGISTER_ENDPOINT = f"{SERVER_URL}/api/clients/register"
PROJECT_ID = 1  # Using the MNIST project
CLIENT_ID = "client1"

def check_api_key():
    """Check if we have an API key for the client"""
    try:
        # Path to the database file
        db_path = 'instance/app.db'
        
        if not os.path.exists(db_path):
            print(f"Error: Database file not found at {db_path}")
            return None
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get API keys from the database
        cursor.execute("SELECT id, key, organization_id FROM api_keys LIMIT 1")
        api_key_row = cursor.fetchone()
        
        if not api_key_row:
            print("No API keys found in the database.")
            return None
        
        api_key = api_key_row[1]
        org_id = api_key_row[2]
        print(f"Found API key: {api_key[:5]}... for organization ID: {org_id}")
        
        return api_key
        
    except Exception as e:
        print(f"Error getting API key: {str(e)}")
        return None

def register_client(api_key):
    """Register the client with the server"""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    data = {
        "client_id": CLIENT_ID,
        "name": "Test Client",
        "device_info": "Python Script",
        "platform": "Windows",
        "data_size": 10000
    }
    
    try:
        # Check if client exists by trying a heartbeat
        heartbeat_url = f"{SERVER_URL}/api/clients/{CLIENT_ID}/heartbeat"
        heartbeat_response = requests.post(heartbeat_url, headers=headers)
        
        if heartbeat_response.status_code == 200:
            print(f"Client {CLIENT_ID} already registered")
            return True
            
        # If heartbeat failed, register the client
        print(f"Registering client {CLIENT_ID}...")
        response = requests.post(REGISTER_ENDPOINT, json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("Registration successful:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error registering client: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Error registering client: {str(e)}")
        return False

def create_dummy_model():
    """Create a simple dummy model with weights"""
    try:
        import tensorflow as tf
        
        # Create a simple model for MNIST
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Get the weights as a numpy array
        weights = model.get_weights()
        
        # Serialize weights to list of bytes arrays
        serialized_weights = []
        for w in weights:
            # Convert to float32 to make sure it's JSON serializable
            w_float32 = w.astype(np.float32)
            serialized_weights.append(w_float32.tolist())
        
        print(f"Created model with {len(weights)} weight arrays")
        return serialized_weights
        
    except ImportError:
        print("TensorFlow is not installed. Please install it with: pip install tensorflow")
        return None

def push_model_to_server(api_key, model_weights):
    """Push the model weights to the server"""
    
    # Prepare the headers with API key
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    # Prepare the metrics data
    metrics = {
        "accuracy": 0.9532,
        "loss": 0.1245,
        "val_accuracy": 0.9678,
        "val_loss": 0.1098,
        "samples": 10000,
        "epoch": 10,
        "round": 1,
        "project_id": PROJECT_ID,
        "client_id": CLIENT_ID,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Prepare the payload
    payload = {
        "weights": model_weights,
        "metrics": metrics
    }
    
    try:
        # Make the POST request
        print(f"Sending model update to {API_ENDPOINT}...")
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        
        # Check the response
        if response.status_code == 200:
            result = response.json()
            print("Server response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Error sending model update: {str(e)}")
        return False

def verify_model_saved():
    """Verify the model was saved to the database"""
    try:
        # Path to the database file
        db_path = 'instance/app.db'
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check models table for the latest model
        cursor.execute("""
            SELECT id, project_id, version, created_at, is_final, is_sample
            FROM models 
            WHERE project_id = ? 
            ORDER BY version DESC 
            LIMIT 1
        """, (PROJECT_ID,))
        
        model_row = cursor.fetchone()
        
        if not model_row:
            print("No models found in the database.")
            return False
        
        model_id, project_id, version, created_at, is_final, is_sample = model_row
        
        print("\nModel in database:")
        print(f"ID: {model_id}")
        print(f"Project ID: {project_id}")
        print(f"Version: {version}")
        print(f"Created: {created_at}")
        print(f"Is Final: {is_final}")
        print(f"Is Sample: {is_sample}")
        
        # Get metrics
        cursor.execute("SELECT metrics FROM models WHERE id = ?", (model_id,))
        metrics_row = cursor.fetchone()
        
        if metrics_row and metrics_row[0]:
            try:
                metrics = json.loads(metrics_row[0])
                print("\nMetrics:")
                print(json.dumps(metrics, indent=2))
            except:
                print("Could not parse metrics JSON")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error verifying model: {str(e)}")
        return False

def verify_client_project_assignment():
    """Verify the client is assigned to the project"""
    try:
        # Path to the database file
        db_path = 'instance/app.db'
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the project_clients table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='project_clients'")
        if not cursor.fetchone():
            print("project_clients table doesn't exist!")
            return False
        
        # Check table structure
        cursor.execute("PRAGMA table_info(project_clients)")
        columns = {row[1]: row for row in cursor.fetchall()}
        print(f"project_clients columns: {list(columns.keys())}")
        
        # First get the client ID
        cursor.execute("""
            SELECT id FROM clients WHERE client_id = ? LIMIT 1
        """, (CLIENT_ID,))
        
        client_row = cursor.fetchone()
        
        if not client_row:
            print(f"Client {CLIENT_ID} not found in database!")
            return False
        
        client_db_id = client_row[0]
        
        # Use the correct column names based on what we found
        id_column = 'id' if 'id' in columns else 'ID' 
        
        # Check project_clients table
        try:
            cursor.execute(f"""
                SELECT pc.{id_column}, pc.client_id, pc.project_id, pc.status
                FROM project_clients pc
                WHERE pc.client_id = ? AND pc.project_id = ? 
                LIMIT 1
            """, (client_db_id, PROJECT_ID))
            
            assignment_row = cursor.fetchone()
        except sqlite3.OperationalError as e:
            print(f"Error querying project_clients: {str(e)}")
            # Try a different approach - just check if any row exists
            cursor.execute("""
                SELECT * FROM project_clients
                WHERE client_id = ? AND project_id = ? 
                LIMIT 1
            """, (client_db_id, PROJECT_ID))
            
            assignment_row = cursor.fetchone()
        
        if not assignment_row:
            print(f"Client {CLIENT_ID} not assigned to project {PROJECT_ID}. Creating assignment...")
            
            # Create the assignment
            cursor.execute("""
                INSERT INTO project_clients (client_id, project_id, status, created_at)
                VALUES (?, ?, 'joined', ?)
            """, (client_db_id, PROJECT_ID, datetime.utcnow().isoformat()))
            
            conn.commit()
            print(f"Created project assignment for client {CLIENT_ID} to project {PROJECT_ID}")
            return True
        else:
            print(f"Client {CLIENT_ID} is already assigned to project {PROJECT_ID}")
            
            # Try to get the status column index
            status_index = 3  # Default position
            try:
                # Try to update the status to 'joined'
                cursor.execute("""
                    UPDATE project_clients 
                    SET status = 'joined'
                    WHERE client_id = ? AND project_id = ?
                """, (client_db_id, PROJECT_ID))
                
                conn.commit()
                print(f"Updated assignment status to 'joined'")
            except Exception as e:
                print(f"Warning: Could not update status: {str(e)}")
            
            return True
        
    except Exception as e:
        print(f"Error checking client project assignment: {str(e)}")
        return False

def main():
    """Main function to run model pushing test"""
    print("=== Testing Model Push to Federated Learning Server ===")
    
    # Check if we have an API key
    api_key = check_api_key()
    if not api_key:
        print("Could not find API key. Exiting.")
        return
    
    # Register the client
    if not register_client(api_key):
        print("Failed to register client. Exiting.")
        return
    
    # Make sure the client is assigned to the project
    if not verify_client_project_assignment():
        print("Failed to assign client to project. Exiting.")
        return
    
    # Create a dummy model
    model_weights = create_dummy_model()
    if not model_weights:
        print("Could not create model. Exiting.")
        return
    
    # Push the model to the server
    success = push_model_to_server(api_key, model_weights)
    if not success:
        print("Failed to push model to server. Exiting.")
        return
    
    # Verify the model was saved
    print("\nWaiting 2 seconds for server to process...")
    time.sleep(2)
    
    verify_success = verify_model_saved()
    if not verify_success:
        print("Could not verify model was saved. Check server logs.")
        return
    
    print("\n=== Model Push Test Completed ===")
    print("You should now see the model in the web interface.")

if __name__ == "__main__":
    main() 