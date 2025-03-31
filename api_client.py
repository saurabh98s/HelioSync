import requests
import argparse
import sys
import json

def explore_api(server_url, api_key, path="/"):
    """Make API requests to explore the structure of the server API"""
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': api_key
    }
    
    print(f"\n--- Exploring API: {path} ---")
    
    # Construct the full URL
    full_url = f"{server_url.rstrip('/')}{path}"
    
    # Make the request
    try:
        response = requests.get(full_url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {response.headers}")
        
        if 'application/json' in response.headers.get('Content-Type', ''):
            try:
                data = response.json()
                print("\nResponse Data:")
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                print("\nResponse (not valid JSON):")
                print(response.text[:500])
        else:
            print("\nResponse (first 500 characters):")
            print(response.text[:500])
            
    except Exception as e:
        print(f"Error making request: {str(e)}")
    
    # If the user wants to explore further
    while True:
        print("\nOptions:")
        print("1. Make another request")
        print("2. Exit")
        
        choice = input("Enter choice (1-2): ")
        
        if choice == "1":
            new_path = input("Enter API path (e.g., /api/clients): ")
            explore_api(server_url, api_key, new_path)
            break
        elif choice == "2":
            print("Exiting API Explorer.")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning API Explorer')
    parser.add_argument('--server', type=str, default='http://localhost:5000', 
                      help='Server URL')
    parser.add_argument('--api_key', type=str, 
                      default='7039844b0472d0c6bdf1d4db1c6aa5d46c8be09bf872b6d9',
                      help='API key for authentication')
    parser.add_argument('--path', type=str, default='/',
                      help='Initial API path to explore')
    
    args = parser.parse_args()
    
    # Start exploring the API
    explore_api(args.server, args.api_key, args.path) 