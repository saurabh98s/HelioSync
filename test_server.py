import requests
import sys
import json
import argparse

def test_server_connection(server_url, api_key):
    """Test connectivity to the federated learning server"""
    server_url = server_url.rstrip('/')
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': api_key
    }
    
    print(f"Testing connection to server: {server_url}")
    print(f"Using API key: {api_key[:8]}...{api_key[-8:]}")
    print("-" * 50)
    
    # Test 1: Basic connectivity
    try:
        print("Test 1: Basic server connectivity")
        response = requests.get(f"{server_url}/")
        print(f"  Status: {response.status_code}")
        print(f"  Response type: {response.headers.get('Content-Type', 'unknown')}")
        if response.status_code < 400:
            print("  Result: SUCCESS - Server is accessible")
        else:
            print("  Result: FAILED - Server returned an error")
    except Exception as e:
        print(f"  Result: FAILED - {str(e)}")
    print("-" * 50)
    
    # Test 2: API health endpoint
    try:
        print("Test 2: API Health check")
        urls_to_try = [
            f"{server_url}/api/health",
            f"{server_url}/health",
            f"{server_url}/api/v1/health"
        ]
        
        success = False
        for url in urls_to_try:
            try:
                print(f"  Trying: {url}")
                response = requests.get(url)
                print(f"  Status: {response.status_code}")
                if response.status_code < 400:
                    print(f"  Response: {response.text[:100]}")
                    print("  Result: SUCCESS - Health endpoint found")
                    success = True
                    break
            except Exception:
                pass
        
        if not success:
            print("  Result: FAILED - Could not find health endpoint")
    except Exception as e:
        print(f"  Result: FAILED - {str(e)}")
    print("-" * 50)
    
    # Test 3: API authentication
    try:
        print("Test 3: API authentication")
        auth_urls = [
            f"{server_url}/api/auth/verify",
            f"{server_url}/api/v1/auth/verify",
            f"{server_url}/api/clients/verify",
            f"{server_url}/api/v1/clients/verify"
        ]
        
        success = False
        for url in auth_urls:
            try:
                print(f"  Trying: {url}")
                response = requests.post(url, headers=headers)
                print(f"  Status: {response.status_code}")
                if response.status_code < 400:
                    print(f"  Response: {response.text[:100]}")
                    print("  Result: SUCCESS - Auth endpoint found")
                    success = True
                    break
            except Exception:
                pass
        
        if not success:
            print("  Result: FAILED - Could not find auth endpoint")
    except Exception as e:
        print(f"  Result: FAILED - {str(e)}")
    print("-" * 50)
    
    # Test 4: API endpoints discovery
    try:
        print("Test 4: API endpoints discovery")
        discovery_urls = [
            f"{server_url}/api",
            f"{server_url}/api/v1"
        ]
        
        for url in discovery_urls:
            try:
                print(f"  Trying: {url}")
                response = requests.get(url)
                print(f"  Status: {response.status_code}")
                if response.status_code < 400:
                    print(f"  Response: {response.text[:150]}...")
                    if 'application/json' in response.headers.get('Content-Type', ''):
                        endpoints = response.json()
                        print(f"  Found endpoints: {list(endpoints.keys()) if isinstance(endpoints, dict) else 'unknown format'}")
            except Exception as e:
                print(f"  Error: {str(e)}")
        
    except Exception as e:
        print(f"  Result: FAILED - {str(e)}")
    
    print("\nServer connectivity test complete. Use this information to adjust your client configuration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Federated Learning Server Connection')
    parser.add_argument('--server', type=str, default='http://localhost:5000', 
                      help='Server URL')
    parser.add_argument('--api_key', type=str, 
                      default='7039844b0472d0c6bdf1d4db1c6aa5d46c8be09bf872b6d9',
                      help='API key for authentication')
    
    args = parser.parse_args()
    test_server_connection(args.server, args.api_key) 