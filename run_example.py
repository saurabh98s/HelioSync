#!/usr/bin/env python3
"""
Federated Learning - Example Runner

This script provides a simple way to run the federated learning examples.
"""

import argparse
import os
import subprocess
import time
import sys
import signal
import atexit

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nStopping all processes...")
    for process in running_processes:
        if process.poll() is None:  # If process is still running
            process.terminate()
    sys.exit(0)

# List to keep track of running processes
running_processes = []

def cleanup():
    """Clean up processes on exit."""
    for process in running_processes:
        if process.poll() is None:  # If process is still running
            process.terminate()

def main():
    """Main function to run federated learning examples."""
    parser = argparse.ArgumentParser(description="Run Federated Learning Examples")
    parser.add_argument("--example", type=str, default="mnist",
                       choices=["mnist", "sentiment"],
                       help="Example to run")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for the server")
    parser.add_argument("--min_clients", type=int, default=2,
                       help="Minimum number of clients")
    parser.add_argument("--num_clients", type=int, default=3,
                       help="Number of clients to start")
    parser.add_argument("--rounds", type=int, default=5,
                       help="Number of federated learning rounds")
    parser.add_argument("--framework", type=str, default="tensorflow",
                       choices=["tensorflow", "pytorch"],
                       help="Framework to use")
    parser.add_argument("--non_iid", action="store_true",
                       help="Use non-IID data partitioning")
    parser.add_argument("--local_epochs", type=int, default=3,
                       help="Number of local epochs")
    
    args = parser.parse_args()
    
    # Register signal handler and cleanup
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup)
    
    # Set paths based on example
    example_dir = os.path.join("examples", args.example)
    server_script = os.path.join(example_dir, "run_server.py")
    clients_script = os.path.join(example_dir, "start_clients.sh")
    
    # Start server
    print(f"Starting {args.example} federated learning server...")
    server_cmd = [
        "python", server_script,
        "--port", str(args.port),
        "--min_clients", str(args.min_clients),
        "--rounds", str(args.rounds)
    ]
    
    server_process = subprocess.Popen(server_cmd)
    running_processes.append(server_process)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Start clients
    print(f"Starting {args.num_clients} {args.example} federated learning clients...")
    client_cmd = [
        "bash", clients_script,
        "--server_address", f"localhost:{args.port}",
        "--num_clients", str(args.num_clients),
        "--framework", args.framework,
        "--local_epochs", str(args.local_epochs)
    ]
    
    if args.non_iid:
        client_cmd.append("--non_iid")
    
    client_process = subprocess.Popen(client_cmd)
    running_processes.append(client_process)
    
    # Wait for both processes to complete
    server_process.wait()
    client_process.wait()
    
    print("Federated learning completed.")

if __name__ == "__main__":
    main() 