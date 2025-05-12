#!/usr/bin/env python3
"""
Start Federated Learning Server

This script directly starts the federated learning server for testing and debugging.
"""

import os
import sys
import subprocess
import argparse
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'startup.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('server_startup')

def check_server_status(process, timeout=30):
    """Check if the server started successfully."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check process status
        if process.poll() is not None:
            # Process has terminated
            stdout, stderr = process.communicate()
            error_msg = f"Server process terminated with exit code {process.returncode}"
            if stdout:
                error_msg += f"\nStdout: {stdout.decode()}"
            if stderr:
                error_msg += f"\nStderr: {stderr.decode()}"
            return False, error_msg
        
        # Check if log file exists and contains startup message
        log_file = os.path.join('logs', 'server.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                if "Server initialized" in content and "Server ready to accept clients" in content:
                    return True, "Server started successfully"
        
        time.sleep(1)
    
    # If we get here, the server didn't start in time
    stdout, stderr = process.communicate()
    error_msg = "Server startup timed out"
    if stdout:
        error_msg += f"\nStdout: {stdout.decode()}"
    if stderr:
        error_msg += f"\nStderr: {stderr.decode()}"
    return False, error_msg

def main():
    """Run the federated learning server directly."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Start Federated Learning Server')
        parser.add_argument('--project_id', type=int, required=True, help='Project ID to run')
        parser.add_argument('--min_clients', type=int, default=1, help='Minimum number of clients')
        parser.add_argument('--rounds', type=int, default=1, help='Number of training rounds')
        args = parser.parse_args()
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.join(script_dir, 'logs'), exist_ok=True)
        
        # Build the path to the server script
        server_script = os.path.join(script_dir, 'fl_server', 'server.py')
        
        if not os.path.exists(server_script):
            logger.error(f"Server script not found at: {server_script}")
            return 1
        
        logger.info(f"Starting server for project {args.project_id}...")
        logger.info(f"Server script: {server_script}")
        logger.info(f"Python executable: {sys.executable}")
        
        # Build the command
        cmd = [
            sys.executable,
            server_script,
            '--project_id', str(args.project_id),
            '--min_clients', str(args.min_clients),
            '--rounds', str(args.rounds)
        ]
        
        # Print the command for reference
        logger.info("Running command: " + ' '.join(cmd))
        
        try:
            # Run the server with output redirection
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            # Check server status
            success, message = check_server_status(process)
            if not success:
                logger.error(f"Failed to start server: {message}")
                return 1
            
            logger.info(f"Server started with PID: {process.pid}")
            logger.info("Press Ctrl+C to stop the server")
            
            # Monitor the server process
            while True:
                try:
                    # Read output lines (non-blocking)
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        print(stdout_line.strip())
                    
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        print(stderr_line.strip(), file=sys.stderr)
                    
                    # Check if process is still running
                    if process.poll() is not None:
                        break
                    
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                    
                except KeyboardInterrupt:
                    logger.info("\nUser interrupted. Stopping server...")
                    process.terminate()
                    return 0
            
            # Process has ended
            return process.returncode
            
        except Exception as e:
            logger.error(f"Error running server: {str(e)}")
            if 'process' in locals():
                process.terminate()
            return 1
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 