#!/usr/bin/env python3
"""
Client Management Script for Federated Learning
"""

import os
import sys
import signal
import psutil
import argparse
import requests

def find_fl_clients():
    """Find all running federated learning clients."""
    fl_clients = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and 'fl_client.run_client' in ' '.join(proc.info['cmdline']):
                fl_clients.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return fl_clients

def kill_client(pid):
    """Kill a client process by PID."""
    try:
        process = psutil.Process(pid)
        process.terminate()
        print(f"Terminated client with PID {pid}")
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")
    except psutil.AccessDenied:
        print(f"Access denied when trying to terminate process {pid}")

def list_clients():
    """List all running federated learning clients."""
    clients = find_fl_clients()
    if not clients:
        print("No federated learning clients running")
        return
    
    print("\nRunning Federated Learning Clients:")
    print("-" * 80)
    print(f"{'PID':<10} {'Command Line':<70}")
    print("-" * 80)
    
    for proc in clients:
        cmdline = ' '.join(proc.info['cmdline'])
        print(f"{proc.info['pid']:<10} {cmdline[:70]}")

def kill_all_clients():
    """Kill all running federated learning clients."""
    clients = find_fl_clients()
    if not clients:
        print("No federated learning clients running")
        return
    
    for proc in clients:
        kill_client(proc.info['pid'])
    
    print(f"\nKilled {len(clients)} client(s)")

def main():
    parser = argparse.ArgumentParser(description='Manage Federated Learning Clients')
    parser.add_argument('action', choices=['list', 'kill-all', 'kill'],
                      help='Action to perform (list, kill-all, or kill specific client)')
    parser.add_argument('--pid', type=int, help='PID of client to kill (required for kill action)')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_clients()
    elif args.action == 'kill-all':
        kill_all_clients()
    elif args.action == 'kill':
        if not args.pid:
            print("Error: --pid is required for kill action")
            sys.exit(1)
        kill_client(args.pid)

if __name__ == '__main__':
    main() 