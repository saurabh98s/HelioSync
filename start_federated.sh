#!/bin/bash

# Install the client package in development mode
echo "Installing client package..."
pip install -e fl_client/

# Start the server
echo "Starting server..."
python run.py --config development --debug &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Start clients
echo "Starting clients..."
python -m fl_client.run_client --client_id client1 --api_key 7039844b0472d0c6bdf1d4db1c6aa5d46c8be09bf872b6d9 --data_split 0.5 &
CLIENT1_PID=$!

python -m fl_client.run_client --client_id client2 --api_key 7039844b0472d0c6bdf1d4db1c6aa5d46c8be09bf872b6d9 --data_split 0.5 &
CLIENT2_PID=$!

# Wait for all processes
wait $SERVER_PID $CLIENT1_PID $CLIENT2_PID 