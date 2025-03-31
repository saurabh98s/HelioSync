#!/bin/bash
# Start multiple sentiment analysis federated learning clients for testing

SERVER_ADDRESS="localhost:8080"
NUM_CLIENTS=3
FRAMEWORK="tensorflow"  # Options: tensorflow, pytorch
USE_NON_IID=false
LOCAL_EPOCHS=3
BATCH_SIZE=32
LEARNING_RATE=0.01

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --server_address)
      SERVER_ADDRESS="$2"
      shift 2
      ;;
    --num_clients)
      NUM_CLIENTS="$2"
      shift 2
      ;;
    --framework)
      FRAMEWORK="$2"
      shift 2
      ;;
    --non_iid)
      USE_NON_IID=true
      shift
      ;;
    --local_epochs)
      LOCAL_EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting $NUM_CLIENTS sentiment analysis federated learning clients..."
echo "Server address: $SERVER_ADDRESS"
echo "Framework: $FRAMEWORK"
echo "Non-IID: $USE_NON_IID"

# Build the non-IID flag if needed
NON_IID_FLAG=""
if [ "$USE_NON_IID" = true ]; then
  NON_IID_FLAG="--non_iid"
fi

# Start clients in separate processes
for ((i=1; i<=NUM_CLIENTS; i++)); do
  CLIENT_ID="client_$i"
  echo "Starting client $CLIENT_ID..."
  
  # Build the command
  CMD="python run_client.py --client_id $CLIENT_ID --server_address $SERVER_ADDRESS --framework $FRAMEWORK --local_epochs $LOCAL_EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --num_clients $NUM_CLIENTS $NON_IID_FLAG"
  
  # Run in background and save the PID
  $CMD &
  
  # Give a small delay between starting clients
  sleep 1
done

echo "All clients started. Press Ctrl+C to stop."

# Wait for all background processes to finish
wait 