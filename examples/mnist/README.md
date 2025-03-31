# MNIST Federated Learning Example

This example demonstrates how to use the federated learning framework with the MNIST dataset.

## Overview

The MNIST dataset consists of handwritten digits (0-9), making it an excellent dataset for image classification tasks. In this example, we implement federated learning for MNIST where:

1. Multiple clients train on disjoint partitions of the MNIST dataset
2. A central server coordinates the training process
3. Clients share model updates (not raw data) with the server
4. The server aggregates these updates to improve the global model

## Requirements

Make sure you have installed all required dependencies:

```bash
pip install -r ../../requirements.txt
```

## Running the Example

### Start the Server

First, start the federated learning server:

```bash
python run_server.py --port 8080 --min_clients 2 --rounds 5
```

Parameters:
- `--port`: Port to run the server on (default: 8080)
- `--min_clients`: Minimum number of clients required (default: 2)
- `--rounds`: Number of federated learning rounds (default: 5)
- `--model_path`: Path to save/load model (default: models/mnist_federated)
- `--fraction_fit`: Fraction of clients to sample in each round (default: 1.0)

### Start Clients

After the server is running, start multiple clients in separate terminals. Each client needs a unique ID.

Client 1:
```bash
python run_client.py --client_id client_1 --server_address localhost:8080
```

Client 2:
```bash
python run_client.py --client_id client_2 --server_address localhost:8080
```

Parameters:
- `--client_id`: Unique identifier for this client (required)
- `--server_address`: Server address in the format host:port (default: localhost:8080)
- `--framework`: Framework to use (tensorflow or pytorch, default: tensorflow)
- `--batch_size`: Batch size for training (default: 32)
- `--local_epochs`: Number of local epochs (default: 3)
- `--learning_rate`: Learning rate for optimization (default: 0.01)
- `--non_iid`: Use non-IID data partitioning (default: False)
- `--alpha`: Dirichlet distribution parameter for non-IID partitioning (default: 0.5)
- `--num_clients`: Total number of clients in the system (default: 10)

## IID vs Non-IID Data

You can experiment with both IID (Independent and Identically Distributed) and non-IID data partitioning:

- IID: All clients have similar data distributions (default)
- Non-IID: Clients have different data distributions, simulating a more realistic scenario

For non-IID partitioning, use the `--non_iid` flag when starting clients:

```bash
python run_client.py --client_id client_1 --non_iid --alpha 0.5
```

The `alpha` parameter controls the distribution heterogeneity (lower values = more heterogeneous).

## Performance Evaluation

After training completes, the server will output the performance of the global model. You can analyze:

1. Convergence rate
2. Final accuracy
3. Impact of the number of clients
4. Effect of IID vs non-IID data distribution

## Example Output

When running the example with default settings, you should see output similar to this:

```
Starting MNIST Federated Learning Server...
Settings: min_clients=2, rounds=5
...
INFO flwr 2023-04-01 12:34:56,789 | server.py:105 | Evaluation round 5: accuracy=0.9257
INFO flwr 2023-04-01 12:34:57,123 | server.py:167 | FL finished in 142.344 seconds
``` 