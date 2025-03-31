# Sentiment Analysis Federated Learning Example

This example demonstrates how to use the federated learning framework for a text classification task - sentiment analysis.

## Overview

Sentiment analysis involves classifying text as positive or negative. In this example, we implement federated learning for sentiment analysis where:

1. Multiple clients train on disjoint partitions of a sentiment dataset
2. A central server coordinates the training process
3. Clients share model updates (not raw data) with the server
4. The server aggregates these updates to improve the global model

The example uses a simple embedding-based approach for text processing.

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
- `--model_path`: Path to save/load model (default: models/sentiment_federated)
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
- `--num_clients`: Total number of clients in the system (default: 10)

## Privacy Considerations

Text data can contain sensitive information. The federated learning approach is particularly valuable for sentiment analysis because:

1. Raw text data remains on client devices
2. Only model updates are shared with the server
3. Personal information in text is not directly exposed

This makes it suitable for processing private messages, reviews, or other text containing personal information.

## Example Output

When running the example with default settings, you should see output similar to this:

```
Starting Sentiment Analysis Federated Learning Server...
Settings: min_clients=2, rounds=5
...
INFO flwr 2023-04-01 12:34:56,789 | server.py:105 | Evaluation round 5: accuracy=0.8765
INFO flwr 2023-04-01 12:34:57,123 | server.py:167 | FL finished in 98.765 seconds
```

## Extending the Example

This example uses a synthetic dataset for demonstration purposes. For real-world applications, you might want to replace it with actual text data:

1. Modify the `load_sentiment` function in `client/data_loader.py`
2. Implement proper text preprocessing and tokenization
3. Use pre-trained word embeddings like GloVe or Word2Vec 