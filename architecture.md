# Federated Learning System Documentation

## System Architecture

```mermaid
graph TD
    subgraph "FL Client Component"
        C1[Client 1] 
        C2[Client 2]
        C3[Client 3]
        C1 --> |Local Training| C1
        C2 --> |Local Training| C2
        C3 --> |Local Training| C3
    end
    
    subgraph "FL Server Component"
        FS[Federated Server]
        MA[Model Aggregation]
        MM[Metrics Management]
        FS --> MA
        FS --> MM
    end
    
    subgraph "Web Interface"
        WA[Web App]
        UI[Dashboard UI]
        DB[(Database)]
        WA --> UI
        WA --> DB
    end
    
    C1 --> |Model Updates| FS
    C2 --> |Model Updates| FS
    C3 --> |Model Updates| FS
    
    FS --> |Global Model| C1
    FS --> |Global Model| C2
    FS --> |Global Model| C3
    
    FS <--> WA
    
    UI --> |User Interaction| Users((Users))
```

The federated learning system consists of three main components:
1. **Client Component** (`fl_client`)
2. **Server Component** (`fl_server`)
3. **Web Interface** (`web`)

These components work together to enable distributed model training across multiple clients while keeping data private.

## Client-Server Communication Flow

```mermaid
sequenceDiagram
    participant Client as FL Client
    participant Server as FL Server
    participant Web as Web Interface
    
    Client->>Server: Register (client_id, data_size, device_info)
    Server->>Web: Store client information
    Server->>Client: Registration confirmation
    
    loop Heartbeat
        Client->>Server: Heartbeat (every 30 seconds)
        Server->>Web: Update client status
    end
    
    loop Training Rounds
        Client->>Server: Request task
        Server->>Client: Send global model + training config
        
        Client->>Client: Train locally on private data
        
        Client->>Server: Send model updates (weights)
        Client->>Server: Send training metrics
        
        Server->>Server: Aggregate model updates
        Server->>Web: Update project status & metrics
    end
    
    Client->>Server: Final update (is_final=true)
    Server->>Server: Save final model
    Server->>Web: Mark project as completed
```

## Key Components and Features

### 1. FL Client Component

**Key Features:**
- Client registration with server
- Local model training on private data
- Secure model update transmission to server
- Automatic polling for training tasks
- Heartbeat mechanism for connection monitoring
- Robust error handling and retry mechanisms

The client component is responsible for:
1. Registering with the federated learning server
2. Training models locally on private data
3. Sending model updates to the server without sharing raw data
4. Receiving aggregated global model for further training

```python
# Client initialization example
client = FederatedClient(
    client_id="client_123",
    model=model, 
    x_train=train_data, 
    y_train=train_labels,
    batch_size=32,
    epochs=5
)
client.start()
```

### 2. FL Server Component

**Key Features:**
- Coordination of federated learning process
- Client management and authentication
- Model weight aggregation with FedAvg algorithm
- Training round management
- Performance metrics collection and analysis
- Global model distribution

The server component:
1. Manages connected clients and their status
2. Distributes the global model to clients
3. Aggregates model updates from clients
4. Tracks training progress and metrics
5. Handles client failures gracefully

```python
# Server initialization example
server = FederatedServer(
    model=initial_model,
    min_clients=2,
    rounds=10
)
server.start()
```

### 3. Web Interface

**Key Features:**
- User authentication and authorization
- Organization-based access control
- Project creation and management
- Real-time training monitoring
- Visualization of training metrics
- Model deployment capabilities
- Client monitoring dashboard

The web interface provides:
1. User and organization management
2. Project creation and configuration
3. Training monitoring and visualization
4. Model deployment tools
5. API for client-server communication

## Client Lifecycle States

```mermaid
stateDiagram-v2
    [*] --> Initializing
    Initializing --> Registering
    Registering --> WaitingForTasks
    Registering --> Failed: Registration failure
    
    WaitingForTasks --> Training: Task received
    WaitingForTasks --> Disconnected: Poll timeout
    
    Training --> UpdatingModel: Local training complete
    Training --> Error: Training failure
    
    UpdatingModel --> WaitingForTasks: Model update sent
    UpdatingModel --> Error: Update failure
    
    Error --> WaitingForTasks: Error handled
    Error --> Disconnected: Unrecoverable error
    
    Disconnected --> Registering: Reconnection attempt
    
    state Training {
        [*] --> LoadingModel
        LoadingModel --> ModelTraining
        ModelTraining --> MetricsCollection
        MetricsCollection --> [*]: Training complete
    }
```

## Project Lifecycle States

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> Initializing: Minimum clients connected
    Initializing --> Running: Project started
    Initializing --> Failed: Initialization error
    
    Running --> Round1
    Round1 --> Round2: Aggregation complete
    Round2 --> Round3: Aggregation complete
    Round3 --> RoundN: ... more rounds ...
    RoundN --> ModelEvaluation: Final round complete
    
    ModelEvaluation --> ModelSaving
    ModelSaving --> Completed: Success
    ModelSaving --> Error: Model save error
    
    Error --> Recovery: Emergency recovery
    Recovery --> Completed: Recovery success
    Recovery --> Failed: Recovery failed
    
    state RoundN {
        [*] --> ClientsTraining
        ClientsTraining --> WaitingForUpdates
        WaitingForUpdates --> AggregatingModel
        AggregatingModel --> [*]
    }
```

## Federated Learning Workflow

```mermaid
sequenceDiagram
    participant Admin as Administrator
    participant Server as FL Server
    participant Client1 as Client 1
    participant Client2 as Client 2
    participant Client3 as Client 3
    
    Admin->>Server: Create Project & Initialize Model
    Server->>Server: Prepare Global Model
    
    par Registration
        Client1->>Server: Register (client_id, data_size)
        Server->>Client1: Confirm Registration
        Client2->>Server: Register (client_id, data_size)
        Server->>Client2: Confirm Registration
        Client3->>Server: Register (client_id, data_size)
        Server->>Client3: Confirm Registration
    end
    
    loop For each round
        par Model Distribution
            Server->>Client1: Send Global Model
            Server->>Client2: Send Global Model
            Server->>Client3: Send Global Model
        end
        
        par Local Training
            Client1->>Client1: Train on Local Data
            Client2->>Client2: Train on Local Data
            Client3->>Client3: Train on Local Data
        end
        
        par Model Updates
            Client1->>Server: Send Model Update
            Client2->>Server: Send Model Update
            Client3->>Server: Send Model Update
        end
        
        Server->>Server: Aggregate Updates (FedAvg)
        Server->>Server: Update Global Model
    end
    
    Server->>Server: Save Final Model
    Admin->>Server: Access Final Model & Metrics
```

## Error Handling & Recovery Mechanisms

```mermaid
flowchart TD
    A[Client/Server Encounters Error] --> B{Error Type?}
    
    B -->|Network Error| C[Retry with Exponential Backoff]
    C --> C1{Max Retries?}
    C1 -->|No| C2[Wait & Retry]
    C1 -->|Yes| C3[Fail & Report]
    
    B -->|Client Disconnect| D[Mark Client as Disconnected]
    D --> D1[Continue with Available Clients]
    D1 --> D2{Enough Clients?}
    D2 -->|Yes| D3[Proceed with Training]
    D2 -->|No| D4[Pause Until Clients Reconnect]
    
    B -->|Model Update Failure| E[Emergency Recovery]
    E --> E1[Create Emergency Model Record]
    E1 --> E2[Mark Project as Completed]
    
    B -->|Database Error| F[Transaction Rollback]
    F --> F1[Retry Critical Operations]
    F1 --> F2[Log & Report if Persistent]
    
    B -->|Training Error| G[Skip Client for Current Round]
    G --> G1[Include in Next Round]
```

## Model Aggregation

The federated learning system uses Federated Averaging (FedAvg) for model aggregation, which is a weighted average of client models based on their data sizes.

### Standard FedAvg Algorithm

The FedAvg algorithm works as follows:

1. Let $w^t$ be the global model weights at round $t$
2. Let $w_k^{t+1}$ be the updated weights from client $k$ after training locally
3. Let $n_k$ be the number of training samples on client $k$
4. Let $n = \sum_{k=1}^K n_k$ be the total number of training samples across all clients

The aggregated global model for round $t+1$ is computed as:

$$ w^{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_k^{t+1} $$

This weighted averaging ensures that clients with more data have a proportionally larger influence on the global model.

### Performance-Weighted FedAvg (PerfFedAvg)

We have implemented PerfFedAvg, an innovative aggregation algorithm that dynamically weights client contributions based on both data size and local model quality (validation accuracy or loss).

```mermaid
flowchart TD
    A[Standard FedAvg] --> B[Weights based only on data size]
    
    C[PerfFedAvg] --> D[Weights based on data size AND model performance]
    D --> E[Hyperparameter α controls balance]
    
    F[Client 1] --> G{Weight Assignment}
    H[Client 2] --> G
    I[Client 3] --> G
    
    G --> J[Standard FedAvg]
    G --> K[PerfFedAvg]
    
    J --> L[Higher weight to clients with more data]
    K --> M[Higher weight to clients with more data AND better performance]
    
    M --> N[Better handling of non-IID data]
    M --> O[More robust to clients with biased data]
    M --> P[Dynamic trust adjustment over time]
```

#### Mathematical Formulation

Let:
- $w_k^t$: Model weights of client $k$ at round $t$
- $n_k$: Number of samples on client $k$
- $acc_k^t$: Validation accuracy of client $k$'s model at round $t$
- $\alpha \in [0,1]$: Hyperparameter balancing data size and performance weighting
- $\eta_k^t$: Client's effective contribution weight

We define the effective contribution weight as:

$$ \eta_k^t = \alpha \cdot \frac{n_k}{\sum_j n_j} + (1-\alpha) \cdot \frac{acc_k^t}{\sum_j acc_j^t} $$

Then, the global model is updated as:

$$ w^{t+1} = \sum_{k=1}^K \eta_k^t \cdot w_k^t $$

#### Interpretation

- If $\alpha = 1$ → reduces to classic FedAvg (only data size matters)
- If $\alpha = 0$ → pure accuracy-based weighting (only performance matters)
- Intermediate values blend sample size trust with observed model generalization

#### Advantages Over Standard FedAvg

1. **Non-IID Robustness**: Clients with biased or unhelpful data (poor generalization) are automatically downweighted
2. **Dynamic Trust**: A client's impact evolves over time based on current model quality
3. **Personalization-Compatible**: Clients that overfit to local data (low validation accuracy) contribute less
4. **Adaptive Learning**: System naturally adapts to heterogeneous client capabilities

#### Implementation Details

PerfFedAvg is implemented by extending the standard aggregation method:

```python
def _aggregate_weights_perfedavg(self, project_id, alpha=0.5):
    """Aggregate weights using Performance-Weighted FedAvg."""
    client_weights = self.client_weights[project_id]
    
    # Get the first client's weights to determine the structure
    first_client = next(iter(client_weights.values()))
    first_weights = first_client['weights']
    
    # Initialize aggregated weights with zeros
    aggregated_weights = [np.zeros_like(w) for w in first_weights]
    
    # Calculate total data size
    total_data_size = sum(client['data_size'] for client in client_weights.values())
    
    # Calculate total accuracy (for normalization)
    total_accuracy = sum(client['metrics'].get('val_accuracy', 0.5) 
                         for client in client_weights.values())
    
    # If no accuracy metrics, fall back to standard FedAvg
    if total_accuracy == 0:
        logger.warning("No accuracy metrics available, falling back to standard FedAvg")
        return self._aggregate_weights(project_id)
    
    # Calculate effective contribution weights with PerfFedAvg
    for client_id, client_data in client_weights.items():
        client_weights_list = client_data['weights']
        client_data_size = client_data['data_size']
        client_accuracy = client_data['metrics'].get('val_accuracy', 0.5)
        
        # Calculate data size component
        data_weight = client_data_size / total_data_size if total_data_size > 0 else 1.0/len(client_weights)
        
        # Calculate accuracy component
        accuracy_weight = client_accuracy / total_accuracy if total_accuracy > 0 else 1.0/len(client_weights)
        
        # Combined weight using alpha
        weight_factor = alpha * data_weight + (1 - alpha) * accuracy_weight
        
        logger.info(f"Client {client_id} - Data weight: {data_weight:.4f}, "
                   f"Accuracy weight: {accuracy_weight:.4f}, "
                   f"Combined weight: {weight_factor:.4f}")
        
        # Sum up the weighted contributions
        for i, w in enumerate(client_weights_list):
            aggregated_weights[i] += w * weight_factor
    
    logger.info(f"Aggregated weights using PerfFedAvg with alpha={alpha}")
    return aggregated_weights
```

### Model Aggregation Process

```mermaid
flowchart TD
    A[Start Aggregation] --> B[Collect Updates from Clients]
    B --> C[Check Minimum Client Threshold]
    C --> D{Enough Updates?}
    
    D -->|No| E[Wait for More Updates]
    E --> F{Timeout?}
    F -->|No| C
    F -->|Yes| G[Proceed with Available Updates]
    
    D -->|Yes| G
    G --> H[Calculate Total Data Size]
    H --> I{Using PerfFedAvg?}
    
    I -->|No| J[Standard FedAvg: Weight by Data Size]
    I -->|Yes| K[PerfFedAvg: Calculate Performance Metrics]
    K --> L[Compute Combined Weights Using α]
    
    J --> M[Compute Weighted Average]
    L --> M
    
    M --> N[Update Global Model]
    N --> O[Save Model for Round]
    O --> P[Update Aggregated Metrics]
    
    P --> Q[Clear Client Updates]
    Q --> R[End Aggregation]
```

### Code Implementation

The standard aggregation is implemented in the `_aggregate_weights` method of the `FederatedServer` class:

```python
def _aggregate_weights(self, project_id):
    client_weights = self.client_weights[project_id]
    
    # Get the first client's weights to determine the structure
    first_client = next(iter(client_weights.values()))
    first_weights = first_client['weights']
    
    # Initialize aggregated weights with zeros
    aggregated_weights = [np.zeros_like(w) for w in first_weights]
    
    # Calculate total data size
    total_data_size = sum(client['data_size'] for client in client_weights.values())
    
    if total_data_size == 0:
        # If data size is 0, use equal weighting
        weight_factor = 1.0 / len(client_weights)
        
        # Sum up with equal weighting
        for client_id, client_data in client_weights.items():
            client_weights_list = client_data['weights']
            
            # Sum up contributions with equal weighting
            for i, w in enumerate(client_weights_list):
                aggregated_weights[i] += w * weight_factor
    else:
        # Weighted averaging based on data size
        for client_id, client_data in client_weights.items():
            client_weights_list = client_data['weights']
            client_data_size = client_data['data_size']
            
            # Weighted contribution
            weight_factor = client_data_size / total_data_size
            
            # Sum up the weighted contributions
            for i, w in enumerate(client_weights_list):
                aggregated_weights[i] += w * weight_factor
    
    return aggregated_weights
```

## Model Saving and Deployment Flow

```mermaid
flowchart TD
    A[Training Complete] --> B[Initiate Model Saving]
    B --> C[Save TensorFlow Model]
    B --> D[Save PyTorch Model]
    
    C --> E{TF Save Successful?}
    D --> F{PT Save Successful?}
    
    E -->|Yes| G[Save TF Model to Database]
    E -->|No| H[Create Emergency TF File]
    H --> G
    
    F -->|Yes| I[Save PT Model to Database]
    F -->|No| J[Create Emergency PT File]
    J --> I
    
    G --> K[Create Model Record]
    I --> K
    
    K --> L{Model Record Created?}
    L -->|Yes| M[Mark Project as Completed]
    L -->|No| N[Emergency Model Record Creation]
    N --> M
    
    M --> O[Deployment Options]
    
    O --> P[API Deployment]
    O --> Q[HuggingFace Hub Deployment]
    O --> R[Download for Local Use]
```

## Data Flow Diagram

```mermaid
flowchart TD
    subgraph Client
        CD[(Private Data)]
        LM[Local Model]
        LT[Local Training]
        CD --> LT
        LT --> LM
    end
    
    subgraph Server
        GM[Global Model]
        MA[Model Aggregation]
        MM[Metrics Manager]
        MA --> GM
        GM --> MM
    end
    
    subgraph WebApp
        UI[User Interface]
        DB[(Database)]
        VIS[Visualizations]
        API[API Endpoints]
        DB --> VIS
        API --> DB
        VIS --> UI
    end
    
    LM --> |Model Updates| MA
    GM --> |Global Model| LM
    MM --> |Metrics| DB
    
    User(User) --> UI
    API <--> |Client Requests| Client
```

The system maintains data privacy by never sharing raw data between components:

1. **Client Data:** Never leaves the client device
2. **Model Updates:** Only model parameters are shared
3. **Aggregated Model:** Combined from multiple clients without accessing original data

## Security Features

- **API Authentication:** Secure API keys for client-server communication
- **User Authentication:** Role-based access in web interface
- **Organization Isolation:** Data and models are separated by organization
- **Encrypted Communication:** Secure data transmission protocols

## Database Schema

```mermaid
erDiagram
    User ||--o{ Organization : belongs_to
    Organization ||--o{ Project : manages
    Organization ||--o{ Client : registers
    Organization ||--o{ ApiKey : has
    Project }|--o{ Model : produces
    Project }|--o{ ProjectClient : connects
    Client }|--o{ ProjectClient : joins
    
    User {
        int id PK
        string username
        string email
        string password
        boolean is_admin
        boolean is_org_admin
        date created_at
        date last_login
        int organization_id FK
    }
    
    Organization {
        int id PK
        string name
        string description
        date created_at
        int creator_id FK
    }
    
    Project {
        int id PK
        string name
        string description
        string dataset_name
        string framework
        int min_clients
        int rounds
        int current_round
        string status
        date created_at
        int creator_id FK
    }
    
    Client {
        int id PK
        string client_id
        string name
        string ip_address
        string device_info
        int data_size
        boolean is_connected
        date last_heartbeat
        int organization_id FK
    }
    
    ProjectClient {
        int project_id PK,FK
        int client_id PK,FK
        date joined_at
        date last_update
        int local_epochs
        int training_samples
        string status
        json metrics
    }
    
    Model {
        int id PK
        int project_id FK
        int version
        float accuracy
        float loss
        int clients_count
        string path
        boolean is_final
        boolean is_deployed
        date created_at
    }
    
    ApiKey {
        int id PK
        string key
        int organization_id FK
        date created_at
        date expires_at
        boolean is_active
    }
```

The system uses a relational database to store:
- Users and organizations
- Projects and models
- Client information and metrics
- Training history and results

## Conclusion

This federated learning system provides a comprehensive solution for training machine learning models across distributed clients while preserving data privacy. It combines:

1. **Privacy-Preserving Learning:** Trains on data that never leaves client devices
2. **Scalable Architecture:** Supports multiple organizations, projects, and clients
3. **Robust Implementation:** Handles network failures and client disconnections
4. **User-Friendly Interface:** Web dashboard for monitoring and management
5. **Secure Communication:** Authentication and encrypted data transmission

The system is suitable for various applications where data privacy is critical, such as healthcare, finance, and sensitive enterprise applications.
