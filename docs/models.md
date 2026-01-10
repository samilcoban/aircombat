# Model Architecture

## 🧠 The "Decoupled Commander"
We solve the multi-agent credit assignment problem by separating tactical perception (Actor) from strategic oversight (Critic).

### 1. The Actor (Tactical Pilot)
A decentralized execution model that runs on every agent independently.
*   **Input**: `Flat(Ego + Tracks)`
*   **Encoders**: Dual MLP encoders project Physics (Node) and Sensor (Edge) data into `D_MODEL=256`.
*   **Backbone**: 
    *   **Transformer Encoder**: 2 Layers, 4 Heads. Applies Self-Attention to prioritize threats (e.g., "Missile" > "Bandit").
    *   **GRU (Gated Recurrent Unit)**: Maintains temporal memory (Hidden State) to estimate unobservable enemy states (e.g., current energy state).
*   **Head**: **TRM (Tiny Recursive Model)**. 
    *   Instead of a simple linear layer, the head runs a 3-step internal recurrence to "refine" the action before outputting.
    *   `y_0 = Initial Guess` -> `y_1 = Refinement` -> `y_final`.

### 2. The Critic (Strategic Commander)
A centralized training model that sees the entire battlefield as a Graph.
*   **Input**: `PyG.Data(x, edge_index, edge_attr)` - The Global Truth.
*   **Backbone**: **EdgeGCNConv**.
    *   Message Passing Neural Network that aggregates local neighborhoods.
    *   Computes `Ally_Context` and `Enemy_Context` vectors via Global Pooling.
*   **Attention Linking**:
    *   **Problem**: How does the Global Graph know which agent is "Me" for PPO calculation?
    *   **Solution**: Differentiable Attention. The Critic uses the Actor's Ego Embedding as a `Query` to attend to the Graph Nodes (`Keys`). This mathematically extracts the specific agent's state from the global graph without index hacking.

### 3. Auxiliary World Model
A side-branch attached to the Actor.
*   **Task**: Predict `Next_State` and `Reward` given `Current_State` and `Action`.
*   **Purpose**: Regularizes the latent space, forcing the GRU to encode meaningful physical dynamics rather than just fitting the policy gradient.