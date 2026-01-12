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

### 🚂 Model Architecture

```mermaid
flowchart BT
    %% --- INDIVIDUAL ENTITY GRAPH (RESTORED) ---
    subgraph Combat_Graph ["Combat Graph (Entities & 16D Edges)"]
        direction LR
        Self(((Self Node<br/>20D)))
        
        %% Entities
        Ally1((Ally 1<br/>20D))
        Ally2((Ally 2<br/>20D))
        En1((Enemy 1<br/>20D))
        En2((Enemy 2<br/>20D))
        En3((Enemy 3<br/>20D))
        Msl((Missile<br/>20D))

        %% Tactical Edges
        Self --- |16D| Ally1
        Self --- |16D| Ally2
        Self --- |16D| En1
        Self --- |16D| En2
        Self --- |16D| En3
        Self --- |16D| Msl
    end

    %% --- FEATURE PROJECTION LAYER ---
    subgraph Encoders ["Feature Projection Layer"]
        direction LR
        subgraph Actor_Embed ["Actor Encoders (D=256)"]
            AEgo["ego_encoder<br/>(20 -> 256)"]
            AEdge["edge_encoder<br/>(16 -> 256)"]
        end

        subgraph Critic_Embed ["Critic Encoders (D=128)"]
            CEgo["GNN Node Encoder<br/>(20 -> 128)"]
            CEdge["GNN Edge Encoder<br/>(16 -> 128)"]
        end
    end

    %% --- ACTOR LOGIC ---
    subgraph Actor_Logic ["ACTOR: Tactical Pilot"]
        direction TB
        Trans["Transformer Encoder<br/>(Spatial Priority)"]
        Memory["GRU Cell<br/>(Temporal Memory)"]
        TRM["TRM Recursive Head<br/>(Action Refinement)"]
        Action["Action Output<br/>(5D Control)"]
        
        Trans ==> Memory ==> TRM ==> Action
    end

    %% --- WORLD MODEL (AUXILIARY) ---
    subgraph World_Model_Block ["Auxiliary Task"]
        WM["World Model<br/>(S, A Prediction)"]
        Preds["Next State + Reward<br/>Prediction"]
        WM ==> Preds
    end

    %% --- CRITIC LOGIC ---
    subgraph Critic_Logic ["CRITIC: Global Strategic Commander"]
        direction TB
        GNN["Edge-Aware GNN<br/>(Message Passing)"]
        Pool["Global Pooling<br/>(Context Vectors)"]
        Attn["Identity Attention Bridge"]
        Value["Value Output<br/>(V)"]
        
        GNN ==> Pool ==> Attn ==> Value
    end

    %% --- DATA FLOW CONNECTIONS ---
    %% To Actor (Thick Arrows)
    Self ==> AEgo
    Ally1 & Ally2 & En1 & En2 & En3 & Msl ==> AEdge
    AEgo & AEdge ==> Trans

    %% To Critic (Thick Arrows)
    Self & Ally1 & Ally2 & En1 & En2 & En3 & Msl ==> CEgo
    Combat_Graph -.-> |"Tactical Geometry"| CEdge
    CEgo & CEdge ==> GNN

    %% World Model Flow
    Memory ==> WM
    Action -.-> |"Action Feedback"| WM

    %% --- STYLING ---
    classDef selfNode fill:#2563eb,stroke:#38bdf8,color:#ffffff,stroke-width:4px;
    classDef allyNode fill:#93c5fd,stroke:#2563eb,color:#000000;
    classDef enNode fill:#fca5a5,stroke:#dc2626,color:#000000;
    
    classDef actorMod fill:#f8fafc,stroke:#38bdf8,color:#0f172a;
    classDef criticMod fill:#f8fafc,stroke:#fbbf24,color:#0f172a;
    classDef worldMod fill:#f8fafc,stroke:#22c55e,color:#0f172a;
    classDef embedMod fill:#1e293b,stroke:#94a3b8,color:#f8fafc;
    
    classDef transparentBox fill:none,stroke:#cbd5e1,stroke-dasharray: 5 5;

    class Self selfNode;
    class Ally1,Ally2 allyNode;
    class En1,En2,En3,Msl enNode;
    
    class Actor_Logic,Critic_Logic,World_Model_Block transparentBox;
    class Trans,Memory,TRM,Action actorMod;
    class GNN,Pool,Attn,Value criticMod;
    class WM,Preds worldMod;
    class Actor_Embed,AEgo,AEdge,Critic_Embed,CEgo,CEdge embedMod;
    
    class Combat_Graph,Encoders transparentBox;
