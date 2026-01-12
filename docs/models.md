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

### 🚂 Training Pipeline

```mermaid
graph TD
    %% --- STAGE 0: BOOTSTRAP ---
    subgraph Stage_0 ["Phase 0: Supervised Pretraining"]
        direction TB
        Experts["Scripted Experts: HardcodedAce vs VictimAce"]
        
        Dataset["Expert Dataset (1,000,000 Steps)
        Tactical Distribution:
        - Recovery (200k)
        - Nav (200k)
        - Tail-Chase (200k)
        - Head-On (200k)
        - Defensive (200k)"]

        BC["Behavioral Cloning (Deep Supervision)"]
        
        Experts ==> Dataset
        Dataset ==> BC
        BC ==> Pretrained_Model["Pretrained Pilot (Ready for Flight)"]
    end

    %% --- STAGE 1-3: REINFORCEMENT ---
    subgraph Stage_RL ["Phases 1-3: Curriculum PPO"]
        direction TB
        
        subgraph Curriculum ["Curriculum Manager"]
            P1["Phase 1: School (Target Practice)"]
            P2["Phase 2: Instructor (Defeat the Ace)"]
            P3["Phase 3: Total War (Self-Play)"]
            
            P1 ==> P2
            P2 ==> P3
        end

        subgraph Feedback_Loops ["Continuous Regularization"]
            GAIL["GAIL Discriminator (Physics Anchor)"]
            PFSP["Opponent Pool (PFSP Sampling)"]
        end
    end

    %% --- THE MAIN ENGINE ---
    Pretrained_Model ==> P1
    
    %% GAIL Loop
    Dataset -.-> GAIL
    P3 <==> GAIL
    
    %% Self-Play Loop
    P3 <==> PFSP
    
    Final_Model((Elite Combat Agent))
    P3 ==> Final_Model

    %% --- STYLING ---
    classDef pretrain fill:#f0f9ff,stroke:#0ea5e9,color:#0c4a6e,stroke-width:2px;
    classDef rl fill:#fffbeb,stroke:#f59e0b,color:#451a03,stroke-width:2px;
    classDef loops fill:#f0fdf4,stroke:#22c55e,color:#064e3b,stroke-width:2px;
    classDef data fill:#ffffff,stroke:#64748b,color:#0f172a,stroke-width:1px;

    class Stage_0,BC,Pretrained_Model pretrain;
    class Stage_RL,Curriculum,P1,P2,P3 rl;
    class Feedback_Loops,GAIL,PFSP loops;
    class Experts,Dataset data;
