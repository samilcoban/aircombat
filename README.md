# AirCombat 3.0: High-Fidelity Multi-Agent RL Environment

**AirCombat 3.0** is a lightweight, high-performance Reinforcement Learning environment designed to train autonomous agents in Beyond-Visual-Range (BVR) and Within-Visual-Range (WVR) air combat.

Unlike arcade-style environments, this project utilizes a **custom Python-native physics engine** based on Energy-Maneuverability theory. Agents must manage kinetic energy, potential energy, turn rates, and G-loads. It is built entirely on **PyTorch** and **Gymnasium** for maximum efficiency on consumer hardware.

## 🚀 Key Features

*   **Decoupled Commander Architecture**: Solves the "Gradient Interference" problem common in Actor-Critic methods by separating Tactical Perception (Actor) from Strategic Assessment (Critic).
*   **Attention-Based Entity Linking**: The Critic uses a differentiable attention mechanism to mathematically "search" for the specific agent within the global graph, solving the batch alignment problem without fragile index mapping.
*   **Physics-Based Flight**: "6-DOF Lite" model supporting high-G maneuvers, negative-G pushovers, induced drag, and corner speeds.
*   **Multi-Agent Training**: Supports up to 30 entity (aircraft + missiles) engagements with **Centralized Training, Decentralized Execution (CTDE)**.
*   **Self-Play**: Agents train against past versions of themselves using **Prioritized Fictitious Self-Play (PFSP)**.

---

## 🧠 Model Architecture: The "Decoupled Commander"

We solve the multi-agent air combat problem using a distinct separation of concerns between the **Pilot (Actor)** and the **Commander (Critic)**.

### 1. The Actor (The Tactical Pilot)
*   **Role**: Decentralized Execution. Sees only local sensors.
*   **Input**: `Box(Obs_Dim)` (Ego State + List of Visible Tracks).
*   **Pipeline**:
    1.  **Dual Encoding**: Separately encodes Ego Physics (16D) and Track Geometry (12D).
    2.  **Transformer Backbone**: Applies self-attention to prioritize threats (e.g., "Missile closing at Mach 3 > Bandit at 40km").
    3.  **Tactical Memory (GRU)**: Maintains a hidden state to infer unobservable variables like enemy turn rate.
    4.  **Auxiliary World Model**: A side-branch predicts next-state physics to regularize the latent space.
*   **Output**: 5D continuous action vector (Roll, G-Pull, Throttle, Fire, Countermeasures).

### 2. The Critic (The Strategic Commander)
*   **Role**: Centralized Training. Sees the Global Truth.
*   **Input**: `Graph(Nodes, Edges)` (The entire battlefield).
*   **The Innovation: Attention-Based Linking**:
    *   Standard GNNs aggregate the whole graph into a single vector, losing the specific context of the agent being graded.
    *   **Our Approach**:
        1.  **Keys**: The Global Graph nodes are processed by a GNN to generate Context-Aware Embeddings.
        2.  **Query**: The Agent's Local Observation is encoded via a **Shared Physics Encoder**.
        3.  **Attention**: The network computes $Attention(Q, K)$ to mathematically "find" the agent's node in the graph based on its physical signature (Speed, Heading, Pos).
        4.  **Value**: The Critic fuses the **Specific Agent's GNN State** with **Global Team Contexts** to predict $V(s)$.

### 3. Why This Architecture?
*   **Prevents Catastrophic Forgetting**: Massive gradients from the Value Loss (e.g., "We crashed!") flow back through the GNN, but **stop** before touching the Actor's perceptual layers. The Pilot keeps its eyes; the Commander changes its mind.
*   **Solves Batch Alignment**: PPO batches agents `[A1, A2]`, while Graphs batch nodes `[N1..N10]`. Dynamic entity counts (missiles) make hard-linking impossible. Attention allows soft, differentiable linking.

---

## ⚔️ Training Methodology

### Rewards: The "Cubic Safety Manifold"
We utilize a shaped reward function designed to guide the agent without "Suicide Optimization" loopholes.
*   **Kill**: `+4.0` (Active) + `+2.0` (Win Bonus).
*   **Crash**: `-5.0` (Catastrophic Pilot Error).
*   **Shot Down**: `-2.5` (Tactical Failure).
*   **Soft Deck**: Below 3000m, a penalty scales with a **Cubic Curve** ($x^3$).
    *   At 2900m: Negligible.
    *   At 100m: Massive penalty (-0.45 per step).
    *   *Result:* Creates a smooth "gravity" pushing agents up without the instability of a hard wall.

### Self-Play with PFSP
1.  **Opponent Pool**: Successful agents (>50% win rate) are frozen and stored.
2.  **Sampling**: Opponents are sampled based on difficulty: $P(i) \propto (1 - \text{WinRate}_i)^2$.
3.  **Result**: The agent focuses on defeating its "nemeses" rather than wasting time on easy opponents.

---

## 🏫 Supervised Pretraining System

Before self-play, we bootstrap the agent using **Behavioral Cloning** on a scripted expert.

### The Expert: "HardcodedAce"
A scripted bot that implements Proportional Navigation (PN) and Energy Management logic.
*   **Missile Registry**: A persistent engine-level registry ensures kills are attributed to the owner even if the owner dies before impact.
*   **Negative Gs**: The expert (and now the agent) can push the nose down (-3G) to dive aggressively.

### Pipeline
1.  **Data Collection**: Run `pretrain.py` to collect 200k steps.
2.  **Filtering**: Only episodes with `Return > 2.0` (Kills) are kept.
3.  **Cloning**: The Actor learns to mimic the Expert's kill chain.
4.  **Hand-off**: The model is saved to `model_pretrained.pt`, ready for PPO fine-tuning.

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.10+
*   PyTorch 2.0+ (Compiled mode supported)
*   Gymnasium
*   PyTorch Geometric

### Install
```bash
pip install -r requirements.txt