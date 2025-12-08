# AirCombat 3.0: High-Fidelity Multi-Agent RL Environment

**AirCombat 3.0** is a lightweight, high-performance Reinforcement Learning environment designed to train autonomous agents in Beyond-Visual-Range (BVR) and Within-Visual-Range (WVR) air combat.

Unlike arcade-style environments, this project utilizes a **custom Python-native physics engine** based on Energy-Maneuverability theory. Agents must manage kinetic energy, altitude, fuel, and G-forces to survive. It is built entirely on **PyTorch** and **Gymnasium** for maximum efficiency on consumer hardware.

## 🚀 Key Features

*   **Physics-Based Flight**: "6-DOF Lite" model with induced drag, gravity, and thrust-to-weight ratios.
*   **Multi-Agent Training**: Supports 2v2 engagements with **Centralized Training, Decentralized Execution (CTDE)**.
*   **Self-Play**: Agents train against past versions of themselves using **Prioritized Fictitious Self-Play (PFSP)**.
*   **Transformer Architecture**: Entity-centric observation space handles variable numbers of missiles and aircraft using attention mechanisms.

---

## 🧠 Model Architecture: Hybrid Actor-Critic with GNN

We solve the multi-agent air combat problem using a **Hybrid Architecture** that combines Transformers and Graph Neural Networks.

### 1. The Network

**Actor (Policy Network)**:
*   **Backbone**: 4-Layer Transformer Encoder (`d_model=128`, `n_head=8`).
*   **Input**: Local observation of entities (Ego + visible enemies + missiles).
*   **CLS Token**: A learnable token aggregates attention from all entities via self-attention.
*   **Temporal Memory**: GRU layer maintains state across timesteps for maneuver planning.
*   **Output**: 5D continuous action vector (Roll, G-Pull, Throttle, Fire, Countermeasures).

**Critic (Value Network)**:
*   **Backbone**: 2-Layer Edge-GCN (Graph Convolutional Network).
*   **Input**: Global graph state representing all entities in the environment.
*   **Graph Construction**: 
    - **Nodes**: All entities (planes, missiles) with 12-dimensional features.
    - **Edges**: Fully connected graph with 6-dimensional edge features (distance, angles, closure rate).
*   **Fusion**: Combines global battlefield embedding with agent-specific ego embedding.
*   **Output**: Scalar value V(s) representing expected return for the specific agent.

### 2. Why Hybrid Architecture?

**Actor uses Transformer**:
*   Handles variable number of entities elegantly via attention.
*   Focuses on relevant threats (e.g., incoming missile) while ignoring distant targets.
*   Decentralized execution: only needs local observations.

**Critic uses GNN**:
*   Captures relational structure of multi-agent combat (formations, pincer attacks).
*   Graph convolutions aggregate tactical context from all entities.
*   Centralized training: sees complete battlefield state for accurate value estimation.

### 3. Graph State Representation

**Node Features** (12D per entity):
- Position (x, y, z normalized), Velocity, Heading (cos/sin), Team, Type (plane/missile), Fuel, Ammo, G-load

**Edge Features** (6D per edge):
- 3D Distance, ATA (Angle-to-Attack), AA (Aspect Angle), Heading Alignment, Closure Rate, Team Relation

### 3. Observation Space (`Box(30, 22)`)
A flattened list of up to **30 Entities**. Each entity has **22 features**:
*   **Kinematics**: Lat, Lon, Heading, Speed, Altitude, Roll, Pitch
*   **Identity**: Team, Type (Plane/Missile), Agent ID (One-Hot)
*   **Sensors**: RWR (Locked Warning), MAWS (Missile Warning)
*   **Status**: Fuel, Ammo
*   **Geometry**: ATA (Antenna Train Angle), AA (Aspect Angle), Closure Rate

---

## ⚔️ Training Methodology

### Self-Play with PFSP
We implement **Prioritized Fictitious Self-Play (PFSP)** to prevent cycles and ensure robustness:
1.  **Opponent Pool**: Successful agents (>50% win rate) are added to a historical pool.
2.  **Sampling**: Opponents are sampled based on difficulty: $P(i) \propto (1 - \text{WinRate}_i)^2$.
3.  **Result**: The agent focuses on defeating its "nemeses" rather than wasting time on easy opponents.

### Curriculum Learning: "Flight School"
We implement a rigorous "Flight School" curriculum to teach the agent basic airmanship before combat:

1.  **Phase 0: Training Wheels (Current)**
    *   **Locked Throttle**: Engine locked to 80% power to prevent stalling.
    *   **Hard Deck**: Immediate termination (-100 penalty) if altitude < 2000m.
    *   **Instructor Rewards**: Explicit rewards for level flight and altitude hold.
    *   **Sink Rate Penalty**: Immediate penalty for diving > 5m/s.
    
2.  **Phase 1: Basic Maneuvers**
    *   "Drunk" Opponent (High noise).
    *   Survival Bonus active.

3.  **Phase 2: Combat Ready**
    *   Competent Opponent (Low noise).
    *   Training wheels removed (full control).

4.  **Phase 3: Self-Play**
    *   Past versions.

5.  **Phase 4: PFSP**
    *   Hardest past versions.

---

## 🏫 Supervised Pretraining System

Before self-play training, we use a **Supervised Learning** pipeline to bootstrap the agent with basic combat skills.

### InstructorBot: The 3-in-1 Expert
A unified expert that dynamically switches between three behavior modes:

1. **Safety Pilot**: Smooth flight, altitude hold, stall recovery
   - Activates when speed drops dangerously low
   - Maintains level flight at 5000m altitude
   - Gentle 2G maneuvers only

2. **BVR Sniper**: Long-range missile employment
   - Lead pursuit geometry (aims ahead of target)
   - Moderate G-loading (4-5G turns)
   - Fires missiles when aligned and in range (\<40km)

3. **Dogfighter**: Close-range knife fighting
   - Pure pursuit (nose pointed at target)
   - High-G turns (up to 9G)
   - Guns-only engagement (\<1.5km)

### ScenarioWrapper: Tactical Diversity
Forces specific tactical scenarios upon environment reset to ensure diverse training data:

- **Tail Chase (30%)**: Blue 2km behind Red, both high-speed
- **Head-On (30%)**: 30km separation, closing head-to-head (BVR)
- **Defensive (20%)**: Blue in front, Red pursuing (defensive tactics)
- **Random (20%)**: Default environment spawning

### Training Pipeline
1. **Data Collection**: InstructorBot flies 500K steps across diverse scenarios
2. **Quality Filtering**: Episodes with return \> -20.0 are kept (filters out crashes)
3. **Supervised Learning**: 10 epochs of behavioral cloning on expert demonstrations
4. **Checkpointing**: Model saved to `checkpoints/model_pretrained.pt`

This pretrained model serves as the initialization for subsequent self-play training, significantly accelerating convergence.

---

### Physics & Realism
*   **Energy-Maneuverability**: High-G turns bleed speed ($Drag \propto G^2$). Climbing trades speed for potential energy.
*   **Missiles**: Boost-Sustain-Glide profile. Can be defeated by "dragging" (energy depletion) or "beaming" (Doppler notch).
*   **Sensors**: Radar with +/- 60° FOV and Doppler Notch logic (invisible if flying perpendicular).

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.10+
*   PyTorch 2.0+
*   Gymnasium

### Install
```bash
pip install -r requirements.txt
```

### Pretrain (Optional but Recommended)
```bash
python pretrain.py
```
*   **Purpose**: Bootstrap agent with basic combat skills via supervised learning
*   **Data**: 500K expert demonstrations from InstructorBot
*   **Output**: `checkpoints/model_pretrained.pt`

### Train
```bash
python train.py
```
*   **Checkpoints**: Saved to `checkpoints/`.
*   **Visuals**: Validation GIFs rendered every 50 updates.
*   **Logs**: TensorBoard logs in `runs/`.
*   **Pretraining**: Automatically loads pretrained checkpoint if available

### Monitor
```bash
tensorboard --logdir runs
```

---

## 🗺️ Roadmap

- [x] **Phase 1: Core Physics** (6-DOF Lite, Vector Movement)
- [x] **Phase 2: Energy Dynamics** (Drag, Gravity, Thrust)
- [x] **Phase 3: Advanced Weaponry** (Missile DLZ, Guidance)
- [x] **Phase 4: Electronic Warfare** (Radar, RWR, Notch)
- [x] **Phase 5: Self-Play** (Opponent Pool, Gate Function)
- [x] **Phase 6: Advanced Architecture** (CLS Token, Scaled Transformer)
- [x] **Phase 7: Multi-Agent RL** (CTDE, PFSP, Agent ID)
- [x] **Phase 8: Temporal Memory** (GRU with Sequence Length Control)
- [x] **Phase 9: Supervised Pretraining** (InstructorBot, Scenario Wrapper)
- [ ] **Phase 10: Advanced Self-Play** (Population-based Training)

