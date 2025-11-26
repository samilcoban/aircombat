# AirCombat 3.0: High-Fidelity Multi-Agent RL Environment

**AirCombat 3.0** is a lightweight, high-performance Reinforcement Learning environment designed to train autonomous agents in Beyond-Visual-Range (BVR) and Within-Visual-Range (WVR) air combat.

Unlike arcade-style environments, this project utilizes a **custom Python-native physics engine** based on Energy-Maneuverability theory. Agents must manage kinetic energy, altitude, fuel, and G-forces to survive. It is built entirely on **PyTorch** and **Gymnasium** for maximum efficiency on consumer hardware.

## 🚀 Key Features

*   **Physics-Based Flight**: "6-DOF Lite" model with induced drag, gravity, and thrust-to-weight ratios.
*   **Multi-Agent Training**: Supports 2v2 engagements with **Centralized Training, Decentralized Execution (CTDE)**.
*   **Self-Play**: Agents train against past versions of themselves using **Prioritized Fictitious Self-Play (PFSP)**.
*   **Transformer Architecture**: Entity-centric observation space handles variable numbers of missiles and aircraft using attention mechanisms.

---

## 🧠 Model Architecture: CTDE Transformer

We solve the "variable number of entities" problem using a **Transformer Encoder** with a custom **CTDE** architecture.

### 1. The Network
*   **Backbone**: 4-Layer Transformer Encoder (`d_model=512`, `n_head=8`).
*   **CLS Token**: A learnable token is prepended to the input sequence. The Transformer updates this token via attention to aggregate a global context summary.
*   **Actor Head**: Decodes the "Ego" embedding (index 1) into actions (Decentralized Execution).
*   **Critic Head**: Decodes the "CLS" token (index 0) from the **Global State** into value estimates (Centralized Training).

### 2. CTDE (MAPPO)
*   **Actor Input**: Local Observation (Masked, Noisy). Simulates "Fog of War".
*   **Critic Input**: Global State (Unmasked, Perfect Info). The critic sees *everything* during training to provide accurate value estimates, but is discarded during inference.

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

## 🌍 The Environment: Istanbul Theatre

The simulation takes place in a geo-accurate representation of the **Marmara Region, Türkiye**.

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

### Train
```bash
python train.py
```
*   **Checkpoints**: Saved to `checkpoints/`.
*   **Visuals**: Validation GIFs rendered every 50 updates.
*   **Logs**: TensorBoard logs in `runs/`.

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
- [ ] **Phase 8: Temporal Memory** (LSTM/Frame Stacking)
