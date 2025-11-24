# AirCombat 3.0: High-Fidelity RL Environment

**AirCombat 3.0** is a lightweight, high-performance Reinforcement Learning environment designed to train autonomous agents in Beyond-Visual-Range (BVR) and Within-Visual-Range (WVR) air combat.

Unlike arcade-style environments, this project utilizes a **custom Python-native physics engine** based on Energy-Maneuverability theory. Agents must manage kinetic energy, altitude, fuel, and G-forces to survive. It is built entirely on **PyTorch** and **Gymnasium** for maximum efficiency on consumer hardware (e.g., 12-core CPU / GTX 1650).

## 🌍 The Environment: Istanbul Theatre

The simulation takes place in a geo-accurate representation of the **Marmara Region, Türkiye**, centered on the Bosphorus Strait.

### Realism Factors & Physics
The core engine (`src/core.py`) implements **"6-DOF Lite"** physics:

*   **Bank-to-Turn:** Agents cannot simply yaw; they must roll (bank) and pull back on the stick to turn, mimicking real flight dynamics.
*   **Energy Management:**
    *   **Induced Drag:** High-G turns bleed speed rapidly ($Drag \propto G^2$).
    *   **Gravity:** Climbing trades speed for potential energy; diving converts altitude to speed.
    *   **Thrust-to-Weight:** Simulated at ~1.2 (Rafale equivalent).
    *   **Fuel:** Afterburner usage consumes fuel rapidly. Empty tanks result in a glider state.
*   **Missile Physics (DLZ):**
    *   **Boost Phase:** 6 seconds of high-thrust acceleration (> Mach 3).
    *   **Glide Phase:** Motor burnout followed by deceleration due to drag.
    *   **Kinetic Defeat:** Missiles can be defeated by "dragging" (running them out of fuel) or "beaming" (high-G turns to bleed their energy).
*   **Sensors & Electronic Warfare:**
    *   **Radar:** Limited Field of View (+/- 60°) and Range (100km).
    *   **Doppler Notch:** Agents flying perpendicular to an enemy radar (within a specific speed gate) become invisible to that radar.
    *   **RWR (Radar Warning Receiver):** Agents detect when they are being locked by an enemy.
    *   **MAWS (Missile Approach Warning System):** Detects incoming missiles targeting the agent.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Map Dimensions** | ~500km x 440km | Istanbul / Marmara Region |
| **Tick Rate** | 0.5s (2Hz) | Balance between precision and training speed |
| **Max G-Load** | 9.0 G | Structural limit for aircraft |
| **Missile G-Load** | 30.0 G | Limit for interceptors |
| **Engagement Type** | 2v2 | Blue Team (RL) vs Red Team (Scripted Physics AI) |

---

## 🧠 Model Architecture: Entity-Centric Transformer

We solve the "variable number of entities" problem (planes, missiles, etc.) using a **Transformer Encoder**. The agent does not see a fixed grid; it sees a list of objects.

### Observation Space (`Box(30, 17)`)
A flattened list of up to **30 Entities** (Self, Allies, Enemies, Missiles). Each entity has **17 features**:
1.  `Relative Latitude` (Normalized)
2.  `Relative Longitude` (Normalized)
3.  `Cos(Heading)`
4.  `Sin(Heading)`
5.  `Speed` (Normalized, Mach 0 - 2.0)
6.  `Team ID` (+1 Friend, -1 Foe)
7.  `Is Missile` (1/0)
8.  `Is Self` (1/0 - Ego Mask)
9.  `Cos(Roll)`
10. `Sin(Roll)`
11. `Cos(Pitch)`
12. `Sin(Pitch)`
13. `RWR Signal` (1.0 if this entity is locking me)
14. `MAWS Signal` (1.0 if this is a missile targeting me)
15. `Altitude` (Normalized)
16. `Fuel` (Normalized 0-1)
17. `Ammo` (Normalized 0-1)

### Action Space (`Box(5)`)
Continuous control inputs normalized between `[-1, 1]`:
1.  **Roll Rate:** Command bank angle change (max 90°/sec).
2.  **G-Pull:** Pitch command. Maps to -1G (Push) to +9G (Pull).
3.  **Throttle:** 0% (Idle/Airbrake) to 100% (Afterburner).
4.  **Fire:** If > 0.0, attempts to launch a missile (if within parameters).
5.  **Countermeasures:** If > 0.5, deploys Chaff/Flares to spoof missiles.

### The Network
*   **Backbone:** 3-Layer Transformer Encoder (`d_model=256`, `n_head=8`).
*   **Actor Head:** Decodes the "Ego" embedding (index 0) into actions.
*   **Critic Head:** Performs Max-Pooling over all entity embeddings to evaluate the global tactical situation ($V(s)$).
*   **Parameter Count:** ~283,000 (Extremely fast training).

---

## 📈 Current Results

The project has successfully reached **Phase 4 (Electronic Warfare)**.
- **Emergent Behavior:** Agents have learned to use the "Notch" maneuver to break radar locks.
- **Energy Management:** Agents correctly trade altitude for speed during engagements.
- **Self-Play:** A self-play mechanism is implemented where agents train against past versions of themselves (`src/self_play.py`), preventing strategy stagnation.
- **Performance:** Training runs at >1000 FPS on a standard laptop GPU (GTX 1650).

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.8+
*   PyTorch 2.0+
*   Gymnasium

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train
The training script auto-detects your hardware (CPU cores/GPU) and handles checkpointing automatically.
```bash
python train.py
```
*   **Checkpoints:** Saved to `checkpoints/` every 50 updates.
*   **Visuals:** A validation GIF is rendered to `checkpoints/` every 50 updates.
*   **Logs:** TensorBoard logs are saved to `runs/`.

### Monitoring
```bash
tensorboard --logdir runs
```

---

## 🗺️ Roadmap & Todo List

We are currently in **Phase 5**.

- [x] **Phase 1: The Core Rewrite**
    - [x] Replace legacy CMANO simulator with `src/core.py`.
    - [x] Implement vector-based movement and rendering.
    - [x] Migrate to `AsyncVectorEnv`.

- [x] **Phase 2: Energy-Maneuverability**
    - [x] Implement Bank-to-Turn physics.
    - [x] Implement Induced Drag (Speed bleed on turns).
    - [x] Update Observations to include Roll/Pitch.

- [x] **Phase 3: Advanced Weaponry**
    - [x] Implement Missile Boost/Glide phases.
    - [x] Implement G-limited missile turning.
    - [x] Update Red Team AI to use physics-based flying and firing.

- [x] **Phase 4: Electronic Warfare & Sensors**
    - [x] **Radar Cone:** Limit vision to +/- 60 degrees.
    - [x] **RWR (Radar Warning Receiver):** Add input feature "Am I being locked?".
    - [x] **The Notch:** Make agents invisible to radar if flying perpendicular (Doppler filter).

- [ ] **Phase 5: Environmental Hazards**
    - [x] **Fuel Limits:** Penalize afterburner usage.
    - [ ] **SAM Sites:** Static ground threats creating "No-Fly Zones".

- [x] **Phase 6: Self-Play**
    - [x] Save "Ace" versions of Blue Team.
    - [x] Load them as Red Team opponents to create an automated curriculum.
