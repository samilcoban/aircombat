# Training Process

The training pipeline in AirCombat 3.0 is a hierarchical, multi-stage process designed to move agents from "blind imitation" to "strategic superiority." 

We utilize a **Hybrid Training Strategy**: 
1.  **Supervised Learning** to teach the *Pilot* (Actor) how to fly.
2.  **Reinforcement Learning** to teach the *Commander* (Critic) how to win.
3.  **Imitation Learning (GAIL)** to keep the physics realistic.

---

## 🏗️ Phase 0: Supervised Pretraining (Behavioral Cloning)
**Script:** `pretrain.py`

In 3D continuous control, a random agent does not "explore"—it crashes. To fix this, we bootstrap the policy using **Behavioral Cloning (BC)** on a massive dataset of scripted expert behaviors.

### 💾 The Expert Dataset (1,000,000 Steps)
We collect data using `HardcodedAce`, a PID-based expert pilot. The dataset is carefully balanced across five distinct tactical scenarios (200,000 steps each) to ensure the agent learns the full manifold of flight.

1.  **Navigation (200k)**
    *   **Scenario**: Formation flying and waypoint navigation at varying altitudes.
    *   **Skill Learned**: PID stability, momentum management, and maintaining headings.
    *   **Why**: Teaches the Actor's GRU how inertia works in the physics engine.

2.  **Recovery (200k)**
    *   **Scenario**: Agents spawn in critical states (Stalls at <150kts, Inverted Dives at low altitude).
    *   **Skill Learned**: Crisis aversion.
    *   **Why**: Counter-intuitive physics. The agent learns that to fix a stall, it must *push the nose down* (sacrifice altitude for speed), preventing death spirals during RL.

3.  **Tail Chase (200k)**
    *   **Scenario**: Blue starts 2km–5km behind Red.
    *   **Skill Learned**: **Lead Pursuit** and **WEZ (Weapon Engagement Zone)** recognition.
    *   **Why**: Teaches the geometry of shooting (aiming where the target *will be*, not where it is).

4.  **Head On (200k)**
    *   **Scenario**: High-speed merge (Closure rate > Mach 1.5).
    *   **Skill Learned**: Sensor management and collision avoidance.
    *   **Why**: Teaches the agent to handle rapid state changes in the `Tracks` observation vector.

5.  **Disadvantage (200k)**
    *   **Scenario**: Blue starts with Red at its 6 o'clock (defensive).
    *   **Skill Learned**: Defensive BFM (Basic Fighter Maneuvers) and Reversals.
    *   **Note**: We use `VictimAce` (a pacifist expert) as the opponent here, allowing the dataset to contain *successful* reversals rather than just 200k examples of Blue dying.

### 🧠 Cloning Technique
*   **Deep Supervision**: Since our Actor uses a **TRM (Tiny Recursive Model)** head, we compute the loss on every internal recursion step, forcing the agent's "reasoning process" to align with the expert, not just the final output.
*   **Result**: A `model_pretrained.pt` checkpoint that is competent at flight and basic combat.

---

## 📈 Phase 1-3: Curriculum Learning (PPO)
**Script:** `train.py`

Once pretrained, we switch to **Proximal Policy Optimization (PPO)**. The `CurriculumManager` monitors the agent's win rate and promotes it through increasingly difficult "Schools."

### Phase 1: Flight School (The "Crawl" Phase)
*   **Opponent**: `StableDrone` (Non-hostile, flies straight and level).
*   **Physics**: "School Mode" (Infinite fuel, infinite ammo, relaxed G-limits).
*   **Objective**: Learn to point the nose at the target and fire without crashing.
*   **Promotion Trigger**: Win Rate > 60% (Last 200 episodes).

### Phase 2: Dogfight Instructor (The "Walk" Phase)
*   **Opponent**: `HardcodedAce` (The scripted expert used for pretraining).
*   **Scenarios**: Mixed starting positions (Merge, Chase, Defensive).
*   **Objective**: Learn to defeat the teacher. The agent must discover maneuvers the PID controller cannot perform (e.g., high-deflection snapshots).
*   **Promotion Trigger**: Win Rate > 65%.

### Phase 3: Total War (The "Run" Phase)
*   **Opponent**: **PFSP (Prioritized Fictitious Self-Play)**.
*   **Physics**: Full realism. Limited fuel (300s). Limited Ammo (4 missiles). Realistic Drag.
*   **Objective**: Develop high-level strategy and counter-strategies.

---

## ⚔️ Self-Play System (PFSP)
In Phase 3, the agent fights its own history. This prevents "cycling" (A beats B, B beats C, C beats A) and ensures robust generalization.

1.  **The Pool**:
    *   Every 50 updates, the current model challenges a **Gatekeeper** (a tournament against recent pool additions).
    *   If Win Rate > 50%, the model is frozen and added to `opponent_pool.json`.

2.  **Prioritized Sampling**:
    *   We do not sample opponents randomly. We sample based on **difficulty**.
    *   $P(opp) \propto (1 - \text{WinRate}_{vs\_opp})^2$
    *   The agent is forced to practice against the specific historical versions (or specific strategies) that it currently struggles against.

---

## 🤖 GAIL (Generative Adversarial Imitation Learning)
**Integrated into:** `train.py`

Throughout the PPO phases, we continue to use the **1,000,000 step Expert Dataset** as a "Physics Anchor."

### The Problem
RL agents often learn to exploit physics bugs (e.g., oscillating controls at 50Hz to gain unnatural lift). This wins matches but is unrealistic.

### The Solution
We run a **Discriminator (GNN)** in parallel with PPO.
1.  **Input**: It receives batches of `(GraphState, Action)` from the Agent and the Expert Dataset.
2.  **Task**: Classify "Real" (Expert) vs. "Fake" (Agent).
3.  **Reward Signal**:
    *   $R_{total} = R_{env} + \lambda \cdot -\log(1 - D(s,a))$
  
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
    *   If the agent flies erratically, the Discriminator identifies it as "Fake," and the agent receives a penalty.

This ensures that while the Commander (Critic) optimizes for Victory, the Pilot (Actor) maintains the smooth, professional flying style learned during pretraining.
