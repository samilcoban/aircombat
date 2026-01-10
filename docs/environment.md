# Environment & Physics

## 🌍 The World Model
The environment (`AirCombatEnv`) is a continuous control simulation built on a custom, vectorized physics engine (`AirCombatCore`). It supports two modes defined in `config.py`:
1.  **Flat**: Cartesian NED (North-East-Down) approximation for high-speed calculation.
2.  **Geodetic**: WGS84 Ellipsoid physics for realistic long-range navigation (optional).

### State Space (Observation)
The observation is a **Graph-Structured** input, flattened for the Actor but preserved for the Critic.

#### 1. Ego State (The Node) - `Dim: 20`
Every agent sees itself as a node with absolute physics and resource data:
*   **Existence**: `[Exist, Team, Type]`
*   **Position**: `[X_rel, Y_rel, Alt_norm]`
*   **Attitude**: `[CosHeading, SinHeading, SinPitch, SinRoll]`
*   **Dynamics**: `[Speed, G_Load, Fuel]`
*   **Status**: `[Ammo, Chaff, CounterMeasures_Active]`
*   **Deltas**: `[d_Heading, d_Pitch, d_Roll, d_Speed]` (Proprioception)

#### 2. Tracks (The Edges) - `Dim: 16`
Every other entity (Ally, Enemy, Missile) is an edge relative to the Ego:
*   **Spatial**: `[Dist, Local_X, Local_Y, Local_Z]`
*   **Geometry**: `[ATA (Antenna Train Angle), AA (Aspect Angle), Alignment_Cos]`
*   **Kinematics**: `[Closure_Rate, Target_Speed]`
*   **ID**: `[Target_Type, Team_Relation]`
*   **Sensor**: `[Visual/Radar_Lock_Flag]`
*   **Target Deltas**: `[Tgt_dH, Tgt_dP, Tgt_dR, Tgt_dS]` (Inferred turn rates)

### Action Space - `Dim: 5`
Continuous `Box(-1, 1)`:
1.  **Roll Rate**: Desired roll rate (commanded).
2.  **G-Load**: Desired G-pull. Positive = Pull Up, Negative = Push Down.
3.  **Throttle**: Engine power (0% to 100%).
4.  **Fire**: Trigger threshold (Discretized > 0).
5.  **Countermeasures**: Flare/Chaff deploy (Discretized > 0.5).

### ✈️ Physics Engine
The core uses an **Energy-Maneuverability** model:
*   **Lift**: Functions of Angle of Attack (AoA) and Speed.
*   **Drag**: Composite of Parasitic Drag ($v^2$) and Induced Drag ($G^2$).
*   **Gravity**: Explicit component based on pitch attitude.
*   **Specific Excess Power ($P_s$)**: Agents must manage energy state; turning hard bleeds speed.

### 🏆 Rewards
The reward function minimizes "Suicide Optimization" via a shaped manifold.
1.  **Kill**: `+4.0` (Attributed via Owner ID registry).
2.  **Win**: `+2.0` (Clear skies).
3.  **Death/Crash**: `-5.0` (Terminal).
4.  **Soft Deck**: Cubic penalty curve starting at 2000m AGL.
5.  **PBRS**: Potential-Based Reward Shaping ($Gamma \cdot \Phi_{t+1} - \Phi_t$) based on distance and alignment.