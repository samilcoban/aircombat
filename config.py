# ================================================
# FILE: config.py
# ================================================
"""
Configuration constants for the air combat reinforcement learning environment.
This file defines hyperparameters for physics simulation, model architecture,
feature dimensions, and PPO training settings.
"""
import torch


class Config:
    """
    Central configuration class holding all static constants for the environment,
    agent model, and training loop.
    """
    # === Hardware Optimization ===
    # Device selection: Use CUDA if available for GPU acceleration, otherwise fallback to CPU.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Number of parallel environments for vectorized training.
    # Higher values improve throughput but require more VRAM/RAM.
    NUM_ENVS = 10

    if torch.cuda.is_available():
        # Set higher precision for matrix multiplications on Ampere+ GPUs.
        torch.set_float32_matmul_precision('high')

    # --- Simulation ---
    # Physics model mode: 'flat' (2D/3D simplified) or potentially 'spherical' (future).
    PHYSICS_MODE = 'flat'
    # Simulation time step (seconds) per environment step.
    DT = 0.2
    # Gravitational acceleration (m/s^2).
    GRAVITY = 9.81
    # Reference scale height (meters) for atmospheric density calculations.
    SCALE_HEIGHT = 7400.0
    # Maximum duration of a single episode in simulation seconds.
    MAX_DURATION_SEC = 1200
    # Map boundaries (meters): (min_x, max_x, min_y, max_y).
    MAP_LIMITS = (-50000.0, 50000.0, -50000.0, 50000.0)

    # --- Physics Sub-stepping ---
    # Number of internal physics integration steps per control step (DT).
    PHYSICS_SUBSTEPS = 5
    # Internal physics time step (seconds): DT / PHYSICS_SUBSTEPS.
    PHYSICS_DT = 0.04

    # --- Model Architecture ---
    # Transformer model dimension (embedding size).
    D_MODEL = 256
    # Number of transformer encoder layers.
    N_LAYERS = 2
    # Number of attention heads in the transformer.
    N_HEADS = 4


    # --- Dimensions ---
    # Number of friendly agents (blue team).
    N_AGENTS = 3
    # Initial number of enemy agents (red team).
    N_ENEMIES = 3
    # Maximum possible enemy agents (for buffer padding).
    N_ENEMIES_MAX = 5
    # Maximum team size for architecture padding.
    MAX_TEAM_SIZE = max(N_AGENTS, N_ENEMIES_MAX)
    # Maximum total entities (agents + missiles) tracked in the environment.
    MAX_ENTITIES = 30

    # --- Feature Dimensions (UNIFIED) ---
    # NODE_DIM (20): [Exist, Team, Type, X, Y, Alt, CosH, SinH, SinP, SinR, Spd, G, Fuel, Ammo, Chaff, CM, d_Head, d_Pitch, d_Roll, d_Speed]
    # EDGE_DIM (16): [Dist, LX, LY, LZ, ATA, AA, Align, Close, TgtSpd, TgtType, TeamRel, Vis, Tgt_dH, Tgt_dP, Tgt_dR, Tgt_dS]
    NODE_DIM = 20
    EDGE_DIM = 16

    # Total Observation Dimension (Actor Input)
    # 1 Ego Node + (MaxEntities-1) Tracks (Edges)
    OBS_DIM = NODE_DIM + ((MAX_ENTITIES - 1) * EDGE_DIM)

    # Number of action dimensions: [Roll, G-load, Throttle, Fire, Countermeasures].
    ACTION_DIM = 5

    # --- Physics Constants ---
    # Maximum G-force load factor the aircraft can sustain.
    MAX_G = 9.0
    # Maximum instantaneous turn rate (degrees per second).
    MAX_TURN_RATE_DEG = 20.0
    # Thrust-to-weight ratio. >1.0 allows vertical acceleration.
    THRUST_WEIGHT = 1.5
    # Parasitic drag coefficient at sea level (zero-lift drag).
    DRAG_PARASITIC_SL = 0.0002
    # Induced drag coefficient factor (drag due to lift).
    DRAG_INDUCED_SL = 0.1
    # Maximum fuel capacity (seconds of full-throttle flight at sea level).
    MAX_FUEL_SEC = 300.0
    # Maximum number of air-to-air missiles per agent.
    MAX_MISSILES = 4
    # Maximum number of chaff countermeasures.
    MAX_CHAFF = 20
    # Maximum missiles an agent can have in flight simultaneously.
    MAX_ACTIVE_MISSILES = 1

    # --- Sensors & Weapons ---
    # Radar max tracking range (kilometers).
    RADAR_RANGE_KM = 20.0
    # Radar field of view (degrees), centered on the nose.
    RADAR_FOV_DEG = 120.0
    # Doppler notch threshold (knots). Targets below this closure rate are hidden.
    RADAR_NOTCH_SPEED_KNOTS = 40.0

    # Missile kinematics
    MISSILE_SPEED = 2500.0           # Constant trim speed (m/s).
    MISSILE_RANGE_KM = 60.0          # Absolute max fly-out range (km).
    MISSILE_MAX_G = 30.0             # Maximum G-load for maneuvering.
    MISSILE_BOOST_SEC = 6.0          # Duration of boost phase (seconds).
    MISSILE_BOOST_ACCEL = 500.0      # Acceleration during boost (m/s^2).
    MISSILE_DRAG_PARASITIC = 0.0001  # Missile parasitic drag coefficient.
    MISSILE_DRAG_INDUCED = 0.005     # Missile induced drag coefficient.
    MISSILE_MIN_SPEED = 200.0        # Min speed before stall/self-destruct.

    # Probability that a single chaff burst decoys a missile.
    CM_SPOOF_PROB = 0.1

    # Cannon constants
    CANNON_RANGE_KM = 1.5            # Effective cannon range (km).
    CANNON_FOV_DEG = 4.0             # Cannon aiming cone (degrees).
    CANNON_DAMAGE_PER_SEC = 1.0      # Damage per second of time-on-target.

    # --- Training & Rewards (PPO Hyperparameters) ---
    # UPDATED: Lower LR for fine-tuning transfer
    LEARNING_RATE = 2.0e-5           # Optimizer learning rate.
    GAMMA = 0.99                     # Discount factor for future rewards.
    GAE_LAMBDA = 0.95                # GAE lambda for advantage estimation.
    CLIP_COEF = 0.2  # Was 0.1 (Too strict for high variance)  # PPO clipping epsilon.
    VF_COEF = 0.5                    # Value function loss coefficient.
    ENT_COEF = 0.0  # INCREASED: Encourage exploration/variance  # Entropy bonus coefficient.
    MAX_GRAD_NORM = 0.5              # Gradient clipping norm.
    AUX_COEF = 0.001                 # Auxiliary loss coefficient.
    TARGET_KL = 0.02                 # Target KL for early stopping.

    # Curriculum / guidance decay steps.
    GUIDANCE_DECAY_STEPS = 3_000_000
    # Steps to freeze actor during initial training.
    FREEZE_ACTOR_STEPS = 50

    # Batch processing parameters
    BATCH_SIZE = 1920                # Total experiences per update.
    SEQ_LEN = 16                     # Sequence length for recurrent policies.
    MINIBATCH_SIZE = 480             # Minibatch size for SGD updates.
    UPDATE_EPOCHS = 4                # Number of epochs per update.
    TOTAL_TIMESTEPS = 10_000_000     # Total training timesteps.
    SAVE_INTERVAL = 50               # Checkpoint save interval (updates).

    # --- Regularization ---
    DROPOUT = 0.1                    # Dropout probability for neural networks.
    WEIGHT_DECAY = 1e-4              # L2 weight decay for optimizer.