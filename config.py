# ================================================
# FILE: config.py
# ================================================
import torch


class Config:
    # === Hardware Optimization ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_ENVS = 10

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # --- Simulation ---
    PHYSICS_MODE = 'flat'
    DT = 0.2
    GRAVITY = 9.81
    SCALE_HEIGHT = 7400.0
    MAX_DURATION_SEC = 1200
    MAP_LIMITS = (-50000.0, 50000.0, -50000.0, 50000.0)

    # --- Physics Sub-stepping ---
    PHYSICS_SUBSTEPS = 5
    PHYSICS_DT = 0.04

    # --- Model Architecture ---
    D_MODEL = 128
    N_LAYERS = 2
    N_HEADS = 4

    # --- Dimensions ---
    N_AGENTS = 2
    N_ENEMIES = 2
    N_ENEMIES_MAX = 5
    MAX_TEAM_SIZE = max(N_AGENTS, N_ENEMIES_MAX)
    MAX_ENTITIES = 30

    # --- Feature Dimensions ---
    FEAT_DIM_EGO = 18
    FEAT_DIM_EDGE = 14  # Actor Edge Features
    GNN_EDGE_DIM = 8  # Critic Edge Features (Dist, ATA, AA, Align, Close, Team, Side, Up)

    # Total Observation Dimension
    OBS_DIM = FEAT_DIM_EGO + ((MAX_ENTITIES - 1) * FEAT_DIM_EDGE)

    ACTION_DIM = 5

    # --- Physics Constants ---
    MAX_G = 9.0
    THRUST_WEIGHT = 1.5
    DRAG_PARASITIC_SL = 0.0002
    DRAG_INDUCED_SL = 0.1
    MAX_FUEL_SEC = 300.0
    MAX_MISSILES = 4
    MAX_CHAFF = 20
    MAX_ACTIVE_MISSILES = 1

    # --- Sensors & Weapons ---
    RADAR_RANGE_KM = 20.0
    RADAR_FOV_DEG = 120.0
    RADAR_NOTCH_SPEED_KNOTS = 40.0

    MISSILE_SPEED = 2500.0
    MISSILE_RANGE_KM = 60.0
    MISSILE_MAX_G = 30.0
    MISSILE_BOOST_SEC = 6.0
    MISSILE_BOOST_ACCEL = 500.0
    MISSILE_DRAG_PARASITIC = 0.0001
    MISSILE_DRAG_INDUCED = 0.005
    MISSILE_MIN_SPEED = 200.0

    CM_SPOOF_PROB = 0.1

    CANNON_RANGE_KM = 1.5
    CANNON_FOV_DEG = 4.0
    CANNON_DAMAGE_PER_SEC = 1.0

    # --- Training & Rewards ---
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_COEF = 0.2
    VF_COEF = 0.5
    ENT_COEF = 0.0001
    MAX_GRAD_NORM = 0.5

    GUIDANCE_DECAY_STEPS = 3_000_000

    BATCH_SIZE = 3840
    MINIBATCH_SIZE = 480
    UPDATE_EPOCHS = 10
    TOTAL_TIMESTEPS = 10_000_000
    SAVE_INTERVAL = 50