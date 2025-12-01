import torch


class Config:
    # === Hardware Optimization ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # OPTIMIZATION: CPU Thread Management
    # You have 12 logical threads.
    # 16 Workers causes context-switching overhead. 10 is the sweet spot.
    NUM_ENVS = 10

    # OPTIMIZATION: Tensor Core Precision (Ampere+)
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
    # 5 substeps (25Hz) is enough for training and reduces CPU load by 50% vs 10
    PHYSICS_SUBSTEPS = 5
    PHYSICS_DT = 0.04

    # --- Model Architecture ---
    D_MODEL = 128
    N_LAYERS = 2
    N_HEADS = 4

    # --- Dimensions ---
    N_AGENTS = 2
    N_ENEMIES = 2
    MAX_TEAM_SIZE = max(N_AGENTS, N_ENEMIES)
    FEAT_DIM = 20 + MAX_TEAM_SIZE
    MAX_ENTITIES = 30
    OBS_DIM = MAX_ENTITIES * FEAT_DIM
    ACTION_DIM = 5

    # --- Physics Constants ---
    MAX_G = 9.0
    THRUST_WEIGHT = 1.2
    DRAG_PARASITIC_SL = 0.0002
    DRAG_INDUCED_SL = 0.005
    MAX_FUEL_SEC = 300.0
    MAX_MISSILES = 4
    MAX_CHAFF = 20

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
    CANNON_RANGE_KM = 1.5  # 1500 meters max range
    CANNON_FOV_DEG = 4.0  # Tight cone (requires precision)
    CANNON_DAMAGE_PER_SEC = 1.0  # 1 second of tracking = Kill (or instant if simplified)

    # --- PPO Parameters ---
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_COEF = 0.2
    VF_COEF = 0.5
    ENT_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    # OPTIMIZATION: Batch Sizing
    # 4000 / (10 Envs * 2 Agents) = 200 steps per agent per update. Clean integer.
    BATCH_SIZE = 4000
    MINIBATCH_SIZE = 500
    UPDATE_EPOCHS = 10
    TOTAL_TIMESTEPS = 10000000
    SAVE_INTERVAL = 50