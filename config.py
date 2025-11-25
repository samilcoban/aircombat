import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_ENVS = 12

    # --- Simulation ---
    DT = 0.5
    GRAVITY = 9.81
    SCALE_HEIGHT = 7400.0
    MAX_DURATION_SEC = 1200
    MAP_LIMITS = (26.0, 39.0, 32.0, 43.0)

    # --- Physics Sub-stepping ---
    PHYSICS_SUBSTEPS = 10        # Run physics 10x per environment step
    PHYSICS_DT = 0.05            # 0.05s internal timestep (DT / SUBSTEPS)

    # --- Model Architecture (Scaled for Complex Tactics) ---
    # Increased from baseline (~400K params) to ~1.5M params
    # This provides capacity for learning complex multi-agent tactics,
    # long-horizon planning, and intricate sensor/weapon employment
    D_MODEL = 512        # Transformer embedding dimension (was 64/128)
    N_LAYERS = 4         # Transformer encoder layers (was 2)
    N_HEADS = 8          # Multi-head attention heads (unchanged)
    ACTION_DIM = 5       # [roll, g, throttle, fire, cm]
    # --- Dimensions ---
    N_AGENTS = 2
    N_ENEMIES = 2

    # --- MODIFIED: Added MDPI Geometry Features and Agent ID to Observations ---
    # Original 17: [lat, lon, cos(hdg), sin(hdg), speed, team, type, is_ego, 
    #               cos(roll), sin(roll), cos(pitch), sin(pitch), rwr, maws, alt, fuel, ammo]
    # +3 MDPI: [ATA, AA, Closure Rate]
    # +2 Agent ID: One-hot encoding for parameter sharing (which blue agent am I?)
    #   20: Agent_ID_0 (1.0 if this is blue agent 0, else 0.0)
    #   21: Agent_ID_1 (1.0 if this is blue agent 1, else 0.0)
    MAX_TEAM_SIZE = max(N_AGENTS, N_ENEMIES)  # 2 (for one-hot encoding)
    FEAT_DIM = 20 + MAX_TEAM_SIZE  # 20 base + 2 ID = 22
    MAX_ENTITIES = 30
    OBS_DIM = MAX_ENTITIES * FEAT_DIM

    # --- MODIFIED: Added Countermeasures Action ---
    # [Roll, G, Throttle, Fire, CM_Trigger]
    ACTION_DIM = 5

    # --- Physics ---
    GRAVITY = 9.81
    MAX_G = 9.0
    THRUST_WEIGHT = 1.2

    # --- MODIFIED: Atmospheric Physics ---
    # Base drag at Sea Level
    DRAG_PARASITIC_SL = 0.0002
    DRAG_INDUCED_SL = 0.01
    # Scale height for Earth atmosphere (meters)
    SCALE_HEIGHT = 8500.0

    # --- Logistics ---
    MAX_FUEL_SEC = 300.0  # Seconds of AFTERBURNER time (approx 5 mins full AB)
    MAX_MISSILES = 4
    MAX_CHAFF = 20

    # --- Sensors & Weapons ---
    RADAR_RANGE_KM = 20.0
    RADAR_FOV_DEG = 120.0 # Widened for WVR/Dogfight training (was 60.0)
    RADAR_NOTCH_SPEED_KNOTS = 40.0

    # --- Missiles ---
    MISSILE_SPEED = 2500.0
    MISSILE_RANGE_KM = 60.0
    MISSILE_MAX_G = 30.0
    MISSILE_BOOST_SEC = 6.0
    MISSILE_BOOST_ACCEL = 500.0
    MISSILE_DRAG_PARASITIC = 0.0001
    MISSILE_DRAG_INDUCED = 0.005
    MISSILE_MIN_SPEED = 200.0
    # Probability of spoofing per tick if CM active
    CM_SPOOF_PROB = 0.1

    # --- Model ---
    D_MODEL = 256  # Increased from 128
    N_HEADS = 8    # Increased from 4
    N_LAYERS = 3   # Increased from 2

    # --- PPO Parameters ---
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_COEF = 0.2
    VF_COEF = 0.5
    ENT_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    BATCH_SIZE = 4096
    MINIBATCH_SIZE = 512
    UPDATE_EPOCHS = 10
    TOTAL_TIMESTEPS = 4_000_000
    SAVE_INTERVAL = 20