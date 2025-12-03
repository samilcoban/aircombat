# ================================================
# FILE: config.py
# ================================================
import torch

class Config:
    # === Hardware Optimization ===
    # Intuition: Select the best available hardware for tensor operations to maximize performance.
    # Math: Checks if CUDA is available; if so, uses GPU, otherwise CPU.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Intuition: Number of parallel environments to run for data collection.
    # Math: Higher N means more diverse experience per update but higher memory usage.
    NUM_ENVS = 10
    
    if torch.cuda.is_available():
        # Intuition: Enable Tensor Cores for faster matrix multiplications on NVIDIA GPUs.
        # Math: Reduces precision from FP32 to TF32 (19 bits) for significant speedup with minimal accuracy loss.
        torch.set_float32_matmul_precision('high')

    # --- Simulation ---
    # Intuition: Defines the physics model used (flat earth vs geodetic).
    PHYSICS_MODE = 'flat'
    
    # Intuition: Time step for the simulation logic (decision interval).
    # Math: Delta t = 0.2s means 5 decisions per second.
    DT = 0.2
    
    # Intuition: Standard gravity acceleration.
    # Math: g = 9.81 m/s^2. Used in F = ma calculations for gravity force.
    GRAVITY = 9.81
    
    # Intuition: Atmospheric scale height for density calculations.
    # Math: Used in barometric formula: rho = rho0 * exp(-h / H), where H = 7400m.
    SCALE_HEIGHT = 7400.0
    
    # Intuition: Maximum allowed time for a single episode.
    # Math: 1200 seconds = 20 minutes of simulated flight time.
    MAX_DURATION_SEC = 1200
    
    # Intuition: Spatial boundaries of the simulation world.
    # Math: (x_min, x_max, y_min, y_max) in meters. 100km x 100km area.
    MAP_LIMITS = (-50000.0, 50000.0, -50000.0, 50000.0)

    # --- Physics Sub-stepping ---
    # Intuition: Number of physics integration steps per decision step.
    # Math: Higher substeps improve numerical stability of integration (Euler/RK4).
    PHYSICS_SUBSTEPS = 5
    
    # Intuition: Time step for physics integration.
    # Math: physics_dt = DT / PHYSICS_SUBSTEPS = 0.2 / 5 = 0.04s.
    PHYSICS_DT = 0.04

    # --- Model Architecture ---
    # Intuition: Dimensionality of the embedding vectors in the Transformer/GNN.
    # Math: Size of the hidden states in the neural network.
    D_MODEL = 128
    
    # Intuition: Number of attention layers in the Transformer.
    # Math: Depth of the network.
    N_LAYERS = 2
    
    # Intuition: Number of attention heads in Multi-Head Attention.
    # Math: Splits D_MODEL into N_HEADS subspaces for parallel attention mechanisms.
    N_HEADS = 4

    # --- Dimensions ---
    # Intuition: Number of agents controlled by the policy.
    N_AGENTS = 2
    
    # Intuition: Number of enemy agents.
    N_ENEMIES = 2
    
    # Intuition: Maximum possible enemies for sizing tensors.
    N_ENEMIES_MAX = 5
    
    # Intuition: Maximum size of a team for fixed-size tensor allocation.
    # Math: max(2, 5) = 5.
    MAX_TEAM_SIZE = max(N_AGENTS, N_ENEMIES_MAX)
    
    # Intuition: Dimension of the feature vector for each entity.
    # Math: 21 base features + MAX_TEAM_SIZE (for one-hot encoding or similar).
    FEAT_DIM = 21 + MAX_TEAM_SIZE
    
    # Intuition: Maximum total entities (agents + enemies + missiles) in the scene.
    MAX_ENTITIES = 30
    
    # Intuition: Total size of the observation vector if flattened.
    # Math: MAX_ENTITIES * FEAT_DIM.
    OBS_DIM = MAX_ENTITIES * FEAT_DIM
    
    # Intuition: Number of continuous actions output by the policy.
    # Math: 5 actions: [pitch, roll, throttle, fire_missile, fire_flare].
    ACTION_DIM = 5

    # --- Physics Constants ---
    # Intuition: Structural G-force limit of the aircraft.
    # Math: Max normal acceleration = 9.0 * g.
    MAX_G = 9.0
    
    # Intuition: Thrust-to-weight ratio.
    # Math: Max Thrust = Weight * 1.5. Allows vertical climb.
    THRUST_WEIGHT = 1.5
    
    # Intuition: Parasitic drag coefficient at sea level.
    # Math: Drag_p ~ v^2. Coefficient C_Dp0.
    DRAG_PARASITIC_SL = 0.0002
    
    # Intuition: Induced drag coefficient at sea level.
    # Math: Drag_i ~ 1/v^2. Coefficient K.
    DRAG_INDUCED_SL = 0.1
    
    # Intuition: Maximum fuel capacity in seconds of full throttle.
    MAX_FUEL_SEC = 300.0
    
    # Intuition: Maximum missiles carried.
    MAX_MISSILES = 4
    
    # Intuition: Maximum chaff/flares carried.
    MAX_CHAFF = 20
    
    # Intuition: Limit on simultaneous missiles in flight per agent.
    MAX_ACTIVE_MISSILES = 1

    # --- Sensors & Weapons ---
    # Intuition: Maximum detection range of the radar.
    # Math: 20 km. Targets beyond this are not observed.
    RADAR_RANGE_KM = 20.0
    
    # Intuition: Field of View of the radar.
    # Math: +/- 60 degrees from nose.
    RADAR_FOV_DEG = 120.0
    
    # Intuition: Minimum relative speed required for radar detection (Doppler notch).
    # Math: If |closing_speed| < 40 knots, target is invisible (notched).
    RADAR_NOTCH_SPEED_KNOTS = 40.0
    
    # Intuition: Constant speed of the missile.
    # Math: 2500 m/s (approx Mach 7+). Simplified kinematics.
    MISSILE_SPEED = 2500.0
    
    # Intuition: Maximum kinematic range of the missile.
    # Math: 60 km.
    MISSILE_RANGE_KM = 60.0
    
    # Intuition: Maximum G-force the missile can pull to turn.
    # Math: 30g. Determines turn radius: r = v^2 / a.
    MISSILE_MAX_G = 30.0
    
    # Intuition: Duration of the missile's boost phase.
    MISSILE_BOOST_SEC = 6.0
    
    # Intuition: Acceleration during boost phase.
    # Math: 500 m/s^2.
    MISSILE_BOOST_ACCEL = 500.0
    
    # Intuition: Missile parasitic drag coefficient.
    MISSILE_DRAG_PARASITIC = 0.0001
    
    # Intuition: Missile induced drag coefficient.
    MISSILE_DRAG_INDUCED = 0.005
    
    # Intuition: Minimum speed before missile stalls/fails.
    MISSILE_MIN_SPEED = 200.0
    
    # Intuition: Probability that a countermeasure (chaff/flare) successfully spoofs a missile per step.
    # Math: 10% chance per step.
    CM_SPOOF_PROB = 0.1
    
    # Intuition: Effective range of the cannon.
    # Math: 1.5 km.
    CANNON_RANGE_KM = 1.5
    
    # Intuition: Firing cone of the cannon.
    # Math: 4 degrees total width.
    CANNON_FOV_DEG = 4.0
    
    # Intuition: Damage inflicted by cannon per second of continuous hit.
    CANNON_DAMAGE_PER_SEC = 1.0

    # --- PPO Parameters ---
    # Intuition: Step size for gradient descent optimization.
    # Math: alpha = 3e-4. New_params = Old_params - alpha * gradient.
    LEARNING_RATE = 3e-4
    
    # Intuition: Discount factor for future rewards.
    # Math: G_t = R_t + gamma * R_{t+1} + ... Values near 1 prioritize long-term rewards.
    GAMMA = 0.99
    
    # Intuition: Smoothing parameter for Generalized Advantage Estimation (GAE).
    # Math: Controls bias-variance trade-off in advantage estimation.
    GAE_LAMBDA = 0.95
    
    # Intuition: Clipping parameter for PPO objective to prevent large policy updates.
    # Math: Limits ratio r(theta) to [1-eps, 1+eps].
    CLIP_COEF = 0.2
    
    # Intuition: Weight of the value function loss in the total loss.
    # Math: Loss = Policy_Loss + VF_COEF * Value_Loss + ...
    VF_COEF = 0.5
    
    # Intuition: Weight of the entropy bonus to encourage exploration.
    # Math: Adds term -ENT_COEF * Entropy to loss.
    ENT_COEF = 0.001
    
    # Intuition: Maximum norm for gradient clipping to prevent exploding gradients.
    # Math: If ||grad|| > 0.5, scale grad to have norm 0.5.
    MAX_GRAD_NORM = 0.5

    # Intuition: Total number of samples collected before a PPO update.
    # Math: Must be divisible by MINIBATCH_SIZE.
    BATCH_SIZE = 3840
    
    # Intuition: Size of minibatches used for SGD updates.
    # Math: 3840 / 480 = 8 minibatches per epoch.
    MINIBATCH_SIZE = 480
    
    # Intuition: Number of times to reuse the batch for updates.
    UPDATE_EPOCHS = 10
    
    # Intuition: Total number of environment steps to train on.
    # Math: 10 million steps. Used for learning rate decay scheduling.
    TOTAL_TIMESTEPS = 10_000_000 # Added for Decay Calc
    
    # Intuition: Frequency of saving model checkpoints.
    # Math: Save every 50 updates.
    SAVE_INTERVAL = 50