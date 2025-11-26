# Import required libraries for RL environment, numerical operations, and simulation utilities
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from config import Config
from src.core import AirCombatCore
from src.utils.map_limits import MapLimits
from src.utils.geodesics import geodetic_distance_km, geodetic_bearing_deg, geodetic_direct


class AirCombatEnv(gym.Env):
    """
    Gymnasium-compatible environment for air combat simulation.
    
    This environment wraps the AirCombatCore physics engine and provides:
    - Observation encoding with sensor simulation (radar, RWR, MAWS)
    - MDPI/SOTA reward structure emphasizing kills over shaping rewards
    - Support for self-play training via concatenated actions
    - Curriculum learning via kappa parameter
    
    The environment simulates a 1v1 air combat scenario where the blue agent
    must outmaneuver and destroy the red opponent using BVR (Beyond Visual Range)
    tactics, energy management, and missile employment.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        """
        Initialize the environment.
        
        Sets up:
        - Map boundaries and rendering limits
        - Action space (5D continuous: roll, g-pull, throttle, fire, countermeasures)
        - Observation space (flattened feature vectors for all entities)
        - Entity tracking lists for team management
        - Curriculum parameter (kappa) for opponent difficulty
        """
        super().__init__()
        self.cfg = Config              # Global configuration parameters
        self.core = None               # Physics core (initialized in reset())

        # === MAP BOUNDARIES ===
        # Defines the geodetic (lat/lon) boundaries for the simulation arena
        self.map_limits = MapLimits(*self.cfg.MAP_LIMITS)

        # Tactical Zoom for Rendering: Zoomed-in view centered on combat area
        # Used for visualization to focus on the engagement rather than entire map
        center_lon = (self.cfg.MAP_LIMITS[0] + self.cfg.MAP_LIMITS[2]) / 2.0
        center_lat = (self.cfg.MAP_LIMITS[1] + self.cfg.MAP_LIMITS[3]) / 2.0
        zoom = 0.75  # Zoom factor (higher = more zoomed in)
        self.render_limits = MapLimits(
            center_lon - zoom, center_lat - zoom,
            center_lon + zoom, center_lat + zoom
        )

        # === ACTION SPACE ===
        # Continuous 5D action vector normalized to [-1, 1]
        # [0] Roll Rate: Command roll angular velocity
        # [1] G-Pull: Command vertical G-load for turns
        # [2] Throttle: Engine power setting
        # [3] Fire: Missile launch trigger
        # [4] Countermeasures: Chaff/flare deployment
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.ACTION_DIM,), dtype=np.float32
        )

        # === OBSERVATION SPACE ===
        # Flattened feature vectors for all visible entities
        # Each entity encoded as FEAT_DIM features (position, heading, speed, etc.)
        # Multiple entities concatenated with zero-padding for missing entities
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.OBS_DIM,), dtype=np.float32
        )

        # Visualization renderer (lazy-loaded on first render() call)
        self.renderer = None
        
        # Entity tracking: Lists of UIDs for team management
        self.blue_ids = []   # Friendly agents (RL-controlled)
        self.red_ids = []    # Enemy agents (AI or self-play controlled)
        
        # Curriculum parameters
        self.kappa = 0.0  # Opponent difficulty: 0.0 = expert, 1.0 = random/novice
        self.phase = 1    # Curriculum phase (1-4)
        
        # For approach reward calculation
        self.prev_dist = None  # Previous distance to target (for delta calculation)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state for a new episode.
        
        Phase-Specific Spawning:
        - Phase 1 & 2: Blue spawns safe (5000m alt, level), Red 5-10km ahead (stable drone)
        - Phase 3 & 4: Battle-box spawning (40-80km separation, random geometry)
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            tuple: (observation, info)
                observation: Initial observation for blue agent
                info: Dict containing red agent observation for self-play
        """
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)  # RNG for spawn position jitter
        self.core = AirCombatCore()        # Create fresh physics simulation

        # Clear entity tracking from previous episode
        self.blue_ids = []
        self.red_ids = []
        
        # Reset distance tracking for approach rewards
        self.prev_dist = None

        # === PHASE-SPECIFIC SPAWNING ===
        if self.phase in [1, 2]:
            # ===  PHASE 1 & 2: FLIGHT SCHOOL / PURSUIT ===
            # Simple spawning: Blue behind, Red ahead (stable drone)
            
            # Pick random center point in middle 60% of map
            center_lat = rng.uniform(0.3, 0.7)
            center_lon = rng.uniform(0.3, 0.7)
            center_lat_abs, center_lon_abs = self.map_limits.absolute_position(center_lat, center_lon)
            
            # Red drone flies straight ahead on random heading
            drone_heading = rng.uniform(0.0, 360.0)
            
            # Place red drone at random distance ahead (5-10km)
            red_dist_m = rng.uniform(5000, 10000)
            red_lat, red_lon = geodetic_direct(center_lat_abs, center_lon_abs, drone_heading, red_dist_m)
            
            # Phase-specific drone speed
            if self.phase == 1:
                red_speed = 100  # Very slow (almost stationary)
                red_alt = 5000   # Same altitude as blue
            else:  # Phase 2
                red_speed = 700  # Medium speed
                red_alt = 5000
            
            # Spawn red drone
            red_uid = self.core.spawn(red_lat, red_lon, drone_heading, red_speed, "red", "plane")
            self.core.entities[red_uid].pitch = 0.0
            self.core.entities[red_uid].roll = 0.0
            self.core.entities[red_uid].alt = red_alt
            self.red_ids.append(red_uid)
            
            # Blue spawns behind red, pointing at it
            blue_dist_behind_m = rng.uniform(3000, 6000)  # 3-6km behind
            blue_lat, blue_lon = geodetic_direct(
                red_lat, red_lon, 
                (drone_heading + 180) % 360,  # Opposite direction
                blue_dist_behind_m
            )
            
            # Blue heading: point toward red
            blue_heading = drone_heading  # Same direction as drone
            blue_speed = 600  # Moderate speed (faster than Phase 1 drone, slower than Phase 2)
            blue_alt = 5000   # Same altitude
            
            # Spawn blue agent
            blue_uid = self.core.spawn(blue_lat, blue_lon, blue_heading, blue_speed, "blue", "plane")
            self.core.entities[blue_uid].pitch = 0.0
            self.core.entities[blue_uid].roll = 0.0
            self.core.entities[blue_uid].alt = blue_alt
            self.blue_ids.append(blue_uid)
            
        else:
            # === PHASE 3 & 4: BATTLE BOX (Existing Logic) ===
            # Random head-on engagement with varied geometry
            
            # 1. Pick random center point in middle 60% of map
            center_lat = rng.uniform(0.2, 0.8)
            center_lon = rng.uniform(0.2, 0.8)
            
            # 2. Choose random separation distance (40-80km)
            separation_km = rng.uniform(40.0, 80.0)
            
            # 3. Choose random engagement axis (0-360°)
            axis_deg = rng.uniform(0.0, 360.0)
            
            # 4. Calculate team positions along the axis
            half_sep_lat, half_sep_lon = self.map_limits.absolute_position(center_lat, center_lon)
            
            # Blue team position: center + separation/2 along axis
            blue_center_lat, blue_center_lon = geodetic_direct(
                half_sep_lat, half_sep_lon, axis_deg, (separation_km / 2.0) * 1000.0
            )
            blue_heading = (axis_deg + 180) % 360  # Point toward red team
            
            # Red team position: center - separation/2 along axis
            red_center_lat, red_center_lon = geodetic_direct(
                half_sep_lat, half_sep_lon, (axis_deg + 180) % 360, (separation_km / 2.0) * 1000.0
            )
            red_heading = axis_deg  # Point toward blue team

            # Spawn blue agents
            for i in range(self.cfg.N_AGENTS):
                jitter_lat = rng.uniform(-0.01, 0.01)
                jitter_lon = rng.uniform(-0.01, 0.01)
                spacing_offset = (i - self.cfg.N_AGENTS / 2) * 0.02
                
                lat = blue_center_lat + jitter_lat
                lon = blue_center_lon + jitter_lon + spacing_offset
                
                uid = self.core.spawn(lat, lon, blue_heading, 900, "blue", "plane")
                self.core.entities[uid].pitch = 0.0
                self.core.entities[uid].roll = 0.0
                self.core.entities[uid].alt = 10000.0
                self.blue_ids.append(uid)

            # Spawn red enemies
            for i in range(self.cfg.N_ENEMIES):
                jitter_lat = rng.uniform(-0.01, 0.01)
                jitter_lon = rng.uniform(-0.01, 0.01)
                spacing_offset = (i - self.cfg.N_ENEMIES / 2) * 0.02
                
                lat = red_center_lat + jitter_lat
                lon = red_center_lon + jitter_lon + spacing_offset
                
                uid = self.core.spawn(lat, lon, red_heading, 900, "red", "plane")
                self.core.entities[uid].pitch = 0.0
                self.core.entities[uid].roll = 0.0
                self.core.entities[uid].alt = 10000.0
                self.red_ids.append(uid)

        # === RETURN INITIAL OBSERVATION ===
        info = {}
        if self.red_ids:
            info["red_obs"] = self._get_obs(self.red_ids[0])
        info["global_state"] = self._get_global_state()
        
        return self._get_obs(self.blue_ids[0]), info

    def set_kappa(self, k):
        """
        Update curriculum difficulty parameter.
        
        Called by training script to implement curriculum learning.
        Kappa controls AI opponent skill level:
        - k=0.0: Expert opponent (perfect execution)
        - k=1.0: Random/novice opponent (easy to defeat)
        
        Args:
            k: Difficulty parameter in [0, 1]
        """
        self.kappa = k
    
    def set_phase(self, phase_id):
        """
        Set curriculum phase (called by training script).
        
        Phase determines spawning logic, opponent behavior, and active rewards.
        
        Args:
            phase_id: Phase identifier (1, 2, 3, or 4)
        """
        self.phase = phase_id

    def _potential(self, x, x_mean, alpha):
        """
        Exponential potential field function for reward shaping (MDPI Paper).
        
        Implements the exponential potential function:
        Φ(x) = 1.0 - exp(-α * x)
        
        This creates a smooth curve that asymptotically approaches 1.0,
        providing reward shaping for continuous improvement rather than
        saturation at a target value.
        
        Args:
            x: Current value
            x_mean: Target/desired value (for compatibility, but not used in exponential form)
            alpha: Steepness parameter (higher = sharper transition)
            
        Returns:
            float: Potential value in approximately [0, 1]
        """
        # Exponential potential: 1 - exp(-α * x)
        # Clip to prevent overflow
        exponent = -alpha * x
        exponent = np.clip(exponent, -20, 20)
        
        return 1.0 - np.exp(exponent)

    def step(self, action, red_actions=None):
        """
        Execute one timestep of the environment.
        
        Processes actions, updates physics, calculates MDPI/SOTA rewards, and returns next state.
        Supports two modes:
        1. Self-Play: Concatenated action contains both blue and red agent actions
        2. AI Opponent: Single action for blue, red controlled by AI (kappa-based)
        
        Reward Structure (MDPI/SOTA):
        - Anti-Farming: Shaping rewards reduced 50x to prevent reward farming
        - Energy Management: Rewards high speed + altitude (prevents corner velocity trap)
        - Shot Penalty: -2.0 per missile to prevent spam
        - Kill Bonus: +100 for destroying enemy (the main objective)
        
        Args:
            action: Either single blue action (5D) or concatenated blue+red actions (10D)
            red_actions: Optional explicit red actions (legacy support)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # === 1. PREPARE ACTIONS ===
        # Build action dictionary mapping entity UID -> action array
        actions = {}
        agent_id = self.blue_ids[0]  # Primary blue agent (single-agent RL)

        # Case A: Concatenated Action (Self-Play / Model)
        # When training with self-play, action contains both blue and red actions
        # Format: [blue_roll, blue_g, blue_throttle, blue_fire, blue_cm, red_roll, red_g, ...]
        if len(action.shape) > 0 and action.shape[0] == 2 * self.cfg.ACTION_DIM:
            blue_action = action[:self.cfg.ACTION_DIM]   # First 5 values: blue agent
            red_action_in = action[self.cfg.ACTION_DIM:]  # Last 5 values: red agent
            if agent_id in self.core.entities: 
                actions[agent_id] = blue_action
            if self.red_ids and self.red_ids[0] in self.core.entities:
                actions[self.red_ids[0]] = red_action_in
                
        # Case B: Single Action (Scripted Opponent)
        # Blue action provided, red will be AI-controlled via kappa
        else:
            if agent_id in self.core.entities:
                actions[agent_id] = action
            # Red actions will be generated by core.step() automatically 
            # because they are missing from 'actions' dict (AI fills in gaps)
            
            # Inject Red Actions if provided explicitly (Legacy/Manual control)
            if red_actions is not None:
                if self.red_ids and self.red_ids[0] in self.core.entities:
                    if isinstance(red_actions, (np.ndarray, list)):
                        actions[self.red_ids[0]] = red_actions
                    elif isinstance(red_actions, dict):
                        actions.update(red_actions)

        # === PHASE 1 & 2: DRONE AI OVERRIDE ===
        # Force red agent to fly straight and level (stable drone behavior)
        if self.phase in [1, 2] and self.red_ids:
            red_id = self.red_ids[0]
            if red_id in self.core.entities and red_id not in actions:
                # Stable drone: no roll, no G-pull, medium throttle, no fire/cm
                # This creates a simple "practice target" for the agent to learn pursuit
                actions[red_id] = np.array([0.0, 0.0, 0.5, 0.0, 0.0])

        # === 2. STEP PHYSICS CORE ===
        # Advance simulation by DT seconds, passing kappa for AI opponent difficulty
        self.core.step(actions, self.kappa)
        
        # === 2.5 HARD DECK SAFETY CHECK ===
        # Terminate episode if agent drops below 2000m (before crash)
        # This saves computation time and provides clear penalty signal
        # Agent learns: "Low altitude = Death" much faster
        agent = self.core.entities.get(agent_id)
        if agent and agent.alt < 2000.0:
            reward = -100.0  # Massive penalty for floor violation
            terminated = True
            term_reason = "floor_violation"
            # Remove from simulation to stop physics
            del self.core.entities[agent_id]
            # Return immediately - no need to calculate other rewards
            info = {
                "termination_reason": term_reason,
                "red_obs": np.zeros(self.cfg.OBS_DIM, dtype=np.float32),
                "global_state": self._get_global_state()
            }
            return self._get_obs(agent_id), reward, terminated, False, info

        #=== 3. CALCULATE REWARDS (MDPI/SOTA Structure) ===
        # Reward philosophy: Sparse main rewards (+100 kill) with minimal shaping
        # to prevent "reward farming" where agent scores points without progressing
        reward = 0.0
        terminated = False  # Episode ends (success or failure)
        truncated = False   # Episode timeout (neither success nor failure)
        term_reason = "none"  # Logging: why episode ended

        # DEATH PENALTY: Agent crashed or was shot down
        if agent_id not in self.core.entities:
            # Agent no longer exists - determine cause of death
            death_event = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            if death_event:
                if death_event['type'] == 'crash':
                    reward = -50.0  # Hit the ground
                    term_reason = "crash"
                elif death_event['type'] == 'kill':
                    reward = -50.0  # Shot down by enemy
                    term_reason = "shot"
            else:
                # No death event found (shouldn't happen, but handle gracefully)
                reward = -50.0
                term_reason = "crash"
            
            terminated = True  # Episode over
        # ALIVE: Agent survived this timestep - calculate shaping and event rewards
        else:
            # === PHASE-GATED ADDITIVE REWARDS ===
            # Each phase ADDS new reward components without changing previous ones
            # This allows retraining Phase 3 agents back to Phase 1 if needed
            
            agent = self.core.entities[agent_id]
            
            # === BASE REWARDS (ALL PHASES) ===
            # 1. Survival: Very small to achieve ~6 total over 1200 steps
            reward += 0.005  # ~6.0 over full episode
            
            # Find nearest enemy for distance-based rewards
            nearest = None
            min_dist = float('inf')
            for e in self.core.entities.values():
                if e.team == "red":
                    d = geodetic_distance_km(agent.lat,  agent.lon, e.lat, e.lon)
                    if d < min_dist:
                        min_dist = d
                        nearest = e
            
            # === PHASE 1+ REWARDS (Flight & Approach) ===
            if self.phase >= 1 and nearest:
                # Approach reward: Reward for closing distance
                # Initialize prev_dist on first step
                if self.prev_dist is None:
                    self.prev_dist = min_dist
                
                # Calculate delta (positive when getting closer)
                approach_delta = self.prev_dist - min_dist
                # Scale to contribute ~6 over episode
                # At 0.01 scale, closing 10km gives 0.01 * 10 = 0.1 per step
                # Over ~60 steps of closing = 6.0 total
                approach_reward = approach_delta * 0.01
                reward += np.clip(approach_reward, -0.01, 0.01)
                
                # Update for next step
                self.prev_dist = min_dist
                
                # Stability bonus: Reward smooth flight (up to +3 total)
                # Penalize extreme attitudes but don't make it negative overall
                roll_penalty = np.clip(abs(agent.roll) * 0.005, 0, 0.01)
                pitch_penalty = np.clip(abs(agent.pitch) * 0.005, 0, 0.01)
                stability_bonus = 0.003 - roll_penalty - pitch_penalty  # Max +0.003/step ~= +3.6 over 1200
                reward += max(0, stability_bonus)  # Only add if positive (bonus, not penalty)
            
            # === PHASE 2+ REWARDS (Positioning) ===
            if self.phase >= 2 and nearest:
                dist_m = min_dist * 1000.0
                bearing = geodetic_bearing_deg(agent.lat, agent.lon, nearest.lat, nearest.lon)
                
                # ATA (Antenna Train Angle): How far off-nose the target is
                ata = abs((bearing - agent.heading + 180) % 360 - 180)
                mu = np.radians(ata)
                
                # AA (Aspect Angle): Target's heading relative to us
                bearing_to_me = (bearing + 180) % 360
                aa = abs((bearing_to_me - nearest.heading + 180) % 360 - 180)
                lam = np.radians(aa)
                
                # Aiming reward: Point nose at target
                r_aim = 0.5 * (1.0 - (mu / np.pi)) ** 2
                reward += r_aim * 0.01  # ~2-3 contribution
                
                # Geometry reward: Tail position advantage
                pos_potential = self._potential(lam/np.pi, 0.5, 18.0)
                r_geo = (1.0 - mu/np.pi) * pos_potential
                reward += r_geo * 0.01  # ~2-3 contribution
            
            # === PHASE 3+ REWARDS (Combat) ===
            if self.phase >= 3:
                # Lock reward (WEZ bonus)
                if nearest and min_dist < self.cfg.MISSILE_RANGE_KM:
                    bearing = geodetic_bearing_deg(agent.lat, agent.lon, nearest.lat, nearest.lon)
                    ata = abs((bearing - agent.heading + 180) % 360 - 180)
                    
                    if ata < 45.0:
                        _, is_locking = self.core.get_sensor_state(agent_id, nearest.uid)
                        if is_locking:
                            lock_fov_deg = self.cfg.RADAR_FOV_DEG * 0.80
                            ata_quality = 1.0 - (ata / lock_fov_deg)
                            scaled_lock_reward = 0.05 * ata_quality
                            reward += scaled_lock_reward
                
                # Event rewards: Kills
                for ev in self.core.events:
                    # NO firing penalty (consistent across phases)
                    # Ammo limit naturally constrains firing rate
                    
                    if ev['type'] == 'kill':
                        if ev['killer'] in self.blue_ids:
                            reward += 50.0  # Main objective achieved!
                        if ev['victim'] in self.blue_ids:
                            reward -= 50.0
                
                # Win condition: All enemies destroyed
                reds_alive = sum(1 for e in self.core.entities.values() if e.team == "red")
                if reds_alive == 0:
                    reward += 50.0  # Mission success bonus
                    # Ammo retention bonus
                    if agent_id in self.core.entities:
                        missiles_remaining = self.core.entities[agent_id].ammo
                        reward += missiles_remaining * 2.0
                    terminated = True
                    term_reason = "win"

        # === 4. CHECK TIME LIMIT ===
        # Episode truncated if maximum duration reached (neither win nor loss)
        if self.core.time >= self.cfg.MAX_DURATION_SEC:
            truncated = True
            term_reason = "timeout"

        # === 5. BUILD INFO DICT ===
        info = {}
        info["termination_reason"] = term_reason  # For logging/debugging
        
        # Include red observation for self-play training
        if self.red_ids and self.red_ids[0] in self.core.entities:
            info["red_obs"] = self._get_obs(self.red_ids[0])
        else:
            # Red agent is dead - provide zero placeholder
            info["red_obs"] = np.zeros(self.cfg.OBS_DIM, dtype=np.float32)
        
        # CTDE: Include global state for centralized critic
        info["global_state"] = self._get_global_state()

        return self._get_obs(agent_id), reward, terminated, truncated, info

    def _get_obs(self, ego_id):
        """
        Generate observation for a given entity with sensor simulation ("fog of war").
        
        Observation structure:
        1. Ego vector (self state - always fully visible)
        2. Other entity vectors (friends/enemies/missiles - subject to sensors)
        3. Zero-padding to fixed observation dimension
        
        Sensor Simulation:
        - Radar: Can see enemies if in range + FOV + not in Doppler notch
        - RWR (Radar Warning Receiver): Detects when being locked by enemy
        - MAWS (Missile Approach Warning System): Detects incoming missiles
        
        Args:
            ego_id: UID of the observing entity
            
        Returns:
            np.ndarray: Flattened observation vector (shape: OBS_DIM)
        """
        vecs = []  # List of feature vectors for each entity

        # === 1. EGO VECTOR (Self State) ===
        # Agent always knows its own state perfectly
        if ego_id in self.core.entities:
            vecs.append(self._vectorize(self.core.entities[ego_id], ego_id, True))
        else:
            # Agent is dead - provide zero placeholder
            vecs.append(np.zeros(self.cfg.FEAT_DIM, dtype=np.float32))

        # === 2. OTHER ENTITIES (Friends, Enemies, Missiles) ===
        # Visibility determined by sensor simulation
        for uid, ent in self.core.entities.items():
            if uid == ego_id: continue  # Skip self (already included above)

            # === SENSOR SIMULATION (Fog of War) ===
            visible = True      # Can we see this entity with radar?
            rwr_active = False  # Is this entity locking us?

            if ego_id in self.core.entities and ent.team != "blue":
                # Enemy entity - check radar visibility
                # Requires: range < max, in FOV, not in Doppler notch
                visible, _ = self.core.get_sensor_state(ego_id, uid)

                # RWR Check: Is enemy locking us with their radar?
                # Even if we can't see them, we know if they're locking us
                _, locking_me = self.core.get_sensor_state(uid, ego_id)
                if locking_me:
                    rwr_active = True  # Warning: "Spiked!" (being locked)

            # === BUILD ENTITY VECTOR ===
            if visible:
                # Entity is on radar - provide full state information
                v = self._vectorize(ent, ego_id, False)
                # If enemy is also locking us, set RWR flag in vector
                if rwr_active: 
                    v[12] = 1.0  # RWR signal index
                vecs.append(v)
            elif rwr_active:
                # Can't see entity, but they're locking us (RWR only)
                # Provide "ghost" vector: only RWR signal, all else zero
                # Agent knows threat direction but not exact position/state
                v = np.zeros(self.cfg.FEAT_DIM, dtype=np.float32)
                v[12] = 1.0  # RWR signal: "Someone is locking you!"
                vecs.append(v)
            else:
                # Entity is hidden (out of sensor range/FOV/notch)
                # Provide zero vector: agent has no information
                vecs.append(np.zeros(self.cfg.FEAT_DIM, dtype=np.float32))

        # === 3. FLATTEN AND PAD ===
        # Convert list of vectors to single flat array
        flat = []
        for v in vecs: 
            flat.extend(v)

        # Truncate if too many entities (should not happen with proper config)
        if len(flat) > self.cfg.OBS_DIM:
            flat = flat[:self.cfg.OBS_DIM]

        # Zero-pad if too few entities (most common case)
        if len(flat) < self.cfg.OBS_DIM:
            flat.extend([0.0] * (self.cfg.OBS_DIM - len(flat)))

        return np.array(flat, dtype=np.float32)
    
    def _get_global_state(self):
        """Generate privileged global state for centralized critic (CT DE/MAPPO)."""
        flat = []
        for e in self.core.entities.values():
            flat.extend(self._vectorize(e, ego_id=None, is_ego=False))
        while len(flat) < self.cfg.OBS_DIM:
            flat.extend([0.0] * self.cfg.FEAT_DIM)
        return np.array(flat[:self.cfg.OBS_DIM], dtype=np.float32)

    def _vectorize(self, e, ego_id, is_ego):
        """
        Convert an entity to a normalized feature vector with MDPI geometry features.
        
        Feature vector (20 dimensions):
        [0-1]   Position: lat_norm, lon_norm (normalized to map bounds)
        [2-3]   Heading: cos(heading), sin(heading) (periodic encoding)
        [4]     Speed: speed/1000 (normalized to ~Mach 3)
        [5]     Team: +1.0=blue, -1.0=red
        [6]     Type: 1.0=missile, 0.0=plane
        [7]     Is Ego: 1.0=self, 0.0=other
        [8-9]   Roll: cos(roll), sin(roll) (periodic encoding)
        [10-11] Pitch: cos(pitch), sin(pitch) (periodic encoding)
        [12]    RWR: 1.0=being locked, 0.0=clear
        [13]    MAWS: 1.0=missile incoming, 0.0=clear
        [14]    Altitude: alt/10000 (normalized to 10km)
        [15]    Fuel: fuel fraction [0, 1]
        [16]    Ammo: ammo/MAX_MISSILES (normalized)
        [17]    ATA: Antenna Train Angle - angle off ego's nose [0-1]
        [18]    AA: Aspect Angle - angle off target's tail [0-1]
        [19]    Closure Rate: Approximate closing speed [normalized]
        
        Args:
            e: Entity to vectorize
            ego_id: UID of observing entity (for MAWS and geometry calculations)
            is_ego: Whether this is the ego entity (self)
            
        Returns:
            list: Feature vector (length FEAT_DIM=20)
        """
        # Normalize position to [0, 1] range within map bounds
        lat_n, lon_n = self.map_limits.relative_position(e.lat, e.lon)
        hr = math.radians(e.heading)  # Convert heading to radians

        # === SENSOR SIGNALS ===
        rwr_signal = 0.0   # Radar Warning Receiver (set by caller if being locked)
        maws_signal = 0.0  # Missile Approach Warning System

        # MAWS: Detecting incoming missiles
        # If this entity is a missile targeting me, set MAWS signal
        if e.type == "missile" and e.target_id == ego_id:
            maws_signal = 1.0  # Warning: missile inbound!

        # === MDPI GEOMETRY FEATURES (ATA, AA, Closure Rate) ===
        ata_norm = 0.0      # Antenna Train Angle (normalized)
        aa_norm = 0.0       # Aspect Angle (normalized)
        closure_norm = 0.0  # Closure Rate (normalized)
        
        # Only calculate geometry for non-ego entities when ego exists
        if not is_ego and ego_id in self.core.entities:
            ego = self.core.entities[ego_id]
            
            # ATA (Antenna Train Angle): Angle off ego's nose to target
            # How far off-axis the target is from ego's heading
            bearing_to_target = geodetic_bearing_deg(ego.lat, ego.lon, e.lat, e.lon)
            ata_deg = abs((bearing_to_target - ego.heading + 180) % 360 - 180)
            ata_norm = ata_deg / 180.0  # Normalize to [0, 1]
            
            # AA (Aspect Angle): Angle off target's tail to ego
            # Classical BFM parameter: 0° = on target's six, 180° = head-on
            bearing_to_ego = geodetic_bearing_deg(e.lat, e.lon, ego.lat, ego.lon)
            aa_deg = abs((bearing_to_ego - e.heading + 180) % 360 - 180)
            aa_norm = aa_deg / 180.0  # Normalize to [0, 1]
            
            # Closure Rate: Approximate closing speed
            # Positive = closing, Negative = opening
            # Calculate component of relative velocity along line-of-sight
            # Simplified: use radial components of both velocities
            ego_radial = ego.speed * math.cos(math.radians(ata_deg))  # Ego's speed toward target
            tgt_radial = e.speed * math.cos(math.radians(aa_deg))      # Target's speed toward ego
            closure_rate_knots = ego_radial + tgt_radial
            # Normalize to [-1, 1] range (assuming max closure ~2000 knots)
            closure_norm = np.clip(closure_rate_knots / 2000.0, -1.0, 1.0)

        # === BUILD FEATURE VECTOR ===
        return [
            # Position (geodetic, normalized to map)
            lat_n, lon_n,
            # Heading (periodic encoding: sin/cos prevents discontinuity at 0°/360°)
            np.cos(hr), np.sin(hr),
            # Speed (normalized to ~1000 km/h max display speed)
            e.speed / 1000.0,
            # Team affiliation (binary: friendly vs enemy)
            1.0 if e.team == "blue" else -1.0,
            # Entity type (plane vs missile)
            1.0 if e.type == "missile" else 0.0,
            # Ego flag (is this me?)
            1.0 if is_ego else 0.0,
            # Roll angle (periodic encoding)
            np.cos(e.roll), np.sin(e.roll),
            # Pitch angle (periodic encoding)
            np.cos(e.pitch), np.sin(e.pitch),
            # RWR signal (being tracked/locked by enemy)
            rwr_signal,
            # MAWS signal (incoming missile warning)
            maws_signal,
            # Altitude (normalized to 10km)
            e.alt / 10000.0,
            # Fuel remaining (fraction)
            e.fuel,
            # Ammunition remaining (normalized)
            e.ammo / float(self.cfg.MAX_MISSILES),
            # MDPI Geometry Features
            ata_norm,      # Antenna Train Angle
            aa_norm,       # Aspect Angle
            closure_norm,  # Closure Rate
            # Agent ID One-Hot
            *self._get_agent_id_onehot(e, ego_id, is_ego)
        ]
    
    def _get_agent_id_onehot(self, e, ego_id, is_ego):
        """Generate one-hot agent ID."""
        agent_id = [0.0] * self.cfg.MAX_TEAM_SIZE
        if is_ego and ego_id in self.blue_ids:
            idx = self.blue_ids.index(ego_id)
            if idx < self.cfg.MAX_TEAM_SIZE:
                agent_id[idx] = 1.0
        elif e.team == "blue" and e.uid in self.blue_ids:
            idx = self.blue_ids.index(e.uid)
            if idx < self.cfg.MAX_TEAM_SIZE:
                agent_id[idx] = 1.0
        return agent_id

    def render(self):
        from src.utils.scenario_plotter import ScenarioPlotter, PlotConfig, Airplane, Missile
        from src.utils.map_limits import MapLimits
        import matplotlib.pyplot as plt

        # === DYNAMIC MAP LIMITS (Camera follows the action) ===
        # Calculate bounds from active entity positions instead of using fixed limits
        if self.core.entities:
            # Find min/max lat/lon from all active entities
            lats = [e.lat for e in self.core.entities.values()]
            lons = [e.lon for e in self.core.entities.values()]
            
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Add 20% buffer around the bounds for visibility
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            buffer_lat = max(lat_range * 0.2, 0.1)  # Minimum 0.1° buffer
            buffer_lon = max(lon_range * 0.2, 0.1)
            
            # Dynamic render limits that follow the fight
            dynamic_limits = MapLimits(
                min_lon - buffer_lon, min_lat - buffer_lat,
                max_lon + buffer_lon, max_lat + buffer_lat
            )
        else:
            # No entities: fallback to fixed limits
            dynamic_limits = self.render_limits

        # Create or update renderer with dynamic limits
        if self.renderer is None:
            p_cfg = PlotConfig()
            p_cfg.units_scale = 20.0
            self.renderer = ScenarioPlotter(dynamic_limits, dpi=100, config=p_cfg)
        else:
            # Update existing renderer's map limits
            self.renderer.map_limits = dynamic_limits

        drawables = []
        for e in self.core.entities.values():
            c = (0, 0, 1, 1) if e.team == "blue" else (1, 0, 0, 1)

            if e.type == "missile":
                drawables.append(Missile(e.lat, e.lon, e.heading, fill_color=c, zorder=10))
            else:
                # Detailed Info Text
                txt = f"{e.uid}\\nA:{int(e.alt)}\\nF:{int(e.fuel * 100)}%"
                if e.cm_active: txt += "\\nCM!"

                drawables.append(Airplane(e.lat, e.lon, e.heading, fill_color=c, info_text=txt, zorder=5))

        try:
            fname = f"temp_render_{self.blue_ids[0] if self.blue_ids else 0}.png"
            self.renderer.to_png(fname, drawables)
            return (plt.imread(fname) * 255).astype(np.uint8)
        except:
            return np.zeros((400, 400, 3), dtype=np.uint8)