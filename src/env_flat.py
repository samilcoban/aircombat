# Import required libraries for RL environment, numerical operations, and simulation utilities
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from config import Config
from src.core_flat import AirCombatCore, dist_2d, bearing_deg
from src.utils.map_limits_flat import MapLimits


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
        # Defines the Cartesian boundaries for the simulation arena (Meters)
        self.map_limits = MapLimits(*self.cfg.MAP_LIMITS)

        # Tactical Zoom for Rendering: Zoomed-in view centered on combat area
        center_x = (self.map_limits.min_x + self.map_limits.max_x) / 2.0
        center_y = (self.map_limits.min_y + self.map_limits.max_y) / 2.0
        zoom = 15000.0  # Zoom factor (15km radius)
        self.render_limits = MapLimits(
            center_x - zoom, center_x + zoom,
            center_y - zoom, center_y + zoom
        )

        # === ACTION SPACE ===
        # Continuous 5D action vector normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.ACTION_DIM,), dtype=np.float32
        )

        # === OBSERVATION SPACE ===
        # Flattened feature vectors for all visible entities
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
        self.phase_progress = 0.0 # 0.0 to 1.0 within current phase
        
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
            center_x_rel = rng.uniform(0.3, 0.7)
            center_y_rel = rng.uniform(0.3, 0.7)
            center_x, center_y = self.map_limits.absolute_position(center_x_rel, center_y_rel)
            
            # Red drone flies straight ahead on random heading
            drone_heading = rng.uniform(0.0, 360.0)
            
            # Place red drone at random distance ahead (5-10km)
            red_dist_m = rng.uniform(5000, 10000)
            
            # Cartesian projection
            red_x = center_x + red_dist_m * math.cos(math.radians(drone_heading))
            red_y = center_y + red_dist_m * math.sin(math.radians(drone_heading))
            
            # Phase-specific drone speed (Annealed)
            if self.phase == 1:
                # Phase 1: 100 km/h -> 300 km/h
                red_speed = 100.0 + (200.0 * self.phase_progress)
                red_alt = 5000   # Same altitude as blue
            else:  # Phase 2
                # Phase 2: 300 km/h -> 700 km/h
                red_speed = 300.0 + (400.0 * self.phase_progress)
                red_alt = 5000
            
            # Spawn red drone
            red_uid = self.core.spawn(red_x, red_y, drone_heading, red_speed, "red", "plane")
            self.core.entities[red_uid].pitch = 0.0
            self.core.entities[red_uid].roll = 0.0
            self.core.entities[red_uid].alt = red_alt
            self.red_ids.append(red_uid)
            
            # Blue spawns behind red, pointing at it
            blue_dist_behind_m = rng.uniform(3000, 6000)  # 3-6km behind
            blue_heading = drone_heading  # Same direction as drone
            
            # Position blue behind red
            # Opposite direction of heading is heading + 180
            blue_x = red_x + blue_dist_behind_m * math.cos(math.radians(blue_heading + 180))
            blue_y = red_y + blue_dist_behind_m * math.sin(math.radians(blue_heading + 180))
            
            blue_speed = 600  # Moderate speed
            blue_alt = 5000   # Same altitude
            
            # Spawn blue agent
            blue_uid = self.core.spawn(blue_x, blue_y, blue_heading, blue_speed, "blue", "plane")
            self.core.entities[blue_uid].pitch = 0.0
            self.core.entities[blue_uid].roll = 0.0
            self.core.entities[blue_uid].alt = blue_alt
            self.blue_ids.append(blue_uid)
            
        else:
            # === PHASE 3 & 4: BATTLE BOX ===
            # Random head-on engagement with varied geometry
            
            # 1. Pick random center point in middle 60% of map
            center_x_rel = rng.uniform(0.2, 0.8)
            center_y_rel = rng.uniform(0.2, 0.8)
            center_x, center_y = self.map_limits.absolute_position(center_x_rel, center_y_rel)
            
            # 2. Choose random separation distance (Annealed)
            # Phase 3: Start closer (20km) and expand to 80km
            # Phase 4: Full random 40-80km
            if self.phase == 3:
                 min_sep = 20000.0
                 max_sep = 20000.0 + (60000.0 * self.phase_progress) # 20k -> 80k
                 separation_m = rng.uniform(min_sep, max_sep)
            else:
                 separation_m = rng.uniform(40000.0, 80000.0)
            
            # 3. Choose random engagement axis (0-360°)
            axis_deg = rng.uniform(0.0, 360.0)
            
            # 4. Calculate team positions along the axis
            # Blue team position: center + separation/2 along axis
            blue_center_x = center_x + (separation_m / 2.0) * math.cos(math.radians(axis_deg))
            blue_center_y = center_y + (separation_m / 2.0) * math.sin(math.radians(axis_deg))
            blue_heading = (axis_deg + 180) % 360  # Point toward red team
            
            # Red team position: center - separation/2 along axis
            red_center_x = center_x + (separation_m / 2.0) * math.cos(math.radians(axis_deg + 180))
            red_center_y = center_y + (separation_m / 2.0) * math.sin(math.radians(axis_deg + 180))
            red_heading = axis_deg  # Point toward blue team

            # Spawn blue agents
            for i in range(self.cfg.N_AGENTS):
                jitter_x = rng.uniform(-1000, 1000)
                jitter_y = rng.uniform(-1000, 1000)
                spacing_offset = (i - self.cfg.N_AGENTS / 2) * 2000.0
                
                x = blue_center_x + jitter_x
                y = blue_center_y + jitter_y + spacing_offset
                
                uid = self.core.spawn(x, y, blue_heading, 900, "blue", "plane")
                self.core.entities[uid].pitch = 0.0
                self.core.entities[uid].roll = 0.0
                self.core.entities[uid].alt = 10000.0
                self.blue_ids.append(uid)

            # Spawn red enemies
            for i in range(self.cfg.N_ENEMIES):
                jitter_x = rng.uniform(-1000, 1000)
                jitter_y = rng.uniform(-1000, 1000)
                spacing_offset = (i - self.cfg.N_ENEMIES / 2) * 2000.0
                
                x = red_center_x + jitter_x
                y = red_center_y + jitter_y + spacing_offset
                
                uid = self.core.spawn(x, y, red_heading, 900, "red", "plane")
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
        self.kappa = k
    
    def set_phase(self, phase_id, progress=0.0):
        self.phase = phase_id
        self.phase_progress = np.clip(progress, 0.0, 1.0)

    def _potential(self, x, x_mean, alpha):
        """Exponential potential field function for reward shaping."""
        exponent = -alpha * x
        exponent = np.clip(exponent, -20, 20)
        return 1.0 - np.exp(exponent)

    def step(self, action, red_actions=None):
        """
        Execute one timestep of the environment.
        """
        # === 1. PREPARE ACTIONS ===
        actions = {}
        agent_id = self.blue_ids[0]  # Primary blue agent

        # Case A: Concatenated Action (Self-Play / Model)
        if len(action.shape) > 0 and action.shape[0] == 2 * self.cfg.ACTION_DIM:
            blue_action = action[:self.cfg.ACTION_DIM]
            red_action_in = action[self.cfg.ACTION_DIM:]
            if agent_id in self.core.entities: 
                actions[agent_id] = blue_action
            if self.red_ids and self.red_ids[0] in self.core.entities:
                actions[self.red_ids[0]] = red_action_in
                
        # Case B: Single Action (Scripted Opponent)
        else:
            if agent_id in self.core.entities:
                actions[agent_id] = action
            
            if red_actions is not None:
                if self.red_ids and self.red_ids[0] in self.core.entities:
                    if isinstance(red_actions, (np.ndarray, list)):
                        actions[self.red_ids[0]] = red_actions
                    elif isinstance(red_actions, dict):
                        actions.update(red_actions)

        # === PHASE 1 & 2: DRONE AI OVERRIDE ===
        if self.phase in [1, 2] and self.red_ids:
            red_id = self.red_ids[0]
            if red_id in self.core.entities and red_id not in actions:
                actions[red_id] = np.array([0.0, 0.0, 0.5, 0.0, 0.0])

        # === 2. STEP PHYSICS CORE ===
        self.core.step(actions, self.kappa)
        
        # === 2.5 HARD DECK SAFETY CHECK ===
        agent = self.core.entities.get(agent_id)
        if agent and agent.alt < 2000.0:
            reward = -100.0
            terminated = True
            term_reason = "floor_violation"
            del self.core.entities[agent_id]
            info = {
                "termination_reason": term_reason,
                "red_obs": np.zeros(self.cfg.OBS_DIM, dtype=np.float32),
                "global_state": self._get_global_state()
            }
            return self._get_obs(agent_id), reward, terminated, False, info

        #=== 3. CALCULATE REWARDS ===
        reward = 0.0
        terminated = False
        truncated = False
        term_reason = "none"

        # DEATH PENALTY
        if agent_id not in self.core.entities:
            death_event = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            if death_event:
                if death_event['type'] == 'crash':
                    reward = -50.0
                    term_reason = "crash"
                elif death_event['type'] == 'kill':
                    reward = -50.0
                    term_reason = "shot"
            else:
                reward = -50.0
                term_reason = "crash"
            terminated = True
        # ALIVE
        else:
            agent = self.core.entities[agent_id]
            
            # === BASE REWARDS ===
            reward += 0.005
            
            # === ENERGY PENALTY (Physics Tax) ===
            # Penalize high-G maneuvers that bleed energy
            # This prevents the "seizure pilot" behavior of constant spiraling
            current_g = agent.g_load
            energy_penalty = 0.0
            if current_g > 1.5:
                # Non-linear penalty: 2G=small, 9G=huge
                # At 2G: penalty ≈ -0.004
                # At 6G: penalty ≈ -0.036
                # At 9G: penalty ≈ -0.081
                energy_penalty = -0.001 * (current_g ** 2)
            reward += energy_penalty
            
            # Find nearest enemy
            nearest = None
            min_dist_m = float('inf')
            for e in self.core.entities.values():
                if e.team == "red":
                    d = dist_2d(agent.x, agent.y, e.x, e.y)
                    if d < min_dist_m:
                        min_dist_m = d
                        nearest = e
            
            min_dist_km = min_dist_m / 1000.0

            # === PHASE 1+ REWARDS (Flight & Approach) ===
            if self.phase >= 1 and nearest:
                # Approach reward: Reduced scale to prevent overpowering energy conservation
                # Target +3 over episode (reduced from +6)
                if self.prev_dist is None:
                    self.prev_dist = min_dist_km
                
                approach_delta = self.prev_dist - min_dist_km
                approach_reward = approach_delta * 0.5  # REDUCED from 1.0 to balance with energy penalty
                reward += np.clip(approach_reward, -0.05, 0.05)  # REDUCED clip from ±0.1 to ±0.05
                
                self.prev_dist = min_dist_km
                
                # Stability bonus: Target +3.6 max over 1200 steps
                roll_penalty = np.clip(abs(agent.roll) * 0.005, 0, 0.01)
                pitch_penalty = np.clip(abs(agent.pitch) * 0.005, 0, 0.01)
                stability_bonus = 0.003 - roll_penalty - pitch_penalty
                reward += max(0, stability_bonus)
            
            # === PHASE 2+ REWARDS (Positioning) ===
            if self.phase >= 2 and nearest:
                bearing = bearing_deg(agent.x, agent.y, nearest.x, nearest.y)
                
                # ATA (Angle To Attack)
                ata = abs((bearing - agent.heading + 180) % 360 - 180)
                mu = np.radians(ata)
                
                # AA (Aspect Angle)
                bearing_to_me = (bearing + 180) % 360
                aa = abs((bearing_to_me - nearest.heading + 180) % 360 - 180)
                lam = np.radians(aa)
                
                # Aiming reward: Target +2-3 over episode
                r_aim = 0.5 * (1.0 - (mu / np.pi)) ** 2
                reward += r_aim * 0.05  # FIXED: Was 0.01 (100x too small)
                
                # Geometry reward: Target +2-3 over episode
                pos_potential = self._potential(lam/np.pi, 0.5, 18.0)
                r_geo = (1.0 - mu/np.pi) * pos_potential
                reward += r_geo * 0.05  # FIXED: Was 0.01 (100x too small)
            
            # === PHASE 3+ REWARDS (Combat) ===
            if self.phase >= 3:
                # Lock reward
                if nearest and min_dist_km < self.cfg.MISSILE_RANGE_KM:
                    bearing = bearing_deg(agent.x, agent.y, nearest.x, nearest.y)
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
                    if ev['type'] == 'kill':
                        if ev['killer'] in self.blue_ids:
                            reward += 50.0
                        if ev['victim'] in self.blue_ids:
                            reward -= 50.0
                
                # Win condition
                reds_alive = sum(1 for e in self.core.entities.values() if e.team == "red")
                if reds_alive == 0:
                    reward += 50.0
                    if agent_id in self.core.entities:
                        missiles_remaining = self.core.entities[agent_id].ammo
                        reward += missiles_remaining * 2.0
                    terminated = True
                    term_reason = "win"

        # === 4. CHECK TIME LIMIT ===
        if self.core.time >= self.cfg.MAX_DURATION_SEC:
            truncated = True
            term_reason = "timeout"

        # === 5. BUILD INFO DICT ===
        info = {}
        info["termination_reason"] = term_reason
        
        if self.red_ids and self.red_ids[0] in self.core.entities:
            info["red_obs"] = self._get_obs(self.red_ids[0])
        else:
            info["red_obs"] = np.zeros(self.cfg.OBS_DIM, dtype=np.float32)
        
        info["global_state"] = self._get_global_state()

        return self._get_obs(agent_id), reward, terminated, truncated, info

    def _get_obs(self, ego_id):
        """Generate observation for a given entity."""
        vecs = []

        # === 1. EGO VECTOR ===
        if ego_id in self.core.entities:
            vecs.append(self._vectorize(self.core.entities[ego_id], ego_id, True))
        else:
            vecs.append(np.zeros(self.cfg.FEAT_DIM, dtype=np.float32))

        # === 2. OTHER ENTITIES ===
        for uid, ent in self.core.entities.items():
            if uid == ego_id: continue

            visible = True
            rwr_active = False

            if ego_id in self.core.entities and ent.team != "blue":
                visible, _ = self.core.get_sensor_state(ego_id, uid)
                _, locking_me = self.core.get_sensor_state(uid, ego_id)
                if locking_me:
                    rwr_active = True

            if visible:
                v = self._vectorize(ent, ego_id, False)
                if rwr_active: 
                    v[12] = 1.0
                vecs.append(v)
            elif rwr_active:
                v = np.zeros(self.cfg.FEAT_DIM, dtype=np.float32)
                v[12] = 1.0
                vecs.append(v)
            else:
                vecs.append(np.zeros(self.cfg.FEAT_DIM, dtype=np.float32))

        # === 3. FLATTEN AND PAD ===
        flat = []
        for v in vecs: 
            flat.extend(v)

        if len(flat) > self.cfg.OBS_DIM:
            flat = flat[:self.cfg.OBS_DIM]
        if len(flat) < self.cfg.OBS_DIM:
            flat.extend([0.0] * (self.cfg.OBS_DIM - len(flat)))

        return np.array(flat, dtype=np.float32)
    
    def _get_global_state(self):
        """Generate privileged global state."""
        flat = []
        for e in self.core.entities.values():
            flat.extend(self._vectorize(e, ego_id=None, is_ego=False))
        while len(flat) < self.cfg.OBS_DIM:
            flat.extend([0.0] * self.cfg.FEAT_DIM)
        return np.array(flat[:self.cfg.OBS_DIM], dtype=np.float32)

    def _vectorize(self, e, ego_id, is_ego):
        """Convert an entity to a normalized feature vector."""
        # Normalize position to [0, 1] range within map bounds
        x_n, y_n = self.map_limits.relative_position(e.x, e.y)
        hr = math.radians(e.heading)

        rwr_signal = 0.0
        maws_signal = 0.0

        if e.type == "missile" and e.target_id == ego_id:
            maws_signal = 1.0

        # === MDPI GEOMETRY FEATURES ===
        ata_norm = 0.0
        aa_norm = 0.0
        closure_norm = 0.0
        
        if not is_ego and ego_id in self.core.entities:
            ego = self.core.entities[ego_id]
            
            bearing_to_target = bearing_deg(ego.x, ego.y, e.x, e.y)
            ata_deg = abs((bearing_to_target - ego.heading + 180) % 360 - 180)
            ata_norm = ata_deg / 180.0
            
            bearing_to_ego = bearing_deg(e.x, e.y, ego.x, ego.y)
            aa_deg = abs((bearing_to_ego - e.heading + 180) % 360 - 180)
            aa_norm = aa_deg / 180.0
            
            ego_radial = ego.speed * math.cos(math.radians(ata_deg))
            tgt_radial = e.speed * math.cos(math.radians(aa_deg))
            closure_rate_knots = ego_radial + tgt_radial
            closure_norm = np.clip(closure_rate_knots / 2000.0, -1.0, 1.0)

        return [
            x_n, y_n,
            np.cos(hr), np.sin(hr),
            e.speed / 1000.0,
            1.0 if e.team == "blue" else -1.0,
            1.0 if e.type == "missile" else 0.0,
            1.0 if is_ego else 0.0,
            np.cos(e.roll), np.sin(e.roll),
            np.cos(e.pitch), np.sin(e.pitch),
            rwr_signal,
            maws_signal,
            e.alt / 10000.0,
            e.fuel,
            e.ammo / float(self.cfg.MAX_MISSILES),
            ata_norm,
            aa_norm,
            closure_norm,
            *self._get_agent_id_onehot(e, ego_id, is_ego)
        ]
    
    def _get_agent_id_onehot(self, e, ego_id, is_ego):
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
        # Placeholder for 2D rendering - requires updating ScenarioPlotter
        # For now, return a blank image to prevent crash
        return np.zeros((600, 800, 3), dtype=np.uint8)