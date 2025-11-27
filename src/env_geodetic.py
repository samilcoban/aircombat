# Import required libraries for RL environment, numerical operations, and simulation utilities
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from config import Config
from src.core import AirCombatCore
from src.utils.map_limits import MapLimits
from src.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


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
        # Defines the geodetic boundaries for the simulation arena (Lat/Lon)
        self.map_limits = MapLimits(*self.cfg.MAP_LIMITS)

        # Tactical Zoom for Rendering: Zoomed-in view centered on combat area
        center_lat = (self.map_limits.bottom_lat + self.map_limits.top_lat) / 2.0
        center_lon = (self.map_limits.left_lon + self.map_limits.right_lon) / 2.0
        zoom = 0.15  # Zoom factor (approx 15km radius)
        self.render_limits = MapLimits(
            center_lon - zoom, center_lat - zoom,
            center_lon + zoom, center_lat + zoom
        )

        # === ACTION SPACE ===
        # Continuous 5D action vector normalized to [-1, 1]
        # [Roll Rate, G-Pull, Throttle, Fire, Countermeasures]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.ACTION_DIM,), dtype=np.float32
        )

        # === OBSERVATION SPACE ===
        # Flattened feature vectors for all visible entities (Ego + Enemies + Missiles)
        # Size = OBS_DIM (fixed size for neural network input)
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
            center_lat_rel = rng.uniform(0.3, 0.7)
            center_lon_rel = rng.uniform(0.3, 0.7)
            center_lat, center_lon = self.map_limits.absolute_position(center_lat_rel, center_lon_rel)
            
            # Red drone flies straight ahead on random heading
            drone_heading = rng.uniform(0.0, 360.0)
            
            # Place red drone at random distance ahead (5-10km)
            # 1 degree lat approx 111km -> 0.045 to 0.09 degrees
            red_dist_deg = rng.uniform(0.045, 0.09)
            
            # Simple flat-earth approximation for spawning offset (valid for small distances)
            red_lat = center_lat + red_dist_deg * math.cos(math.radians(drone_heading))
            red_lon = center_lon + red_dist_deg * math.sin(math.radians(drone_heading))
            
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
            blue_dist_behind_deg = rng.uniform(0.027, 0.054)  # 3-6km behind
            blue_heading = drone_heading  # Same direction as drone
            
            # Position blue behind red
            # Opposite direction of heading is heading + 180
            blue_lat = red_lat + blue_dist_behind_deg * math.cos(math.radians(blue_heading + 180))
            blue_lon = red_lon + blue_dist_behind_deg * math.sin(math.radians(blue_heading + 180))
            
            blue_speed = 600  # Moderate speed
            blue_alt = 5000   # Same altitude
            
            # Spawn blue agent
            blue_uid = self.core.spawn(blue_lat, blue_lon, blue_heading, blue_speed, "blue", "plane")
            self.core.entities[blue_uid].pitch = 0.0
            self.core.entities[blue_uid].roll = 0.0
            self.core.entities[blue_uid].alt = blue_alt
            self.blue_ids.append(blue_uid)
            
        else:
            # === PHASE 3 & 4: BATTLE BOX ===
            # Random head-on engagement with varied geometry
            
            # 1. Pick random center point in middle 60% of map
            center_lat_rel = rng.uniform(0.2, 0.8)
            center_lon_rel = rng.uniform(0.2, 0.8)
            center_lat, center_lon = self.map_limits.absolute_position(center_lat_rel, center_lon_rel)
            
            # 2. Choose random separation distance (40-80km)
            # 1 deg ~ 111km -> 0.36 to 0.72 degrees
            separation_deg = rng.uniform(0.36, 0.72)
            
            # 3. Choose random engagement axis (0-360°)
            axis_deg = rng.uniform(0.0, 360.0)
            
            # 4. Calculate team positions along the axis
            # Blue team position: center + separation/2 along axis
            blue_center_lat = center_lat + (separation_deg / 2.0) * math.cos(math.radians(axis_deg))
            blue_center_lon = center_lon + (separation_deg / 2.0) * math.sin(math.radians(axis_deg))
            blue_heading = (axis_deg + 180) % 360  # Point toward red team
            
            # Red team position: center - separation/2 along axis
            red_center_lat = center_lat + (separation_deg / 2.0) * math.cos(math.radians(axis_deg + 180))
            red_center_lon = center_lon + (separation_deg / 2.0) * math.sin(math.radians(axis_deg + 180))
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
        self.kappa = k
    
    def set_phase(self, phase_id):
        self.phase = phase_id

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
            
            # Find nearest enemy
            nearest = None
            min_dist_km = float('inf')
            for e in self.core.entities.values():
                if e.team == "red":
                    d = geodetic_distance_km(agent.lat, agent.lon, e.lat, e.lon)
                    if d < min_dist_km:
                        min_dist_km = d
                        nearest = e

            # === PHASE 1+ REWARDS (Flight & Approach) ===
            if self.phase >= 1 and nearest:
                # Approach reward
                if self.prev_dist is None:
                    self.prev_dist = min_dist_km
                
                approach_delta = self.prev_dist - min_dist_km
                approach_reward = approach_delta * 0.01
                reward += np.clip(approach_reward, -0.01, 0.01)
                
                self.prev_dist = min_dist_km
                
                # Stability bonus
                roll_penalty = np.clip(abs(agent.roll) * 0.005, 0, 0.01)
                pitch_penalty = np.clip(abs(agent.pitch) * 0.005, 0, 0.01)
                stability_bonus = 0.003 - roll_penalty - pitch_penalty
                reward += max(0, stability_bonus)
            
            # === PHASE 2+ REWARDS (Positioning) ===
            if self.phase >= 2 and nearest:
                bearing = geodetic_bearing_deg(agent.lat, agent.lon, nearest.lat, nearest.lon)
                
                # ATA
                ata = abs((bearing - agent.heading + 180) % 360 - 180)
                mu = np.radians(ata)
                
                # AA
                bearing_to_me = (bearing + 180) % 360
                aa = abs((bearing_to_me - nearest.heading + 180) % 360 - 180)
                lam = np.radians(aa)
                
                # Aiming reward
                r_aim = 0.5 * (1.0 - (mu / np.pi)) ** 2
                reward += r_aim * 0.01
                
                # Geometry reward
                pos_potential = self._potential(lam/np.pi, 0.5, 18.0)
                r_geo = (1.0 - mu/np.pi) * pos_potential
                reward += r_geo * 0.01
            
            # === PHASE 3+ REWARDS (Combat) ===
            if self.phase >= 3:
                # Lock reward
                if nearest and min_dist_km < self.cfg.MISSILE_RANGE_KM:
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
        lat_n, lon_n = self.map_limits.relative_position(e.lat, e.lon)
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
            
            bearing_to_target = geodetic_bearing_deg(ego.lat, ego.lon, e.lat, e.lon)
            ata_deg = abs((bearing_to_target - ego.heading + 180) % 360 - 180)
            ata_norm = ata_deg / 180.0
            
            bearing_to_ego = geodetic_bearing_deg(e.lat, e.lon, ego.lat, ego.lon)
            aa_deg = abs((bearing_to_ego - e.heading + 180) % 360 - 180)
            aa_norm = aa_deg / 180.0
            
            ego_radial = ego.speed * math.cos(math.radians(ata_deg))
            tgt_radial = e.speed * math.cos(math.radians(aa_deg))
            closure_rate_knots = ego_radial + tgt_radial
            closure_norm = np.clip(closure_rate_knots / 2000.0, -1.0, 1.0)

        return [
            lat_n, lon_n,
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
