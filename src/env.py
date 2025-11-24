import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from config import Config
from src.core import AirCombatCore
from aircombat_sim.utils.map_limits import MapLimits
from aircombat_sim.utils.geodesics import geodetic_distance_km, geodetic_bearing_deg


class AirCombatEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.cfg = Config
        self.core = None

        # --- Maps ---
        self.map_limits = MapLimits(*self.cfg.MAP_LIMITS)

        # Tactical Zoom for Rendering
        center_lon = (self.cfg.MAP_LIMITS[0] + self.cfg.MAP_LIMITS[2]) / 2.0
        center_lat = (self.cfg.MAP_LIMITS[1] + self.cfg.MAP_LIMITS[3]) / 2.0
        zoom = 0.75
        self.render_limits = MapLimits(
            center_lon - zoom, center_lat - zoom,
            center_lon + zoom, center_lat + zoom
        )

        # --- Spaces ---
        # Actions: [Roll, G-Pull, Throttle, Fire, Countermeasures]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.ACTION_DIM,), dtype=np.float32
        )

        # Observations: Flattened Entity List
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.OBS_DIM,), dtype=np.float32
        )

        self.renderer = None
        self.blue_ids = []
        self.red_ids = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.core = AirCombatCore()

        self.blue_ids = []
        self.red_ids = []

        center_lat = 0.5
        center_lon = 0.5

        # Spawn Blue Agents (South)
        for i in range(self.cfg.N_AGENTS):
            lat_pct = (center_lat - 0.02) + rng.uniform(-0.01, 0.01)  # Reduced from 0.05
            lon_pct = center_lon + ((i - self.cfg.N_AGENTS / 2) * 0.02)
            lat, lon = self.map_limits.absolute_position(lat_pct, lon_pct)
            # Spawn with heading 0 (North)
            uid = self.core.spawn(lat, lon, 0, 600, "blue", "plane")
            self.blue_ids.append(uid)

        # Spawn Red Enemies (North)
        for i in range(self.cfg.N_ENEMIES):
            lat_pct = (center_lat + 0.02) + rng.uniform(-0.01, 0.01)  # Reduced from 0.05
            lon_pct = center_lon + ((i - self.cfg.N_ENEMIES / 2) * 0.02)
            lat, lon = self.map_limits.absolute_position(lat_pct, lon_pct)
            # Spawn with heading 180 (South)
            uid = self.core.spawn(lat, lon, 180, 600, "red", "plane")
            self.red_ids.append(uid)

        # Return observation for the first Blue agent
        info = {}
        if self.red_ids:
            info["red_obs"] = self._get_obs(self.red_ids[0])
        
        return self._get_obs(self.blue_ids[0]), info

    def step(self, action, red_actions=None):
        # 1. Prepare Actions
        actions = {}
        agent_id = self.blue_ids[0]

        # Check for Concatenated Actions (Blue + Red)
        # This allows AsyncVectorEnv to pass both without changing signature
        if len(action.shape) > 0 and action.shape[0] == 2 * self.cfg.ACTION_DIM:
            blue_action = action[:self.cfg.ACTION_DIM]
            red_action_in = action[self.cfg.ACTION_DIM:]
            
            if agent_id in self.core.entities:
                actions[agent_id] = blue_action
            
            if self.red_ids and self.red_ids[0] in self.core.entities:
                actions[self.red_ids[0]] = red_action_in
        else:
            # Standard Single Agent
            if agent_id in self.core.entities:
                actions[agent_id] = action
            
            # Inject Red Actions if provided explicitly (Legacy/Manual)
            if red_actions is not None:
                if self.red_ids and self.red_ids[0] in self.core.entities:
                    if isinstance(red_actions, (np.ndarray, list)):
                        actions[self.red_ids[0]] = red_actions
                    elif isinstance(red_actions, dict):
                        actions.update(red_actions)

        # 2. Step Physics Core
        self.core.step(actions)

        # 3. Calculate Rewards
        reward = 0.0
        terminated = False
        truncated = False

        if agent_id not in self.core.entities:
            reward = -50.0  # Massive Penalty for Dying (Cowardice/Incompetence)
            terminated = True
        else:
            # 1. Existential Penalty (Time Pressure)
            reward -= 0.005 

            agent = self.core.entities[agent_id]

            # --- Speed Maintenance Reward ---
            if agent.speed < 200.0:
                reward -= 0.1  # Penalize slow/stalling flight

            # --- Shaping Reward (Alignment & Distance) ---
            nearest = None
            min_dist = float('inf')
            for e in self.core.entities.values():
                if e.team == "red":
                    d = geodetic_distance_km(agent.lat, agent.lon, e.lat, e.lon)
                    if d < min_dist:
                        min_dist = d
                        nearest = e

            if nearest:
                # Pointing at enemy?
                bearing = geodetic_bearing_deg(agent.lat, agent.lon, nearest.lat, nearest.lon)
                angle = abs((bearing - agent.heading + 180) % 360 - 180)
                
                # WEAK alignment reward (just to help them find the enemy initially)
                # Only if within reasonable range (e.g. 2x missile range) to prevent infinite chasing
                if min_dist < self.cfg.MISSILE_RANGE_KM * 2:
                    alignment_bonus = (1.0 - (angle / 180.0)) * 0.01
                    reward += alignment_bonus

                # Engagement Bonus: Within weapons range AND good alignment
                if min_dist < self.cfg.MISSILE_RANGE_KM and angle < 45.0:
                    # STRONG Lock Reward (Requires Radar Lock)
                    is_locking, _ = self.core.get_sensor_state(agent_id, nearest.uid)
                    if is_locking:
                        reward += 0.1

            # --- Event Rewards (Kills Only) ---
            for ev in self.core.events:
                if ev['type'] == 'kill':
                    if ev['killer'] in self.blue_ids: 
                        reward += 100.0  # JACKPOT. Nothing else matters.
                    if ev['victim'] in self.blue_ids: 
                        reward -= 50.0  # Redundant with death penalty but good for event tracking
                # REMOVED MISSILE FIRE REWARD
                # REMOVED SURVIVAL REWARD

            # --- Win Condition ---
            reds_alive = sum(1 for e in self.core.entities.values() if e.team == "red")
            if reds_alive == 0:
                reward += 100.0 # Extra bonus for winning
                terminated = True

        # 4. Check Time Limit
        if self.core.time >= self.cfg.MAX_DURATION_SEC:
            truncated = True

        info = {}
        if self.red_ids and self.red_ids[0] in self.core.entities:
            info["red_obs"] = self._get_obs(self.red_ids[0])
        else:
            # Dead Red Agent placeholder
            info["red_obs"] = np.zeros(self.cfg.OBS_DIM, dtype=np.float32)

        return self._get_obs(agent_id), reward, terminated, truncated, info

    def _get_obs(self, ego_id):
        vecs = []

        # 1. Ego Vector (Always present)
        if ego_id in self.core.entities:
            vecs.append(self._vectorize(self.core.entities[ego_id], ego_id, True))
        else:
            # Dead agent placeholder
            vecs.append(np.zeros(self.cfg.FEAT_DIM, dtype=np.float32))

        # 2. Other Entities (Friends, Enemies, Missiles)
        for uid, ent in self.core.entities.items():
            if uid == ego_id: continue

            # --- Sensor Logic (Fog of War) ---
            visible = True
            rwr_active = False

            if ego_id in self.core.entities and ent.team != "blue":
                # Can I see them? (Radar + Doppler)
                visible, _ = self.core.get_sensor_state(ego_id, uid)

                # Can they see me? (RWR)
                # Check if they are locking me
                _, locking_me = self.core.get_sensor_state(uid, ego_id)
                if locking_me:
                    rwr_active = True

            if visible:
                v = self._vectorize(ent, ego_id, False)
                # If visible + locking, set RWR flag in the visible vector
                if rwr_active: v[12] = 1.0
                vecs.append(v)
            elif rwr_active:
                # Not visible but Locking -> Ghost Vector (RWR only)
                v = np.zeros(self.cfg.FEAT_DIM, dtype=np.float32)
                v[12] = 1.0  # Set RWR flag
                vecs.append(v)
            else:
                # Hidden
                vecs.append(np.zeros(self.cfg.FEAT_DIM, dtype=np.float32))

        # 3. Padding
        flat = []
        for v in vecs: flat.extend(v)

        # Truncate if too many
        if len(flat) > self.cfg.OBS_DIM:
            flat = flat[:self.cfg.OBS_DIM]

        # Pad if too few
        if len(flat) < self.cfg.OBS_DIM:
            flat.extend([0.0] * (self.cfg.OBS_DIM - len(flat)))

        return np.array(flat, dtype=np.float32)

    def _vectorize(self, e, ego_id, is_ego):
        lat_n, lon_n = self.map_limits.relative_position(e.lat, e.lon)
        hr = math.radians(e.heading)

        # Sensor Signals
        rwr_signal = 0.0
        maws_signal = 0.0

        # MAWS: If it's a missile targeting me
        if e.type == "missile" and e.target_id == ego_id:
            maws_signal = 1.0

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
            # --- Logistics (Phase 5) ---
            e.fuel,
            e.ammo / float(self.cfg.MAX_MISSILES)
        ]

    def render(self):
        from aircombat_sim.utils.scenario_plotter import ScenarioPlotter, PlotConfig, Airplane, Missile
        import matplotlib.pyplot as plt

        if self.renderer is None:
            p_cfg = PlotConfig()
            p_cfg.units_scale = 20.0
            self.renderer = ScenarioPlotter(self.render_limits, dpi=100, config=p_cfg)

        drawables = []
        for e in self.core.entities.values():
            c = (0, 0, 1, 1) if e.team == "blue" else (1, 0, 0, 1)

            if e.type == "missile":
                drawables.append(Missile(e.lat, e.lon, e.heading, fill_color=c, zorder=10))
            else:
                # Detailed Info Text
                txt = f"{e.uid}\nA:{int(e.alt)}\nF:{int(e.fuel * 100)}%"
                if e.cm_active: txt += "\nCM!"

                drawables.append(Airplane(e.lat, e.lon, e.heading, fill_color=c, info_text=txt, zorder=5))

        try:
            fname = f"temp_render_{self.blue_ids[0] if self.blue_ids else 0}.png"
            self.renderer.to_png(fname, drawables)
            return (plt.imread(fname) * 255).astype(np.uint8)
        except:
            return np.zeros((400, 400, 3), dtype=np.uint8)