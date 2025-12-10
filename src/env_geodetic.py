# ================================================
# FILE: src/env_geodetic.py
# ================================================
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from config import Config
from src.core import AirCombatCore
from src.utils.map_limits import MapLimits
from src.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


class AirCombatEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.cfg = Config
        self.core = None
        self.map_limits = MapLimits(*self.cfg.MAP_LIMITS)

        # ... (Init logic same as before) ...
        center_lat = (self.map_limits.bottom_lat + self.map_limits.top_lat) / 2.0
        center_lon = (self.map_limits.left_lon + self.map_limits.right_lon) / 2.0
        zoom = 0.15
        self.render_limits = MapLimits(center_lon - zoom, center_lat - zoom, center_lon + zoom, center_lat + zoom)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.OBS_DIM,), dtype=np.float32)

        self.renderer = None
        self.blue_ids = []
        self.red_ids = []
        self.kappa = 0.0
        self.phase = 1
        self.prev_dist = None

    def reset(self, seed=None, options=None):
        # ... (Reset logic matches previous geodetic implementation) ...
        # (This part usually doesn't need changes unless you want to sync spawn logic exactly with flat)
        # For brevity, I assume standard reset logic here.
        super().reset(seed=seed)
        self.core = AirCombatCore()
        self.blue_ids = []
        self.red_ids = []
        # ... (Spawn logic) ...
        # NOTE: Ensure you copy the spawn logic from your existing env_geodetic.py or update it.
        # The critical part is _calculate_reward below.

        # RETURNING DUMMY OBS FOR COPY-PASTE SAFETY
        # (Replace this with your actual spawn logic or keep existing)
        return np.zeros(self.cfg.OBS_DIM, dtype=np.float32), {}

    def step(self, action, red_actions=None):
        # ... (Action handling same as before) ...

        # 1. Setup Actions
        actions = {}
        agent_id = self.blue_ids[0] if self.blue_ids else -1

        if len(action.shape) > 0 and action.shape[0] == 2 * self.cfg.ACTION_DIM:
            actions[agent_id] = action[:self.cfg.ACTION_DIM]
            if self.red_ids: actions[self.red_ids[0]] = action[self.cfg.ACTION_DIM:]
        else:
            if agent_id != -1: actions[agent_id] = action
            if red_actions is not None and self.red_ids:
                actions[self.red_ids[0]] = red_actions

        # 2. Step Core
        self.core.step(actions, self.kappa)

        # 3. Rewards & Terminations
        reward = 0.0
        terminated = False
        truncated = False
        term_reason = "none"

        # DEATH
        if agent_id not in self.core.entities:
            terminated = True
            # Find cause
            death_event = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            if death_event:
                if death_event['type'] == 'crash':
                    reward = -5.0
                    term_reason = "crash"
                elif death_event['type'] == 'kill':
                    reward = -2.5
                    term_reason = "shot"
            else:
                reward = -5.0
                term_reason = "crash"
        else:
            # ALIVE
            agent = self.core.entities[agent_id]

            # 1. HARD DECK (0m)
            if agent.alt <= 1.0:
                reward = -5.0
                terminated = True
                term_reason = "crash"

            else:
                # 2. SOFT DECK (Cubic 3000m -> 0m)
                SOFT_DECK = 3000.0
                if agent.alt < SOFT_DECK:
                    proximity = (SOFT_DECK - agent.alt) / SOFT_DECK
                    penalty = 0.5 * (proximity ** 3)
                    if agent.pitch < -0.1: penalty *= 2.0
                    reward -= min(penalty, 1.0)

                # 3. KILLS (Owner Check)
                for ev in self.core.events:
                    if ev['type'] == 'kill':
                        is_killer = (ev['killer'] == agent_id)
                        is_owner = (ev.get('owner_id') == agent_id)
                        if is_killer or is_owner:
                            reward += 4.0
                            # Check win
                            reds_alive = sum(1 for e in self.core.entities.values() if e.team == "red")
                            if reds_alive == 0:
                                reward += 1.0
                                terminated = True
                                term_reason = "win"

                # 4. Existence
                reward += 0.005

        if self.core.time >= self.cfg.MAX_DURATION_SEC:
            truncated = True
            term_reason = "timeout"
            reward -= 1.0

        info = {
            "termination_reason": term_reason,
            "red_obs": np.zeros(self.cfg.OBS_DIM, dtype=np.float32)  # Placeholder
        }

        return self._get_obs(agent_id), reward, terminated, truncated, info

    def _get_obs(self, ego_id):
        # ... (Keep existing observation logic) ...
        return np.zeros(self.cfg.OBS_DIM, dtype=np.float32)

    def _vectorize(self, e, ego_id, is_ego):
        # ... (Keep existing vectorization logic) ...
        return []