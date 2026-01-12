# ================================================
# FILE: src/env_geodetic.py
# ================================================
"""
Gymnasium environment for air combat using geodetic (curved-earth) physics.

This module implements a Gymnasium environment using WGS84 geodetic
coordinates (lat/lon) instead of flat Cartesian coordinates.
This is the alternative to env_flat.py for global-scale simulations.

Note: This is a simplified version of the flat environment and may
not have all features fully implemented. The flat version is the
primary implementation used for training.

Coordinate System:
- Position: Latitude/Longitude in degrees
- Heading: Degrees (0=North, 90=East)
- Altitude: Meters above WGS84 ellipsoid
"""
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
    Geodetic air combat Gymnasium environment.
    
    Uses latitude/longitude coordinates for position tracking.
    This version is less optimized than env_flat.py but more
    geographically accurate for large-area scenarios.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        """Initialize environment spaces and parameters."""
        super().__init__()
        self.cfg = Config
        self.core = None
        self.map_limits = MapLimits(*self.cfg.MAP_LIMITS)

        # Rendering viewport centered on map.
        center_lat = (self.map_limits.bottom_lat + self.map_limits.top_lat) / 2.0
        center_lon = (self.map_limits.left_lon + self.map_limits.right_lon) / 2.0
        zoom = 0.15
        self.render_limits = MapLimits(center_lon - zoom, center_lat - zoom, center_lon + zoom, center_lat + zoom)

        # Define action and observation spaces.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.OBS_DIM,), dtype=np.float32)

        self.renderer = None
        self.blue_ids = []
        self.red_ids = []
        self.kappa = 0.0
        self.phase = 1
        self.prev_dist = None

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        NOTE: This is a simplified implementation. The spawn logic
        should be adapted from the flat version or customized for
        geodetic scenarios.
        
        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)
        self.core = AirCombatCore()
        self.blue_ids = []
        self.red_ids = []
        
        # TODO: Implement full spawn logic matching env_flat.py
        # For now, returning dummy observation.
        return np.zeros(self.cfg.OBS_DIM, dtype=np.float32), {}

    def step(self, action, red_actions=None):
        """
        Step the environment forward.
        
        Args:
            action: Agent action array.
            red_actions: Optional red team actions.
            
        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        # 1. Setup Actions.
        actions = {}
        agent_id = self.blue_ids[0] if self.blue_ids else -1

        # Handle joint action space (if both teams controlled).
        if len(action.shape) > 0 and action.shape[0] == 2 * self.cfg.ACTION_DIM:
            actions[agent_id] = action[:self.cfg.ACTION_DIM]
            if self.red_ids: actions[self.red_ids[0]] = action[self.cfg.ACTION_DIM:]
        else:
            if agent_id != -1: actions[agent_id] = action
            if red_actions is not None and self.red_ids:
                actions[self.red_ids[0]] = red_actions

        # 2. Step Physics.
        self.core.step(actions, self.kappa)

        # 3. Calculate Rewards & Check Terminations.
        reward = 0.0
        terminated = False
        truncated = False
        term_reason = "none"

        # DEATH: Agent no longer in simulation.
        if agent_id not in self.core.entities:
            terminated = True
            # Determine cause of death.
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
            # ALIVE: Process rewards.
            agent = self.core.entities[agent_id]

            # Hard Deck: Ground collision.
            if agent.alt <= 1.0:
                reward = -5.0
                terminated = True
                term_reason = "crash"

            else:
                # Soft Deck: Altitude penalty zone (3000m -> 0m).
                SOFT_DECK = 3000.0
                if agent.alt < SOFT_DECK:
                    proximity = (SOFT_DECK - agent.alt) / SOFT_DECK
                    penalty = 0.5 * (proximity ** 3)  # Cubic penalty.
                    if agent.pitch < -0.1: penalty *= 2.0  # Diving makes it worse.
                    reward -= min(penalty, 1.0)

                # Kill rewards with owner attribution.
                for ev in self.core.events:
                    if ev['type'] == 'kill':
                        is_killer = (ev['killer'] == agent_id)
                        is_owner = (ev.get('owner_id') == agent_id)
                        if is_killer or is_owner:
                            reward += 4.0
                            # Check for win condition.
                            reds_alive = sum(1 for e in self.core.entities.values() if e.team == "red")
                            if reds_alive == 0:
                                reward += 1.0
                                terminated = True
                                term_reason = "win"

                # Existence reward.
                reward += 0.005

        # Timeout check.
        if self.core.time >= self.cfg.MAX_DURATION_SEC:
            truncated = True
            term_reason = "timeout"
            reward -= 1.0

        info = {
            "termination_reason": term_reason,
            "red_obs": np.zeros(self.cfg.OBS_DIM, dtype=np.float32)  # Placeholder.
        }

        return self._get_obs(agent_id), reward, terminated, truncated, info

    def _get_obs(self, ego_id):
        """
        Get observation for agent.
        
        NOTE: This is a placeholder. Full observation logic should
        match the structure in env_flat.py.
        
        Returns:
            Observation array.
        """
        # TODO: Implement full observation matching env_flat.py
        return np.zeros(self.cfg.OBS_DIM, dtype=np.float32)

    def _vectorize(self, e, ego_id, is_ego):
        """
        Convert entity to feature vector.
        
        NOTE: Placeholder for feature extraction.
        
        Returns:
            Feature list.
        """
        # TODO: Implement full vectorization matching env_flat.py
        return []