# ================================================
# FILE: src/bot.py
# ================================================
import numpy as np
import math
from config import Config


class HardcodedAce:
    """
    Scripted Expert Agent for Air Combat.
    Implements Proportional Navigation, PID Control, and Energy Management.

    Strategies:
    1. Survival: Priority pull-up if near hard deck or breaking for missiles.
    2. Energy Management: Unloads Gs if speed drops too low to prevent stalling.
    3. Intercept: Banks towards target and pulls Gs to lead the turn.
    """

    def __init__(self):
        self.cfg = Config

    def get_action(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        # 1. PARSE EGO STATE (Unified Node: 20D)
        ego_vec = obs[0:self.cfg.NODE_DIM]
        if ego_vec[0] < 0.5: return np.zeros(5, dtype=np.float32)

        ego_speed_kts = ego_vec[10] * 1000.0
        alt_m = ego_vec[5] * 15000.0
        current_roll = math.asin(np.clip(ego_vec[9], -1, 1))

        # Derivatives (Normalized) for D-Term
        ego_d_roll = ego_vec[18]
        ego_d_speed = ego_vec[19]
        ammo = ego_vec[13]

        # 2. PARSE TARGETS
        track_data = obs[self.cfg.NODE_DIM:]
        num_tracks = len(track_data) // self.cfg.EDGE_DIM

        target = None
        closest_dist = float('inf')
        missile_threat = False
        collision_risk = False

        for i in range(num_tracks):
            start = i * self.cfg.EDGE_DIM
            vec = track_data[start: start + self.cfg.EDGE_DIM]
            dist_norm = vec[0]
            if dist_norm < 1e-5: continue

            is_missile = (vec[9] < -0.5)
            team_rel = vec[10]

            if is_missile and team_rel < -0.5 and dist_norm < 0.15 and vec[7] > 0.5:
                missile_threat = True

            if team_rel < -0.5 and not is_missile:
                if dist_norm < closest_dist:
                    closest_dist = dist_norm
                    target = vec

            if team_rel > 0.5 and dist_norm < 0.0067:
                collision_risk = True

        # 3. TACTICAL LOGIC
        desired_roll = 0.0
        desired_g = 0.0
        throttle = 1.0
        fire = 0.0
        cm = 0.0

        if alt_m < 3500.0 or collision_risk:
            desired_roll = 0.0
            desired_g = 1.0
        elif missile_threat:
            desired_roll = 1.5
            desired_g = 1.0
            cm = 1.0
        elif target is not None:
            lx, ly, lz = target[1], target[2], target[3]

            # Lead Pursuit
            tgt_turn_rate = target[12] * 2.0
            az_rad = math.atan2(ly, lx) + tgt_turn_rate
            el_rad = math.atan2(lz, math.sqrt(lx * lx + ly * ly))

            desired_roll = np.clip(az_rad * 3.5, -1.5, 1.5)
            g_demand = abs(az_rad) * 2.5 + max(0, el_rad) * 4.0

            # Smart Energy Management
            is_losing_energy = (ego_d_speed < -0.1)
            if ego_speed_kts < 250.0:
                g_demand = -0.2
                desired_roll = 0.0
            elif ego_speed_kts < 400.0 and is_losing_energy:
                g_demand = min(g_demand, 0.3)

            if abs(current_roll) > 0.5 and ego_speed_kts > 250.0:
                g_demand += 0.3
            desired_g = np.clip(g_demand, -0.2, 1.0)

            # Weapons
            if closest_dist < 0.6 and ammo > 0:
                if (abs(az_rad) < 0.2) or (target[7] > 0.5):  # Aligned or Closing Fast
                    if np.random.rand() < 0.1: fire = 1.0

        # 4. PD CONTROLLER
        roll_error = desired_roll - current_roll
        p_term = roll_error * 6.0
        d_term = ego_d_roll * 1.5  # Damping
        roll_cmd = np.clip(p_term - d_term, -1.0, 1.0)
        g_cmd = np.clip(desired_g, -0.5, 1.0)

        return np.array([roll_cmd, g_cmd, throttle, fire, cm], dtype=np.float32)