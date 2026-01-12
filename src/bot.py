# ================================================
# FILE: src/bot.py
# ================================================
"""
Scripted expert agent for air combat.

This module implements a hardcoded "Ace" bot that serves as an expert
for behavior cloning. It uses proportional navigation, PID control,
and energy management to engage targets effectively.

The bot's decisions prioritize:
1. Survival (altitude, collision avoidance, missile defense)
2. Energy management (maintaining sufficient airspeed)
3. Target engagement (pursuit geometry, weapons employment)
"""
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
    
    This bot is used to generate expert trajectories for behavior cloning
    in the pretrain.py script.
    """

    def __init__(self):
        """Initialize bot with reference to config."""
        self.cfg = Config

    def get_action(self, obs):
        """
        Compute the expert action given an observation.
        
        Args:
            obs: Observation array containing ego state and track data.
                 Format: [ego_node (NODE_DIM), track_1 (EDGE_DIM), track_2, ...]
                 
        Returns:
            Action array [roll_cmd, g_cmd, throttle, fire, countermeasures].
            All values in [-1, 1] range.
        """
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        # ==================== 1. PARSE EGO STATE ====================
        # Extract ego aircraft state from unified node format (20D).
        ego_vec = obs[0:self.cfg.NODE_DIM]
        
        # Check existence flag - if dead, return zero action.
        if ego_vec[0] < 0.5: return np.zeros(5, dtype=np.float32)

        # Denormalize key state variables.
        ego_speed_kts = ego_vec[10] * 1000.0  # Speed in knots.
        alt_m = ego_vec[5] * 15000.0          # Altitude in meters.
        current_roll = math.asin(np.clip(ego_vec[9], -1, 1))  # Roll angle from sin(roll).

        # Derivatives (Normalized) for D-Term in PD controller.
        ego_d_roll = ego_vec[18]   # Roll rate.
        ego_d_speed = ego_vec[19]  # Speed rate (acceleration).
        ammo = ego_vec[13]         # Remaining missiles (normalized).

        # ==================== 2. PARSE TARGETS ====================
        # Extract track data (edges) for situational awareness.
        track_data = obs[self.cfg.NODE_DIM:]
        num_tracks = len(track_data) // self.cfg.EDGE_DIM

        target = None
        closest_dist = float('inf')
        missile_threat = False
        collision_risk = False

        # Scan all tracks to find threats and targets.
        for i in range(num_tracks):
            start = i * self.cfg.EDGE_DIM
            vec = track_data[start: start + self.cfg.EDGE_DIM]
            dist_norm = vec[0]  # Normalized distance.
            if dist_norm < 1e-5: continue  # Skip invalid tracks.

            is_missile = (vec[9] < -0.5)  # Type flag: -1 = missile.
            team_rel = vec[10]             # Team relation: -1 = enemy, +1 = friendly.

            # Check for incoming missile threat.
            # Conditions: enemy missile, close range, closing on us.
            if is_missile and team_rel < -0.5 and dist_norm < 0.15 and vec[7] > 0.5:
                missile_threat = True

            # Find closest enemy aircraft for targeting.
            if team_rel < -0.5 and not is_missile:
                if dist_norm < closest_dist:
                    closest_dist = dist_norm
                    target = vec

            # Check for collision risk with friendlies.
            if team_rel > 0.5 and dist_norm < 0.0067:
                collision_risk = True

        # ==================== 3. TACTICAL LOGIC ====================
        # Default control outputs.
        desired_roll = 0.0
        desired_g = 0.0
        throttle = 1.0  # Always full throttle.
        fire = 0.0
        cm = 0.0  # Countermeasures.

        # Priority 1: Survival - altitude or collision avoidance.
        if alt_m < 3500.0 or collision_risk:
            desired_roll = 0.0  # Wings level.
            desired_g = 1.0     # Pull up.
            
        # Priority 2: Missile defense - break and deploy countermeasures.
        elif missile_threat:
            desired_roll = 1.5  # Hard bank.
            desired_g = 1.0     # Pull hard.
            cm = 1.0            # Deploy chaff/flares.
            
        # Priority 3: Target engagement.
        elif target is not None:
            # Extract local coordinates of target (in body frame).
            lx, ly, lz = target[1], target[2], target[3]

            # Lead Pursuit - anticipate target movement.
            tgt_turn_rate = target[12] * 2.0  # Target's heading rate.
            az_rad = math.atan2(ly, lx) + tgt_turn_rate  # Azimuth with lead.
            el_rad = math.atan2(lz, math.sqrt(lx * lx + ly * ly))  # Elevation.

            # Compute desired roll to align with target azimuth.
            desired_roll = np.clip(az_rad * 3.5, -1.5, 1.5)
            
            # Compute G demand based on azimuth and elevation error.
            g_demand = abs(az_rad) * 2.5 + max(0, el_rad) * 4.0

            # Smart Energy Management - reduce Gs if losing energy.
            is_losing_energy = (ego_d_speed < -0.1)
            if ego_speed_kts < 250.0:
                # Critically low speed - unload to accelerate.
                g_demand = -0.2
                desired_roll = 0.0
            elif ego_speed_kts < 400.0 and is_losing_energy:
                # Low speed and slowing - limit Gs.
                g_demand = min(g_demand, 0.3)

            # Add extra G when banked to maintain altitude.
            if abs(current_roll) > 0.5 and ego_speed_kts > 250.0:
                g_demand += 0.3
            desired_g = np.clip(g_demand, -0.2, 1.0)

            # Weapons Employment - fire when aligned and in range.
            if closest_dist < 0.6 and ammo > 0:
                if (abs(az_rad) < 0.2) or (target[7] > 0.5):  # Aligned or Closing Fast.
                    if np.random.rand() < 0.1: fire = 1.0  # Stochastic fire.

        # ==================== 4. PD CONTROLLER ====================
        # Convert desired roll/G to actual commands using PD control.
        roll_error = desired_roll - current_roll
        p_term = roll_error * 6.0        # Proportional term.
        d_term = ego_d_roll * 1.5        # Derivative term (damping).
        roll_cmd = np.clip(p_term - d_term, -1.0, 1.0)
        g_cmd = np.clip(desired_g, -0.5, 1.0)

        return np.array([roll_cmd, g_cmd, throttle, fire, cm], dtype=np.float32)