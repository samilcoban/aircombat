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

        # ---------------------------------------------------------
        # 1. PARSE EGO STATE (Unified Node: 16D)
        # ---------------------------------------------------------
        ego_vec = obs[0:self.cfg.NODE_DIM]
        # Indices: [Exist, Team, Type, X, Y, Alt, CosH, SinH, SinP, SinR, Spd, G, Fuel, Ammo, Chaff, CM]

        if ego_vec[0] < 0.5:  # Dead
            return np.zeros(5, dtype=np.float32)

        # Recover State
        # Speed is normalized by 1000 in env
        ego_speed_norm = ego_vec[10]
        ego_speed_kts = ego_speed_norm * 1000.0

        alt_norm = ego_vec[5]  # 1.0 = 15000m
        alt_m = alt_norm * 15000.0

        # Angles (approximate from Sin, assuming mostly upright for simple control)
        current_roll = math.asin(np.clip(ego_vec[9], -1, 1))

        ammo = ego_vec[13]

        # ---------------------------------------------------------
        # 2. PARSE TARGETS (Unified Edge: 12D)
        # ---------------------------------------------------------
        track_data = obs[self.cfg.NODE_DIM:]
        num_tracks = len(track_data) // self.cfg.EDGE_DIM

        target = None
        closest_dist = float('inf')
        missile_threat = False

        # For Collision Avoidance
        collision_risk = False

        for i in range(num_tracks):
            start = i * self.cfg.EDGE_DIM
            vec = track_data[start: start + self.cfg.EDGE_DIM]

            # [Dist, LX, LY, LZ, ATA, AA, Align, Close, TgtSpd, TgtType, TeamRel, Vis]
            dist_norm = vec[0]
            if dist_norm < 1e-5: continue

            is_missile = (vec[9] < -0.5)
            team_rel = vec[10] # 1.0 = Friend, -1.0 = Enemy

            # A. THREAT LOGIC (Enemy Missile)
            if is_missile and team_rel < -0.5 and vec[7] > 0.1:
                if dist_norm < 0.15:  # ~9km warning
                    missile_threat = True

            # B. TARGET LOGIC (Enemy Plane)
            if team_rel < -0.5 and not is_missile:
                if dist_norm < closest_dist:
                    closest_dist = dist_norm
                    target = vec

            # C. COLLISION AVOIDANCE (Friendly Wingman)
            # 500m / 60000m = 0.0083
            # If a friend is within ~400m (0.0067), we panic.
            if team_rel > 0.5 and dist_norm < 0.0067:
                collision_risk = True

        # ---------------------------------------------------------
        # 3. TACTICAL LOGIC
        # ---------------------------------------------------------
        desired_roll = 0.0
        desired_g = 0.0  # 0.0 maps to roughly 1G (Level) in core physics
        throttle = 1.0
        fire = 0.0
        cm = 0.0

        # PRIORITY 1: TERRAIN & COLLISION SURVIVAL
        if alt_m < 4100.0 or collision_risk:
            # Emergency Pull Up / Break Away
            # Level wings to maximize vertical lift vector
            desired_roll = 0.0
            desired_g = 1.0  # Pull Max G (Action 1.0 -> 9G)

        # PRIORITY 2: MISSILE DEFENSE
        elif missile_threat:
            # Defensive Break (Beam the missile)
            desired_roll = 1.5  # Bank ~85 degrees
            desired_g = 1.0  # Pull Hard
            cm = 1.0  # Pop Chaff/Flares

        # PRIORITY 3: ENGAGEMENT
        elif target is not None:
            # Unpack Target Geometry from Local Coordinates
            lx, ly, lz = target[1], target[2], target[3]

            # Azimuth to target (in local frame)
            # Positive means target is Right
            az_rad = math.atan2(ly, lx)

            # Elevation to target
            # Positive means target is Above nose
            el_rad = math.atan2(lz, math.sqrt(lx * lx + ly * ly))

            # 1. ROLL TO TARGET
            # Bank towards the target to place lift vector on them
            desired_roll = np.clip(az_rad * 3.0, -1.5, 1.5)

            # 2. G-PULL (TURN)
            # Standard G demand based on angle error
            g_demand = abs(az_rad) * 2.0 + max(0, el_rad) * 3.0

            # 3. ENERGY MANAGEMENT
            if ego_speed_kts < 250.0:
                # STALL RECOVERY / UNLOAD
                g_demand = -0.2  # Unload to ~0.5G
                desired_roll = 0.0  # Wings level helps acceleration
            elif ego_speed_kts < 450.0:
                # SUSTAINED TURN / CORNER SPEED
                g_demand = min(g_demand, 0.4)

            # Gravity Compensation
            if abs(current_roll) > 0.5 and ego_speed_kts > 250.0:
                g_demand += 0.3

            desired_g = np.clip(g_demand, -0.2, 1.0)

            # 4. WEAPONS
            # Normalization factor is 60000.0
            # Guns Range: 1.5km -> 0.025
            is_aligned = abs(az_rad) < 0.15

            if is_aligned:
                if closest_dist < 0.025:  # Inside 1.5km
                    if np.random.rand() < 0.5:
                        fire = 1.0
                elif 0.02 <= closest_dist < 0.65 and ammo > 0:
                    if np.random.rand() < 0.05:
                        fire = 1.0

        # ---------------------------------------------------------
        # 4. LOW LEVEL CONTROLLER (PID)
        # ---------------------------------------------------------

        # Roll Rate Command (Proportional Controller)
        # Error = Desired Bank - Current Bank
        roll_error = desired_roll - current_roll

        # Action[0] is Roll Rate (-1 to 1 corresponds to -90 to 90 deg/s)
        # High gain (5.0) for snappy response
        roll_cmd = np.clip(roll_error * 5.0, -1.0, 1.0)

        # G Command (Direct map)
        g_cmd = np.clip(desired_g, -0.5, 1.0)

        return np.array([roll_cmd, g_cmd, throttle, fire, cm], dtype=np.float32)