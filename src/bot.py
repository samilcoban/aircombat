# ================================================
# FILE: src/bot.py
# ================================================
import numpy as np
import math
from config import Config


class HardcodedAce:
    """
    Scripted Expert Agent for Air Combat.
    Acts as a baseline and robust opponent for Self-Play.

    Logic:
    1. Decodes the 'Dual Projection' observation (Ego + Tracks).
    2. Recovers geometric angles from Sin/Cos features.
    3. Uses P-Controllers to execute Pure Pursuit and Evasion.
    """

    def __init__(self):
        self.cfg = Config

    def get_action(self, obs):
        """
        Input: Flattened Observation Vector (OBS_DIM,)
        Output: Action Vector (5,) -> [Roll, G, Throttle, Fire, CM]
        """
        # Ensure input is numpy
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        # ---------------------------------------------------------
        # 1. PARSE EGO STATE (Indices 0 to FEAT_DIM_EGO)
        # ---------------------------------------------------------
        ego_dim = self.cfg.FEAT_DIM_EGO
        ego_vec = obs[0:ego_dim]

        # Feature Mapping (Matches src/env_flat.py -> _vectorize_ego):
        # 0: Exists (1.0)
        # 1: Alt (Normalized / 15000)
        # 2: Speed (Normalized / 1000)
        # 3: Fuel
        # 4: Cos(Heading)
        # 5: Sin(Heading)
        # 6: Pitch (Normalized / 1.57)
        # 7: Roll (Normalized / 3.14)
        # 8: Ammo (Fraction)
        # ...

        # Check if alive (Existence flag at index 0)
        if ego_vec[0] < 0.5:
            # Dead: Output safe zeros
            return np.array([0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)

        # Denormalize useful states
        ego_alt_norm = ego_vec[1]
        ego_ammo = ego_vec[8]

        # Roll is normalized by PI (3.14). We recover Radians for control.
        current_roll_rad = ego_vec[7] * math.pi

        # ---------------------------------------------------------
        # 2. PARSE TRACKS (The rest of the vector)
        # ---------------------------------------------------------
        edge_dim = self.cfg.FEAT_DIM_EDGE
        track_data = obs[ego_dim:]

        # Calculate number of tracks
        num_tracks = (len(track_data)) // edge_dim

        missiles = []
        enemies = []

        for i in range(num_tracks):
            start = i * edge_dim
            end = start + edge_dim
            vec = track_data[start:end]

            # Feature Mapping (Matches src/env_flat.py -> _vectorize_track):
            # 0: Range (Normalized / 60km)
            # 1: Cos(Azimuth)
            # 2: Sin(Azimuth)
            # 3: Sin(Elevation)
            # 4: Closure Rate
            # 5: Abs Speed
            # 6: Is Missile (1.0)
            # 7: Is Teammate (1.0) / Enemy (-1.0)
            # ...

            # Check if this is padding (Range is 0)
            if vec[0] < 1e-5: continue

            # Extract Data
            is_missile = (vec[6] > 0.5)
            is_enemy = (vec[7] < -0.5)

            # Recover Azimuth from Sin/Cos (Atan2 handles quadrants correctly)
            # vec[2] is Sin(Az) (Left/Right), vec[1] is Cos(Az) (Forward/Back)
            az_rad = math.atan2(vec[2], vec[1])
            az_deg = math.degrees(az_rad)

            ent = {
                'range_norm': vec[0],
                'azimuth_deg': az_deg,
                'elevation_sin': vec[3],
                'closure': vec[4]
            }

            if is_missile and is_enemy:
                missiles.append(ent)
            elif is_enemy and not is_missile:
                enemies.append(ent)

        # ---------------------------------------------------------
        # 3. TACTICAL LOGIC
        # ---------------------------------------------------------

        # --- A. EVADE MISSILES (Survival Priority) ---
        # If a missile is close (< 6km) and closing, panic.
        # Range norm 0.1 * 60km = 6km
        for m in missiles:
            if m['range_norm'] < 0.1 and m['closure'] > 0:
                # Notch Maneuver / Break Turn
                # Roll 90 degrees (1.0), Pull Max G (1.0), Drop Chaff (1.0)
                return np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)

        # --- B. ENGAGE ENEMIES ---
        if enemies:
            # Target Selection: Minimize "Angle + Distance" cost
            # Heuristic: 10% range is worth 10 degrees of angle
            target = min(enemies, key=lambda e: abs(e['azimuth_deg']) + e['range_norm'] * 100)

            ata = target['azimuth_deg']  # Degrees

            # Fire Logic (Trigger Discipline)
            fire = 0.0
            # Fire if: Angle < 15 deg AND Range < 30km (0.5) AND Ammo > 0
            if abs(ata) < 15.0 and target['range_norm'] < 0.5 and ego_ammo > 0:
                # Stochastic firing to prevent perfectly periodic shooting
                if np.random.rand() < 0.15:
                    fire = 1.0

            # === PID CONTROLLERS ===

            # Roll Controller: Bank to the target
            # Gain: 1.0 roll command per 45 degrees of error
            roll_cmd = np.clip(ata / 45.0, -1.0, 1.0)

            # Pitch/G Controller: Pull into the target
            # Only pull Gs if we have rolled enough to put the target "above" us (in body frame)
            # or if the target is already in front.
            # Simplified: Pull Gs proportional to azimuth error magnitude (Turn Rate)
            g_cmd = np.clip(abs(ata) / 30.0, 0.0, 1.0)

            # Elevation Correction
            # vec[3] is Sin(Elevation). Positive = Target is Up.
            # If target is Up (relative to nose), Pull Harder.
            # If target is Down, Push Nose Down (negative G) or Roll Inverted.
            if target['elevation_sin'] > 0.1:
                g_cmd += 0.3
            elif target['elevation_sin'] < -0.1:
                g_cmd -= 0.2  # Unload Gs to drop nose

            # Clamp Gs (-1 to 9G map usually, here simple clip)
            g_cmd = np.clip(g_cmd, -0.2, 1.0)

            return np.array([roll_cmd, g_cmd, 1.0, fire, 0.0], dtype=np.float32)

        # --- C. PATROL / RECOVER (No Enemies) ---
        # 1. Level Wings: P-Controller on Roll
        # If roll is positive (right), command negative (left)
        roll_cmd = np.clip(-current_roll_rad * 2.0, -1.0, 1.0)

        # 2. Altitude Hold: Target 33% of max altitude (~5000m)
        target_alt = 0.33
        alt_err = target_alt - ego_alt_norm

        # If low, pull up. If high, push down.
        g_cmd = np.clip(alt_err * 5.0, -0.2, 0.5)

        return np.array([roll_cmd, g_cmd, 0.8, 0.0, 0.0], dtype=np.float32)