# ================================================
# FILE: src/bot.py
# ================================================
import numpy as np
import math
from config import Config


class HardcodedAce:
    """
    Scripted Expert Agent for Air Combat.
    Updated for Relative/Egocentric Observation Space (29-dim).
    """

    def __init__(self):
        self.cfg = Config

    def get_action(self, obs):
        # obs shape: (OBS_DIM,) -> Flattened list of entities
        feat_dim = self.cfg.FEAT_DIM

        # 1. Parse Ego (First Entity)
        # New Index Mapping:
        # 7: Speed, 8: Alt, 16: Ammo
        ego_vec = obs[0:feat_dim]

        # Check if alive (Team index 17 should be non-zero)
        if ego_vec[17] == 0:
            return np.array([0.0, 0.0, 0.5, 0.0, 0.0])

        ego_alt_norm = ego_vec[8]
        ego_ammo = ego_vec[16]

        # 2. Scan for Threats
        enemies = []
        missiles = []
        num_entities = self.cfg.MAX_ENTITIES

        for i in range(1, num_entities):
            start = i * feat_dim
            end = start + feat_dim
            vec = obs[start:end]

            # Check if valid entity (Team != 0)
            if vec[17] == 0: continue

            # Feature Indices (Relative Obs):
            # 0: Range (Norm)
            # 1: Azimuth Cos, 2: Azimuth Sin
            # 3: Elevation Sin
            # 4: Aspect Cos
            # 6: Closure
            # 17: Team
            # 18: Type (1=Missile)
            # 20: RWR, 21: MAWS

            is_missile = (vec[18] > 0.5)
            is_enemy = (vec[17] != ego_vec[17])

            # Calculate Azimuth (ATA) in degrees
            # Azimuth is horizontal angle off nose
            # vector (Cos, Sin). Atan2(Sin, Cos)
            az_rad = math.atan2(vec[2], vec[1])
            az_deg = math.degrees(az_rad)

            ent = {
                'type': vec[18],
                'range': vec[0],  # Normalized
                'azimuth': az_deg,
                'elevation': vec[3],  # Sin of elevation
                'rwr': vec[20],
                'maws': vec[21],
                'closure': vec[6]
            }

            if is_missile:
                missiles.append(ent)
            elif is_enemy:
                enemies.append(ent)

        # === 3. TACTICAL LOGIC ===

        # A. EVADE MISSILES (MAWS Trigger)
        # New obs has explicit MAWS flag at index 21
        for m in missiles:
            if m['maws'] > 0.5:
                # Emergency Notch/Break
                # Pull 9G, Roll 90, Flare
                return np.array([1.0, 1.0, 1.0, 0.0, 1.0])

        # B. ENGAGE ENEMIES
        if enemies:
            # Sort by proximity/angle (Heuristic)
            # Prefer targets in front (small azimuth) and close
            target = min(enemies, key=lambda e: abs(e['azimuth']) + e['range'] * 100)

            ata = target['azimuth']

            # Fire Logic
            fire = 0.0
            # 15 degrees cone, within range (approx < 0.3 norm range)
            if abs(ata) < 15.0 and target['range'] < 0.5 and ego_ammo > 0:
                if np.random.rand() < 0.15:
                    fire = 1.0

            # Maneuver (P-Controller on Azimuth)
            # We want Azimuth -> 0
            roll_cmd = np.clip(ata / 45.0, -1.0, 1.0)

            # G-Pull (P-Controller on Azimuth Error)
            # If target is far to side, pull hard. If in front, pull less.
            # Also factor in elevation: If target is above (elevation > 0), pull up.
            g_cmd = np.clip(abs(ata) / 30.0, 0.0, 1.0)

            # Elevation correction
            if target['elevation'] > 0.1:  # Target high
                g_cmd += 0.3
            elif target['elevation'] < -0.1:  # Target low
                g_cmd -= 0.1  # Unload Gs to dive

            g_cmd = np.clip(g_cmd, -0.2, 1.0)

            return np.array([roll_cmd, g_cmd, 1.0, fire, 0.0])

        # C. PATROL / RECOVER
        # Level flight
        current_roll = math.atan2(ego_vec[14], ego_vec[13])  # Roll Sin/Cos
        roll_cmd = np.clip(-current_roll * 2.0, -1.0, 1.0)

        # Altitude Hold (10k ft target)
        target_alt = 0.33  # approx 5000m / 15000
        alt_err = target_alt - ego_alt_norm
        g_cmd = np.clip(alt_err * 5.0, -0.2, 0.5)

        return np.array([roll_cmd, g_cmd, 0.8, 0.0, 0.0])