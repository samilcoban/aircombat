# ================================================
# FILE: src/bot.py
# ================================================
import numpy as np
import math
from config import Config


class HardcodedAce:
    """
    Scripted Expert Agent for Air Combat.
    UPDATED: Supports Dual Projection Observation Space (Ego + Tracks).
    """

    def __init__(self):
        self.cfg = Config

    def get_action(self, obs):
        # obs shape: (OBS_DIM,) -> Flat vector

        # 1. Parse Ego (Indices 0 to FEAT_DIM_EGO)
        # ----------------------------------------
        ego_dim = self.cfg.FEAT_DIM_EGO
        ego_vec = obs[0:ego_dim]

        # Feature Mapping (based on env_flat.py _vectorize_ego):
        # 0: Exists, 1: Alt, 2: Speed, 3: Fuel,
        # 4: CosH, 5: SinH, 6: Pitch, 7: Roll
        # 8: Ammo, 9: Flares, 10: CM, 11: Team

        # Check if alive (Existence flag at 0)
        if ego_vec[0] < 0.5:
            return np.array([0.0, 0.0, 0.5, 0.0, 0.0])

        ego_alt_norm = ego_vec[1]
        ego_ammo = ego_vec[8]

        # 2. Parse Tracks (The rest of the vector)
        # ----------------------------------------
        edge_dim = self.cfg.FEAT_DIM_EDGE
        track_data = obs[ego_dim:]  # Slice off ego

        # Reshape to list of tracks
        # Note: In numpy we can't just .view(), we iterate based on stride
        num_tracks = (len(track_data)) // edge_dim

        missiles = []
        enemies = []

        for i in range(num_tracks):
            start = i * edge_dim
            end = start + edge_dim
            vec = track_data[start:end]

            # Feature Mapping (based on env_flat.py _vectorize_track):
            # 0: Range, 1: AzCos, 2: AzSin, 3: ElSin
            # 4: Closure, 5: Speed, 6: Type (1=Missile), 7: Faction

            # Check if this track is padding (Range is 0)
            if vec[0] < 1e-5: continue

            # Extract Data
            is_missile = (vec[6] > 0.5)
            is_enemy = (vec[7] < 0.0)  # -1.0 is enemy, 1.0 is ally

            # Calculate Azimuth degrees
            az_rad = math.atan2(vec[2], vec[1])
            az_deg = math.degrees(az_rad)

            ent = {
                'range': vec[0],  # Normalized
                'azimuth': az_deg,
                'elevation': vec[3],
                'closure': vec[4]
            }

            if is_missile and is_enemy:
                missiles.append(ent)
            elif is_enemy and not is_missile:
                enemies.append(ent)

        # === 3. TACTICAL LOGIC (Same as before) ===

        # A. EVADE MISSILES
        # Simple logic: If missile is close and closing, panic
        for m in missiles:
            if m['range'] < 0.1 and m['closure'] > 0:  # Very close
                # Emergency Notch/Break
                return np.array([1.0, 1.0, 1.0, 0.0, 1.0])

        # B. ENGAGE ENEMIES
        if enemies:
            # Sort by proximity
            target = min(enemies, key=lambda e: abs(e['azimuth']) + e['range'] * 100)
            ata = target['azimuth']

            # Fire Logic
            fire = 0.0
            if abs(ata) < 15.0 and target['range'] < 0.5 and ego_ammo > 0:
                if np.random.rand() < 0.15:
                    fire = 1.0

            # Maneuver (P-Controller on Azimuth)
            roll_cmd = np.clip(ata / 45.0, -1.0, 1.0)

            # G-Pull (P-Controller on Azimuth Error)
            g_cmd = np.clip(abs(ata) / 30.0, 0.0, 1.0)

            # Elevation correction
            if target['elevation'] > 0.1:
                g_cmd += 0.3
            elif target['elevation'] < -0.1:
                g_cmd -= 0.1

            g_cmd = np.clip(g_cmd, -0.2, 1.0)

            return np.array([roll_cmd, g_cmd, 1.0, fire, 0.0])

        # C. PATROL / RECOVER
        # Level flight
        current_roll = ego_vec[7] * 3.14  # Denormalize
        roll_cmd = np.clip(-current_roll * 2.0, -1.0, 1.0)

        # Altitude Hold (10k ft target)
        target_alt = 0.33
        alt_err = target_alt - ego_alt_norm
        g_cmd = np.clip(alt_err * 5.0, -0.2, 0.5)

        return np.array([roll_cmd, g_cmd, 0.8, 0.0, 0.0])