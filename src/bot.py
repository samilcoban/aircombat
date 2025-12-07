# ================================================
# FILE: src/bot.py
# ================================================
import numpy as np
import math
from config import Config


class HardcodedAce:
    """
    Scripted Expert Agent for Air Combat.
    UPDATED: Uses Unified Node/Edge structure.
    """

    def __init__(self):
        self.cfg = Config

    def get_action(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        # ---------------------------------------------------------
        # 1. PARSE EGO STATE (Unified Node: NODE_DIM = 16)
        # ---------------------------------------------------------
        # Indices: [Exist, Team, Type, X, Y, Alt, CosH, SinH, SinP, SinR, Spd, G, Fuel, Ammo, Chaff, CM]
        ego_vec = obs[0:self.cfg.NODE_DIM]

        if ego_vec[0] < 0.5:  # Existence Check
            return np.array([0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)

        ego_alt_norm = ego_vec[5]
        ego_ammo = ego_vec[13]

        # Recover Angles
        # We stored Sin(Roll) at index 9.
        # This loses the sign of Cos(Roll), meaning we can't distinguish upright vs inverted perfectly
        # without Cos. However, for a simple bot, arcsin is "okay" for small angles.
        current_roll_rad = math.asin(np.clip(ego_vec[9], -1, 1))

        # ---------------------------------------------------------
        # 2. PARSE TRACKS (Unified Edge: EDGE_DIM = 12)
        # ---------------------------------------------------------
        track_data = obs[self.cfg.NODE_DIM:]
        num_tracks = (len(track_data)) // self.cfg.EDGE_DIM

        missiles = []
        enemies = []

        for i in range(num_tracks):
            start = i * self.cfg.EDGE_DIM
            end = start + self.cfg.EDGE_DIM
            vec = track_data[start:end]

            # Indices: [Dist, LX, LY, LZ, ATA, AA, Align, Close, TgtSpd, TgtType, TeamRel, Vis]

            if vec[0] < 1e-5: continue  # Padding/Empty

            is_missile = (vec[9] < -0.5)
            is_enemy = (vec[10] < -0.5)

            # Recover Azimuth/Elevation from Local Coords (Indices 1,2,3)
            # Local X=Fwd, Y=Right, Z=Up
            lx, ly, lz = vec[1], vec[2], vec[3]
            dist_flat = math.hypot(lx, ly)

            # Note: The raw values in vec are normalized. We care about the RATIO for atan2, so scaling cancels out.
            az_rad = math.atan2(ly, lx) if dist_flat > 1e-6 else 0.0
            az_deg = math.degrees(az_rad)

            # For Elevation, we need to be careful with scaling if we used it.
            # Local Z was divided by 10000, X/Y by 60000.
            # Real Z = lz * 10000. Real Dist = dist * 60000.
            real_z = lz * 10000.0
            real_dist = vec[0] * 60000.0
            el_sin = np.clip(real_z / (real_dist + 1e-5), -1, 1)

            ent = {
                'range_norm': vec[0],
                'azimuth_deg': az_deg,
                'elevation_sin': el_sin,
                'closure': vec[7]
            }

            if is_missile and is_enemy:
                missiles.append(ent)
            elif is_enemy and not is_missile:
                enemies.append(ent)

        # ---------------------------------------------------------
        # 3. TACTICAL LOGIC (Logic remains mostly same, just inputs changed)
        # ---------------------------------------------------------

        # A. Evade Missiles
        for m in missiles:
            if m['range_norm'] < 0.1 and m['closure'] > 0:
                return np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)

        # B. Engage Enemies
        if enemies:
            target = min(enemies, key=lambda e: abs(e['azimuth_deg']) + e['range_norm'] * 100)
            ata = target['azimuth_deg']

            fire = 0.0
            if abs(ata) < 15.0 and target['range_norm'] < 0.5 and ego_ammo > 0:
                if np.random.rand() < 0.15: fire = 1.0

            roll_cmd = np.clip(ata / 45.0, -1.0, 1.0)
            g_cmd = np.clip(abs(ata) / 30.0, 0.0, 1.0)

            if target['elevation_sin'] > 0.1:
                g_cmd += 0.3
            elif target['elevation_sin'] < -0.1:
                g_cmd -= 0.2

            return np.array([roll_cmd, np.clip(g_cmd, -0.2, 1.0), 1.0, fire, 0.0], dtype=np.float32)

        # C. Patrol
        roll_cmd = np.clip(-current_roll_rad * 2.0, -1.0, 1.0)
        target_alt = 0.33
        alt_err = target_alt - ego_alt_norm
        g_cmd = np.clip(alt_err * 5.0, -0.2, 0.5)

        return np.array([roll_cmd, g_cmd, 0.8, 0.0, 0.0], dtype=np.float32)