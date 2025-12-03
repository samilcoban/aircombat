# ================================================
# FILE: src/bot.py
# ================================================
import numpy as np
import math
from config import Config


class HardcodedAce:
    """
    Scripted Expert Agent for Air Combat.
    Updated to use 3D geometry provided by the environment's observation vector.
    """

    def __init__(self):
        self.cfg = Config

    def get_action(self, obs):
        # obs is a numpy array of shape (OBS_DIM,)

        # === 1. PARSE OBSERVATION ===
        # We need to extract the Ego part (first entity) and Enemy parts
        # Feature Dim = 21 + TeamSize.
        feat_dim = self.cfg.FEAT_DIM

        # Ego is the first block
        # Intuition: The first 'feat_dim' elements correspond to the agent's own state.
        ego_vec = obs[0:feat_dim]

        # Check if alive
        # Intuition: If the ego vector is all zeros, the agent is dead. Return no-op.
        if np.all(ego_vec == 0):
            return np.array([0.0, 0.0, 0.5, 0.0, 0.0])

        # Parse Ego State
        # Indices: 14=Alt, 16=Ammo
        # Intuition: Extract normalized altitude and ammunition count.
        ego_alt_norm = ego_vec[14]
        ego_ammo = ego_vec[16]

        # Scan for Enemies and Missiles
        enemies = []
        missiles = []

        num_entities = self.cfg.MAX_ENTITIES
        # Intuition: Iterate through all other potential entities in the observation.
        for i in range(1, num_entities):  # Skip 0 (Ego)
            start = i * feat_dim
            end = start + feat_dim
            vec = obs[start:end]

            # Intuition: Skip empty slots.
            if np.all(vec == 0): continue

            # Parse Key Features directly from Observation
            # 17: ATA (Normalized 3D)
            # 18: AA (Normalized 3D)
            # 19: Closure (Normalized 3D)
            # 12: RWR
            # 13: MAWS
            # 6: Type (1.0=Missile)
            # 5: Team

            ent = {
                'type': vec[6],
                'team': vec[5],
                'rwr': vec[12],
                'maws': vec[13],
                'ata': vec[17] * 180.0,  # Denormalize to degrees
                'aa': vec[18] * 180.0,  # Denormalize
                'closure': vec[19],  # Normalized
                'dist_proxy': np.linalg.norm(vec[0:2]),  # Rough XY dist for sorting
                'vec': vec
            }

            # Intuition: Classify entity as missile or enemy aircraft.
            if ent['type'] > 0.5:
                missiles.append(ent)
            else:
                enemies.append(ent)

        # === 2. TACTICAL LOGIC ===

        # A. EVADE MISSILES (Notch)
        # 3D Note: Ideally we should dive/climb, but simple Notch is 2D turn.
        # We rely on MAWS signal.
        # Intuition: If a missile is detected (MAWS active), perform evasive maneuvers.
        for m in missiles:
            if m['maws'] > 0.5:
                # Emergency break turn
                # Intuition: Roll 90 deg, pull max Gs, and deploy flares.
                return np.array([1.0, 1.0, 1.0, 0.0, 1.0])  # Full Roll, Max G, Flare

        # B. ENGAGE ENEMIES
        if enemies:
            # Sort by "threat" (closest / best angle)
            # Heuristic: Smallest absolute ATA is best target
            # Intuition: Target the enemy closest to our nose (smallest Antenna Train Angle).
            target = min(enemies, key=lambda e: abs(e['ata']))

            ata = target['ata']

            # Fire Logic (3D Aware)
            fire = 0.0
            # Intuition: Fire if target is within a narrow cone (15 degrees) and we have ammo.
            if abs(ata) < 15.0 and ego_ammo > 0:
                # 3D Check: If we have good tone (small ATA), fire.
                # Probabilistic delay to simulate human reaction
                if np.random.rand() < 0.15:
                    fire = 1.0

            # Maneuver Logic
            # P-Controller for Roll based on ATA
            # We want ATA -> 0

            # Roll to align lift vector with target
            # If target is to the right (ATA > 0), roll right.
            # NOTE: This is a simplification. True Pure Pursuit requires aligning
            # velocity vector. Here we bank to pull nose.

            # Intuition: Proportional controller to roll towards the target.
            # Math: Command = Error * Kp. Here Kp = 1/45.0.
            roll_cmd = np.clip(ata / 45.0, -1.0, 1.0)

            # G-Pull Logic
            # If ATA is large, pull harder. If ATA is small, pull less (fine adjust).
            # Also maintain altitude if not fighting.
            # Intuition: Pull Gs proportional to the angle error to turn towards target.
            g_cmd = np.clip(abs(ata) / 30.0, 0.0, 1.0)

            # Energy management
            # Intuition: Full throttle during combat.
            throttle = 1.0

            return np.array([roll_cmd, g_cmd, throttle, fire, 0.0])

        # C. PATROL / RECOVER
        # Climb back to 10k ft (approx 0.2 norm alt)
        target_pitch = 0.0
        # Intuition: If too low, try to climb.
        if ego_alt_norm < 0.2: target_pitch = 0.2  # Pitch up

        # We can't set pitch directly, we must use G and Roll.
        # Level wings
        roll_cmd = -ego_vec[8]  # -Cos(Roll) is wrong...
        # Actually obs[8]=cos_r, obs[9]=sin_r. arctan2(sin, cos) -> angle
        # Intuition: Calculate current roll angle from sine/cosine components.
        current_roll = math.atan2(ego_vec[9], ego_vec[8])
        # Intuition: P-controller to level the wings (roll -> 0).
        roll_cmd = np.clip(-current_roll * 2.0, -1.0, 1.0)

        g_cmd = 0.0
        # Intuition: Gentle pull up if low altitude.
        if ego_alt_norm < 0.2: g_cmd = 0.5  # Gentle pull up

        return np.array([roll_cmd, g_cmd, 0.8, 0.0, 0.0])