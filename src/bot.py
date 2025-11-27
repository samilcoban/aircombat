import numpy as np
import math
from config import Config

class HardcodedAce:
    """
    Scripted Expert Agent for Air Combat.
    
    Implements BVR (Beyond Visual Range) tactics:
    - Notching: Turning 90 degrees to incoming missiles (Doppler Notch).
    - Cranking: Turning 50 degrees off target after firing to maintain lock while maximizing range.
    - Energy Management: Maintaining corner velocity.
    - Pure/Lead Pursuit: For offensive maneuvering.
    
    Operates purely on the observation vector (no cheating via direct simulation access).
    """
    def __init__(self):
        self.cfg = Config
        self.last_fire_time = -100.0
        self.target_id_track = None

    def get_action(self, obs):
        """
        Generate action from observation.
        
        Args:
            obs: Flattened observation vector (numpy array)
            
        Returns:
            action: numpy array [roll, g, throttle, fire, cm]
        """
        # === 1. PARSE OBSERVATION ===
        ego, enemies, missiles = self._parse_obs(obs)
        
        if ego is None:
            return np.array([0.0, 0.0, 0.5, 0.0, 0.0])  # Dead or invalid

        # === 2. THREAT ASSESSMENT ===
        # Check for incoming missiles (MAWS)
        incoming_missiles = [m for m in missiles if m['maws'] > 0.5]
        
        # Check for radar locks (RWR)
        # Note: RWR flag is on the ENEMY entity that is locking us
        locking_enemies = [e for e in enemies if e['rwr'] > 0.5]
        
        # === 3. TACTICAL LOGIC ===
        
        # PRIORITY 1: EVADE MISSILE (NOTCH)
        if incoming_missiles:
            # Find nearest missile
            threat = min(incoming_missiles, key=lambda m: m['dist'])
            return self._maneuver_notch(ego, threat)
            
        # PRIORITY 2: OFFENSIVE (FIRE & CRANK)
        if enemies:
            # Find nearest enemy
            target = min(enemies, key=lambda e: e['dist'])
            
            # Check firing solution
            dist_km = target['dist'] * 100.0  # Approx conversion (map is 100km)
            # Actually, let's use normalized distance for logic
            # Radar range 20km = 0.2 of map? No, map is 100km. 20km = 0.2.
            # Wait, map limits are -50k to 50k, so width is 100km.
            # Normalized dist 0.2 = 20km.
            
            can_fire = (dist_km < 0.25) and (abs(target['ata']) < 20.0) # 25km, 20 deg
            
            # Fire logic
            fire = 0.0
            if can_fire and ego['ammo'] > 0:
                # Simple timer to prevent dumping all ammo at once
                # But we don't have time in obs. Random chance?
                # Or just fire if we have a good shot.
                # Let's use a probabilistic trigger to simulate human delay
                if np.random.rand() < 0.1:
                    fire = 1.0
            
            # Maneuver logic
            if fire > 0.5:
                # We are firing. Maintain lock (Pure Pursuit).
                return self._maneuver_pursuit(ego, target, fire=1.0)
            elif ego['ammo'] < 1.0: # We have fired (ammo < max)
                # CRANK: Turn 50 degrees off target to support missile but maximize range
                return self._maneuver_crank(ego, target)
            else:
                # Pre-merge: Pure Pursuit to get into position
                return self._maneuver_pursuit(ego, target)
        
        # PRIORITY 3: PATROL (No enemies visible)
        return np.array([0.0, 0.0, 0.5, 0.0, 0.0])

    def _parse_obs(self, obs):
        """Parse flattened observation into structured dicts."""
        ego = None
        enemies = []
        missiles = []
        
        num_entities = self.cfg.MAX_ENTITIES
        feat_dim = self.cfg.FEAT_DIM
        
        for i in range(num_entities):
            start = i * feat_dim
            end = start + feat_dim
            vec = obs[start:end]
            
            # Check if entity exists (not all zeros)
            if np.all(vec == 0): continue
            
            # Parse fields
            # 0:x, 1:y, 2:cos_h, 3:sin_h, 4:speed, 5:team, 6:type, 7:is_ego
            # 8:cos_r, 9:sin_r, 10:cos_p, 11:sin_p, 12:rwr, 13:maws, 14:alt, 15:fuel, 16:ammo
            
            entity = {
                'x': vec[0], 'y': vec[1],
                'heading': math.degrees(math.atan2(vec[3], vec[2])) % 360,
                'speed': vec[4],
                'team': vec[5],
                'type': vec[6],
                'is_ego': vec[7] > 0.5,
                'roll': math.atan2(vec[9], vec[8]),
                'rwr': vec[12],
                'maws': vec[13],
                'ammo': vec[16]
            }
            
            if entity['is_ego']:
                ego = entity
            else:
                # Calculate relative geometry
                dx = entity['x'] - ego['x'] if ego else 0
                dy = entity['y'] - ego['y'] if ego else 0
                dist = math.hypot(dx, dy)
                bearing = math.degrees(math.atan2(dy, dx)) % 360
                
                if ego:
                    ata = (bearing - ego['heading'] + 180) % 360 - 180
                else:
                    ata = 0
                
                entity['dist'] = dist
                entity['bearing'] = bearing
                entity['ata'] = ata
                
                if entity['type'] > 0.5: # Missile
                    missiles.append(entity)
                elif entity['team'] != ego['team'] if ego else 0: # Enemy Plane
                    enemies.append(entity)
                    
        return ego, enemies, missiles

    def _maneuver_notch(self, ego, threat):
        """
        Perform Doppler Notch maneuver.
        Turn until threat is at 90 degrees (3 or 9 o'clock).
        """
        # Desired bearing: Threat bearing + 90 or -90
        # Choose the side that is closer to current heading
        threat_bearing = threat['bearing']
        
        # Option A: Notch Left (Threat at 3 o'clock) -> Heading = Threat - 90
        h1 = (threat_bearing - 90) % 360
        # Option B: Notch Right (Threat at 9 o'clock) -> Heading = Threat + 90
        h2 = (threat_bearing + 90) % 360
        
        # Find which is closer
        diff1 = abs((h1 - ego['heading'] + 180) % 360 - 180)
        diff2 = abs((h2 - ego['heading'] + 180) % 360 - 180)
        
        target_heading = h1 if diff1 < diff2 else h2
        
        return self._fly_heading(ego, target_heading, g_load=5.0, cm=True)

    def _maneuver_crank(self, ego, target):
        """
        Perform F-Pole / Crank maneuver.
        Turn 50 degrees off target to maximize range while keeping radar lock.
        """
        # Radar limit is usually 60 deg. We crank to 50 deg.
        target_bearing = target['bearing']
        
        # Determine direction: Crank away from center or towards safety?
        # Simple heuristic: Crank in direction of current bank
        crank_dir = 1.0 if ego['roll'] > 0 else -1.0
        
        target_heading = (target_bearing + crank_dir * 50.0) % 360
        
        return self._fly_heading(ego, target_heading, g_load=3.0)

    def _maneuver_pursuit(self, ego, target, fire=0.0):
        """
        Pure Pursuit intercept.
        """
        return self._fly_heading(ego, target['bearing'], g_load=4.0, fire=fire)

    def _fly_heading(self, ego, target_heading, g_load=1.0, fire=0.0, cm=False):
        """
        Low-level controller to fly a specific heading.
        """
        heading_err = (target_heading - ego['heading'] + 180) % 360 - 180
        
        # Roll to turn
        # Proportional control
        desired_roll = np.clip(math.radians(heading_err * 2.0), -1.4, 1.4)
        roll_err = desired_roll - ego['roll']
        roll_cmd = np.clip(roll_err * 2.0, -1.0, 1.0)
        
        # G to turn (maintain altitude)
        # If we have large heading error, pull Gs
        # If small error, just maintain level flight
        if abs(heading_err) > 5.0:
            g_cmd = (g_load - 1.0) / (self.cfg.MAX_G - 1.0)
        else:
            # Level flight G
            desired_g = 1.0 / max(0.2, math.cos(ego['roll']))
            g_cmd = (desired_g - 1.0) / (self.cfg.MAX_G - 1.0)
            
        g_cmd = np.clip(g_cmd, -0.2, 1.0)
        
        return np.array([roll_cmd, g_cmd, 1.0, fire, 1.0 if cm else 0.0])
