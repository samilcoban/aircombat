# ================================================
# FILE: src/core_flat.py
# ================================================
"""
Core physics simulation engine using flat-earth model.

This module implements the physics simulation for the air combat environment
using a simplified flat-earth (Cartesian) coordinate system. This approach
is computationally faster than geodetic calculations and sufficient for
training RL agents in typical combat scenarios.

Key Components:
1. Entity: Data class representing aircraft or missiles
2. AirCombatCore: Main simulation engine managing physics integration

Physics Model:
- 6-DOF aircraft dynamics with realistic constraints
- G-loading, stall behavior, and energy management
- Proportional navigation for missiles
- Radar and sensor simulation
- Collision detection
"""
import numpy as np
import math
from dataclasses import dataclass
from config import Config


def dist_3d(x1, y1, z1, x2, y2, z2):
    """
    Calculate Euclidean distance between two points in 3D space (slant range).
    
    Args:
        x1, y1, z1: Coordinates of first point.
        x2, y2, z2: Coordinates of second point.
        
    Returns:
        Distance in meters.
    """
    return math.hypot(x2 - x1, y2 - y1, z2 - z1)


@dataclass
class Entity:
    """
    Represents an aircraft or missile in the simulation.
    
    Coordinate System:
    - X: North (+X = heading 0)
    - Y: East (+Y = heading 90°)
    - Z: Up (altitude in meters)
    
    Attitude Angles (in radians):
    - Heading: 0 = North, π/2 = East
    - Pitch: Positive = nose up
    - Roll: Positive = right wing down
    
    Dynamics Tracking:
    The d_* fields track rate of change (derivatives) of state variables,
    computed each step as the difference from previous values.
    """
    uid: int         # Unique identifier.
    team: str        # 'blue' or 'red'.
    type: str        # 'plane' or 'missile'.

    # Cartesian Coordinates (Meters).
    x: float
    y: float
    alt: float

    # Attitude (RADIANS).
    # 0 = North (+X), PI/2 = East (+Y).
    heading: float
    roll: float = 0.0
    pitch: float = 0.0

    # Physics State.
    speed: float = 0.0  # Knots.
    g_load: float = 1.0  # Current G-loading.

    # Resources.
    fuel: float = 1.0    # Fuel remaining (0-1 normalized).
    ammo: int = 4        # Missiles remaining.
    chaff: int = 20      # Countermeasure rounds.
    cm_active: bool = False  # Countermeasures currently deployed.

    # Logic.
    target_id: int = None    # UID of current target (for missiles).
    time_alive: float = 0.0  # Time since spawn (for missiles).
    owner_id: int = None     # UID of launching aircraft (for missiles).

    # --- Dynamics Tracking ---
    # Previous state for computing derivatives.
    prev_heading: float = 0.0
    prev_pitch: float = 0.0
    prev_roll: float = 0.0
    prev_speed: float = 0.0

    # Rate of change (derivatives).
    d_heading: float = 0.0  # Heading rate (rad/step).
    d_pitch: float = 0.0    # Pitch rate (rad/step).
    d_roll: float = 0.0     # Roll rate (rad/step).
    d_speed: float = 0.0    # Acceleration (knots/step).


class AirCombatCore:
    """
    Core simulation engine. Manages entities, physics integration, and combat events.
    
    Optimized with Vectorized Spatial Cache and Radian-based Physics.
    
    The spatial cache pre-computes pairwise distances, relative positions,
    and geometric relationships between all entities each timestep for
    efficient sensor and observation calculations.
    """

    def __init__(self):
        """Initialize empty simulation."""
        self.cfg = Config
        self.entities = {}       # UID -> Entity mapping.
        self.next_uid = 1        # Counter for generating unique IDs.
        self.events = []         # Events this timestep (kills, crashes, etc.).
        self.time = 0.0          # Simulation time.

        self.missile_registry = {}  # Missile UID -> Owner UID mapping.

        # Spatial Cache Containers.
        # These are recomputed each timestep via update_spatial_cache().
        self.cached_step = -1
        self.dist_matrix = None        # Pairwise distances [N x N].
        self.rel_pos_matrix = None     # Relative positions [N x N x 3].
        self.rel_vel_matrix = None     # Relative velocities [N x N x 3].
        self.ata_cos_matrix = None     # Antenna-Train-Angle cosines [N x N].
        self.aa_cos_matrix = None      # Aspect-Angle cosines [N x N].
        self.local_pos_matrix = None   # Local body-frame positions [N x N x 3].
        self.uid_to_index = {}         # UID -> matrix index mapping.

    def spawn(self, x, y, alt, heading, speed, team, etype):
        """
        Spawn a new entity in the simulation.
        
        Args:
            x, y: Initial position (meters).
            alt: Initial altitude (meters).
            heading: Initial heading (radians).
            speed: Initial speed (knots).
            team: Team identifier ('blue' or 'red').
            etype: Entity type ('plane' or 'missile').
            
        Returns:
            UID of spawned entity.
        """
        e = Entity(
            uid=self.next_uid, team=team, type=etype,
            x=x, y=y, alt=alt,
            heading=heading, speed=speed
        )
        # Initialize prev states to current to avoid massive deltas on spawn.
        e.prev_heading = heading
        e.prev_pitch = 0.0
        e.prev_roll = 0.0
        e.prev_speed = speed

        # Set resources based on entity type.
        e.ammo = self.cfg.MAX_MISSILES if etype == "plane" else 0
        e.chaff = self.cfg.MAX_CHAFF if etype == "plane" else 0
        e.fuel = 1.0

        self.entities[self.next_uid] = e
        self.next_uid += 1
        return e.uid

    def update_spatial_cache(self):
        """
        Vectorized computation of pairwise spatial relationships.
        
        This method pre-computes:
        - Distance matrix (slant range between all entity pairs)
        - Relative position matrix (in global coordinates)
        - Relative velocity matrix
        - ATA (Antenna Train Angle) cosines
        - AA (Aspect Angle) cosines
        - Local position matrix (in each entity's body frame)
        
        This cache enables O(1) lookups for sensor and observation queries.
        """
        if self.cached_step == self.time: return

        current_uids = list(self.entities.keys())
        if not current_uids: return

        self.uid_to_index = {uid: i for i, uid in enumerate(current_uids)}
        n = len(current_uids)

        # Preallocate arrays.
        pos_arr = np.zeros((n, 3), dtype=np.float32)
        vel_arr = np.zeros((n, 3), dtype=np.float32)
        fwd_arr = np.zeros((n, 3), dtype=np.float32)  # Forward (nose) vector.
        rgt_arr = np.zeros((n, 3), dtype=np.float32)  # Right wing vector.
        up_arr = np.zeros((n, 3), dtype=np.float32)   # Up vector.

        # Build entity state arrays.
        for i, uid in enumerate(current_uids):
            e = self.entities[uid]
            pos_arr[i] = [e.x, e.y, e.alt]
            
            # Compute body frame axes from Euler angles.
            ch, sh = math.cos(e.heading), math.sin(e.heading)
            cp, sp = math.cos(e.pitch), math.sin(e.pitch)
            cr, sr = math.cos(e.roll), math.sin(e.roll)
            
            # Forward vector (nose direction).
            fx, fy, fz = cp * ch, cp * sh, sp
            fwd_arr[i] = [fx, fy, fz]
            
            # Velocity vector (forward * speed in m/s).
            spd_ms = e.speed * 0.514444  # Knots to m/s.
            vel_arr[i] = [fx * spd_ms, fy * spd_ms, fz * spd_ms]
            
            # Right vector (right wing direction).
            rx = ch * sp * sr - sh * cr
            ry = sh * sp * sr + ch * cr
            rz = -cp * sr
            rgt_arr[i] = [rx, ry, rz]
            
            # Up vector.
            ux = -ch * sp * cr - sh * sr
            uy = -sh * sp * cr + ch * sr
            uz = cp * cr
            up_arr[i] = [ux, uy, uz]

        # Compute pairwise matrices using broadcasting.
        self.rel_pos_matrix = pos_arr[None, :, :] - pos_arr[:, None, :]
        self.rel_vel_matrix = vel_arr[None, :, :] - vel_arr[:, None, :]
        self.dist_matrix = np.linalg.norm(self.rel_pos_matrix, axis=2)
        
        # Unit line-of-sight vectors.
        safe_dist = self.dist_matrix[:, :, None] + 1e-6
        u_los = self.rel_pos_matrix / safe_dist
        
        # ATA: Cosine of angle from observer's nose to target.
        self.ata_cos_matrix = np.einsum('ijk,ijk->ij', fwd_arr[:, None, :], u_los)
        
        # AA: Cosine of angle from target's nose to observer (aspect angle).
        self.aa_cos_matrix = np.einsum('ijk,ijk->ij', fwd_arr[None, :, :], -u_los)
        
        # Clip to valid cosine range.
        self.ata_cos_matrix = np.clip(self.ata_cos_matrix, -1.0, 1.0)
        self.aa_cos_matrix = np.clip(self.aa_cos_matrix, -1.0, 1.0)
        
        # Local coordinates (target position in observer's body frame).
        local_x = np.einsum('ijk,ik->ij', self.rel_pos_matrix, fwd_arr)
        local_y = np.einsum('ijk,ik->ij', self.rel_pos_matrix, rgt_arr)
        local_z = np.einsum('ijk,ik->ij', self.rel_pos_matrix, up_arr)
        self.local_pos_matrix = np.stack([local_x, local_y, local_z], axis=2)
        
        self.cached_step = self.time

    def get_relative_data(self, uid_a, uid_b):
        """
        Get cached relative data between two entities.
        
        Args:
            uid_a: Observer UID.
            uid_b: Target UID.
            
        Returns:
            Tuple of (distance, rel_pos, rel_vel, ata_cos, aa_cos, local_pos)
            or None if either entity not found.
        """
        if uid_a not in self.uid_to_index or uid_b not in self.uid_to_index: return None
        idx_a = self.uid_to_index[uid_a]
        idx_b = self.uid_to_index[uid_b]
        if idx_a >= self.dist_matrix.shape[0] or idx_b >= self.dist_matrix.shape[0]: return None
        return (
            self.dist_matrix[idx_a, idx_b],
            self.rel_pos_matrix[idx_a, idx_b],
            self.rel_vel_matrix[idx_a, idx_b],
            self.ata_cos_matrix[idx_a, idx_b],
            self.aa_cos_matrix[idx_a, idx_b],
            self.local_pos_matrix[idx_a, idx_b]
        )

    def step(self, actions, kappa=0.0):
        """
        Advance simulation by one timestep.
        
        Args:
            actions: Dict mapping aircraft UID to action array.
                    Actions not provided will be computed by AI.
            kappa: Randomness factor for AI opponents (0=deterministic, 1=random).
        """
        self.events = []

        # 1. Update Deltas (Proprioception) BEFORE physics integration.
        # This computes rate-of-change values for observation features.
        for uid, ent in self.entities.items():
            # Angle wrapping logic (-180 to 180 diff).
            def angle_diff(a, b):
                d = a - b
                return (d + math.pi) % (2 * math.pi) - math.pi

            ent.d_heading = angle_diff(ent.heading, ent.prev_heading)
            ent.d_pitch = ent.pitch - ent.prev_pitch
            ent.d_roll = angle_diff(ent.roll, ent.prev_roll)
            ent.d_speed = ent.speed - ent.prev_speed

            # Update history for next step.
            ent.prev_heading = ent.heading
            ent.prev_pitch = ent.pitch
            ent.prev_roll = ent.roll
            ent.prev_speed = ent.speed

        # 2. AI Logic & Physics Loop.
        # Compute AI actions for entities not controlled by player.
        ai_actions = {}
        for uid, ent in self.entities.items():
            if ent.type == "plane" and uid not in actions:
                ai_actions[uid] = self._calculate_ai_action(ent, kappa)

        # Physics substeps for stability.
        for substep in range(self.cfg.PHYSICS_SUBSTEPS):
            is_first_substep = (substep == 0)
            
            # Update planes.
            for uid, ent in list(self.entities.items()):
                if ent.type == "plane":
                    act = actions.get(uid, ai_actions.get(uid))
                    if act is not None:
                        self._update_plane_physics(ent, act, is_first_substep)
            
            # Update missiles.
            for uid, ent in list(self.entities.items()):
                if ent.type == "missile":
                    self._update_missile(ent)
            
            # Collision detection.
            self._resolve_collisions()
            self._check_midair_collisions()

        self.time += self.cfg.DT

    def get_sensor_state(self, observer_uid, target_uid):
        """
        Determine sensor visibility and lock status.
        
        Args:
            observer_uid: Observing aircraft UID.
            target_uid: Target entity UID.
            
        Returns:
            Tuple (is_visible, is_locked).
            is_visible: True if target is within radar range or visual range.
            is_locked: True if target can be engaged (within lock envelope).
        """
        data = self.get_relative_data(observer_uid, target_uid)
        if data is None: return False, False
        
        dist, rel_pos, rel_vel, ata_cos, _, _ = data
        
        # Radar parameters.
        radar_range_m = self.cfg.RADAR_RANGE_KM * 1000.0
        fov_half_rad = math.radians(self.cfg.RADAR_FOV_DEG / 2.0)
        min_cos_detect = math.cos(fov_half_rad)
        min_cos_lock = math.cos(fov_half_rad * 0.8)  # Lock FOV is narrower.
        
        # Doppler notch detection (target flying perpendicular = invisible).
        is_notched = False
        if dist > 0:
            closure = -np.dot(rel_vel, rel_pos / dist)
            notch_limit = self.cfg.RADAR_NOTCH_SPEED_KNOTS * 0.514444
            is_notched = abs(closure) < notch_limit
        
        # Visual range (always visible if close enough).
        VISUAL_RANGE = 5000.0
        is_visual = (dist < VISUAL_RANGE)
        
        # Radar detection (range + FOV + not notched).
        is_radar_detect = ((dist < radar_range_m) and (ata_cos > min_cos_detect) and (not is_notched))
        
        # Radar lock (stricter requirements).
        is_radar_lock = (is_radar_detect and (dist < radar_range_m * 0.75) and (ata_cos > min_cos_lock))
        
        return (is_visual or is_radar_detect), is_radar_lock

    def _get_air_density(self, alt):
        """
        Calculate atmospheric density ratio using exponential model.
        
        Args:
            alt: Altitude in meters.
            
        Returns:
            Density ratio (1.0 at sea level, decreasing with altitude).
        """
        return math.exp(-alt / self.cfg.SCALE_HEIGHT)

    def _update_plane_physics(self, ent, action, execute_discrete_actions=True):
        """
        Update aircraft physics for one substep.
        
        This implements a simplified 6-DOF flight model including:
        - Roll rate command
        - G-loading (coordinated turns)
        - Throttle/thrust
        - Aerodynamic drag (parasitic and induced)
        - Stall behavior
        - Gravity
        
        Args:
            ent: The aircraft Entity.
            action: [roll_rate, g_command, throttle, fire, countermeasures].
            execute_discrete_actions: If True, process weapons/CM commands.
        """
        dt = self.cfg.PHYSICS_DT
        g = self.cfg.GRAVITY
        KNOTS_TO_MS = 0.514444
        MS_TO_KNOTS = 1.94384
        
        # Parse continuous actions.
        roll_rate = np.clip(action[0], -1, 1) * (math.pi / 2.0)
        g_norm = np.clip(action[1], -1, 1)
        
        # Convert normalized G command to actual G.
        if g_norm >= 0:
            target_g = 1.0 + (g_norm * (self.cfg.MAX_G - 1.0))
        else:
            MIN_NEG_G = -3.0
            target_g = 1.0 + (g_norm * (1.0 - MIN_NEG_G))
        
        throttle = (np.clip(action[2], -1, 1) + 1.0) / 2.0
        
        # Process discrete actions (weapons, countermeasures).
        if execute_discrete_actions:
            if action[3] > 0.0: self._handle_weapons_system(ent)
            ent.cm_active = False
            if len(action) > 4 and action[4] > 0.5 and ent.chaff > 0:
                ent.cm_active = True
                if np.random.rand() < 0.1: ent.chaff -= 1  # Consume CM.
        
        # Update roll.
        ent.roll += roll_rate * dt
        ent.roll = (ent.roll + math.pi) % (2 * math.pi) - math.pi
        
        # G-loading is limited by available aerodynamic lift.
        safe_speed = max(ent.speed, 10.0)
        max_aero_g = (safe_speed / 200.0) ** 2  # More speed = more G available.
        actual_g = min(target_g, max_aero_g)
        ent.g_load = actual_g
        
        # Dynamic pressure factor (control authority).
        q_factor = np.clip(ent.speed / 100.0, 0.0, 1.0)
        v_denom = max(ent.speed * KNOTS_TO_MS, 10.0)
        
        # Horizontal G produces turn rate.
        horizontal_g = actual_g * math.sin(ent.roll)
        turn_rate = (horizontal_g * g) / v_denom
        turn_rate *= q_factor
        ent.heading = (ent.heading + turn_rate * dt) % (2 * math.pi)
        
        # Vertical G produces pitch rate.
        vertical_g = actual_g * math.cos(ent.roll) - 1.0  # Subtract 1G for level flight.
        pitch_rate = (vertical_g * g) / v_denom
        pitch_rate *= q_factor
        ent.pitch += pitch_rate * dt
        ent.pitch = np.clip(ent.pitch, -1.4, 1.4)  # Limit pitch angle.
        
        # Stall dynamics.
        STALL_SPEED = 150.0
        stall_ratio = np.clip((ent.speed - 100.0) / 80.0, 0.0, 1.0)
        
        # Aerodynamic forces.
        rho = self._get_air_density(ent.alt)
        v_ms = ent.speed * KNOTS_TO_MS
        
        # Drag: parasitic (V^2) + induced (G^2) + stall penalty.
        drag_p = self.cfg.DRAG_PARASITIC_SL * rho * (v_ms ** 2)
        drag_i = self.cfg.DRAG_INDUCED_SL * rho * (actual_g ** 2)
        drag_stall = (1.0 - stall_ratio) * 50.0
        
        # Thrust.
        thrust = throttle * self.cfg.THRUST_WEIGHT * g * (rho ** 0.7)
        if ent.fuel <= 0:
            thrust = 0.0
        else:
            ent.fuel -= (throttle / self.cfg.MAX_FUEL_SEC) * dt
        
        # Gravity component along flight path.
        gravity_drag = g * math.sin(ent.pitch)
        
        # Net acceleration.
        accel = thrust - (drag_p + drag_i + drag_stall) - gravity_drag
        ent.speed += (accel * MS_TO_KNOTS) * dt
        
        # Stall behavior: nose drops when stalled.
        if ent.speed < STALL_SPEED:
            ent.pitch -= 0.5 * (1.0 - stall_ratio) * dt
        
        ent.speed = max(ent.speed, 0.0)
        
        # Position integration.
        v_ms = ent.speed * KNOTS_TO_MS
        v_horiz = v_ms * math.cos(ent.pitch)
        v_vert = v_ms * math.sin(ent.pitch)
        dist_h = v_horiz * dt
        ent.x += dist_h * math.cos(ent.heading)
        ent.y += dist_h * math.sin(ent.heading)
        ent.alt += v_vert * dt
        
        # Ground collision.
        if ent.alt <= 0:
            self.events.append({"killer": -1, "victim": ent.uid, "type": "crash"})
            del self.entities[ent.uid]

    def _handle_weapons_system(self, ent):
        """
        Process weapons engagement for an aircraft.
        
        Checks for cannon or missile engagement opportunities
        and executes appropriate attack.
        
        Args:
            ent: The attacking aircraft Entity.
        """
        cannon_range = getattr(self.cfg, 'CANNON_RANGE_KM', 1.5) * 1000.0
        cannon_cos = math.cos(math.radians(getattr(self.cfg, 'CANNON_FOV_DEG', 10.0) / 2.0))
        
        for tid, t in self.entities.items():
            if t.team == ent.team or t.type != "plane": continue
            
            data = self.get_relative_data(ent.uid, tid)
            if data is None: continue
            
            dist, _, _, ata_cos, _, _ = data
            
            # Cannon kill (close range, aligned).
            if dist < cannon_range and ata_cos > cannon_cos:
                self._fire_cannon(ent, t, dist)
                return
            
            # Missile launch (if have ammo and target is locked).
            if ent.ammo > 0 and dist > 500.0:
                vis, lock = self.get_sensor_state(ent.uid, tid)
                if lock:
                    # Check max simultaneous missiles in flight.
                    active = sum(1 for m in self.entities.values() if m.type == 'missile' and m.owner_id == ent.uid)
                    if active < self.cfg.MAX_ACTIVE_MISSILES:
                        self._fire_missile(ent, t)
                        return

    def _fire_cannon(self, ent, target, dist):
        """
        Execute cannon kill.
        
        Args:
            ent: Attacking aircraft.
            target: Target aircraft.
            dist: Distance (unused, for consistency).
        """
        self.events.append({"killer": ent.uid, "victim": target.uid, "type": "kill"})
        if target.uid in self.entities: del self.entities[target.uid]

    def _fire_missile(self, ent, target):
        """
        Launch a missile at target.
        
        Creates new missile entity with initial state copied from launcher.
        
        Args:
            ent: Launching aircraft.
            target: Target aircraft.
        """
        m_uid = self.spawn(ent.x, ent.y, ent.alt, ent.heading, ent.speed, ent.team, "missile")
        m_ent = self.entities[m_uid]
        m_ent.target_id = target.uid
        m_ent.owner_id = ent.uid
        m_ent.pitch = ent.pitch
        self.missile_registry[m_uid] = ent.uid
        ent.ammo -= 1
        self.events.append({"shooter": ent.uid, "target": target.uid, "type": "missile_fired"})

    def _calculate_ai_action(self, ent, kappa=0.0):
        """
        Calculate AI action for non-player aircraft.
        
        Simple pursuit logic: turn toward closest enemy and maintain altitude.
        
        Args:
            ent: The AI-controlled aircraft.
            kappa: Randomness factor (0=deterministic, 1=random).
            
        Returns:
            Action array [roll, g, throttle, fire, cm].
        """
        # Find closest enemy.
        best_tid = None
        min_dist = float('inf')
        best_rel_pos = None
        
        for tid, t in self.entities.items():
            if t.team == ent.team or t.type != "plane": continue
            data = self.get_relative_data(ent.uid, tid)
            if data and data[0] < min_dist:
                min_dist = data[0]
                best_tid = tid
                best_rel_pos = data[1]
        
        if best_tid is None: return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        dx, dy, dz = best_rel_pos
        des_heading = math.atan2(dy, dx)
        h_err = (des_heading - ent.heading + math.pi) % (2 * math.pi) - math.pi
        
        # Random noise injection.
        if np.random.rand() < kappa:
            return [np.random.uniform(-1, 1), np.random.uniform(-0.5, 1), 0.5, 0, 0]
        
        # Pursuit steering.
        des_roll = np.clip(h_err * 2.0, -1.4, 1.4)
        roll_cmd = np.clip((des_roll - ent.roll) * 2.0, -1.0, 1.0)
        
        # Altitude hold with bank compensation.
        bank_factor = 1.0 / max(0.2, math.cos(ent.roll))
        alt_err = 10000.0 - ent.alt
        g_cmd = np.clip(((bank_factor + np.clip(alt_err * 0.0005, -0.5, 1.0)) - 1.0) / 8.0, -0.2, 1.0)
        
        # Fire if aligned and close.
        fire = 0.0
        if kappa < 0.5 and min_dist < 20000.0 and abs(h_err) < 0.2:
            if np.random.rand() < 0.05: fire = 1.0
        
        return [roll_cmd, g_cmd, 1.0, fire, 0.0]

    def _update_missile(self, ent):
        """
        Update missile physics and guidance.
        
        Implements proportional navigation with lead pursuit.
        
        Args:
            ent: The missile Entity.
        """
        dt = self.cfg.PHYSICS_DT
        ent.time_alive += dt
        
        # Check if target still exists.
        if ent.target_id not in self.entities:
            del self.entities[ent.uid];
            return
        
        t = self.entities[ent.target_id]
        
        # Countermeasure spoofing.
        if t.cm_active and np.random.rand() < self.cfg.CM_SPOOF_PROB:
            del self.entities[ent.uid];
            return

        # Calculate intercept geometry.
        dx, dy, dz = t.x - ent.x, t.y - ent.y, t.alt - ent.alt
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Simplified closing speed estimate (assume head-on max closure).
        closure_est = ent.speed * 0.514 + t.speed * 0.514  # m/s

        # Time to intercept.
        tti = dist / (closure_est + 1e-5)

        # Predict target position based on velocity.
        t_v_ms = t.speed * 0.514444
        t_vx = t_v_ms * math.cos(t.pitch) * math.cos(t.heading)
        t_vy = t_v_ms * math.cos(t.pitch) * math.sin(t.heading)
        t_vz = t_v_ms * math.sin(t.pitch)

        # Lead point (where target will be at intercept time).
        lead_x = dx + t_vx * tti
        lead_y = dy + t_vy * tti
        lead_z = dz + t_vz * tti

        # Calculate desired angles to lead point.
        des_pitch = math.asin(np.clip(lead_z / (math.sqrt(lead_x ** 2 + lead_y ** 2 + lead_z ** 2) + 1e-5), -1, 1))
        des_head = math.atan2(lead_y, lead_x)

        # Guidance law with turn rate limiting.
        spd = ent.speed * 0.514444
        max_turn = (self.cfg.MISSILE_MAX_G * 9.81) / (spd + 1e-5)
        max_step = max_turn * dt
        
        # Heading update.
        h_diff = (des_head - ent.heading + math.pi) % (2 * math.pi) - math.pi
        ent.heading = (ent.heading + np.clip(h_diff, -max_step, max_step)) % (2 * math.pi)
        
        # Pitch update.
        p_diff = des_pitch - ent.pitch
        ent.pitch += np.clip(p_diff, -max_step, max_step)
        
        # Thrust and drag.
        thrust = self.cfg.MISSILE_BOOST_ACCEL if ent.time_alive < self.cfg.MISSILE_BOOST_SEC else 0.0
        drag = self.cfg.MISSILE_DRAG_PARASITIC * spd ** 2
        ent.speed += (thrust - drag) * 1.944 * dt
        
        # Missile runs out of energy.
        if ent.speed < self.cfg.MISSILE_MIN_SPEED:
            del self.entities[ent.uid];
            return
        
        # Position integration.
        v_h = spd * math.cos(ent.pitch)
        ent.x += v_h * math.cos(ent.heading) * dt
        ent.y += v_h * math.sin(ent.heading) * dt
        ent.alt += spd * math.sin(ent.pitch) * dt

    def _resolve_collisions(self):
        """
        Check for missile-target collision (impact).
        
        Missiles impact when within 200m of their target.
        """
        ms = [e for e in self.entities.values() if e.type == "missile"]
        sq_lim = 200.0 ** 2  # Impact radius squared.
        
        for m in ms:
            if m.target_id in self.entities:
                t = self.entities[m.target_id]
                ds = (m.x - t.x) ** 2 + (m.y - t.y) ** 2 + (m.alt - t.alt) ** 2
                if ds < sq_lim:
                    owner_id = self.missile_registry.get(m.uid, -1)
                    self.events.append({
                        "killer": m.uid,
                        "victim": t.uid,
                        "type": "kill",
                        "owner_id": owner_id
                    })
                    del self.entities[t.uid]
                    del self.entities[m.uid]

    def _check_midair_collisions(self):
        """
        Check for aircraft-aircraft midair collision.
        
        Aircraft collide if within 30m of each other.
        """
        ps = [e for e in self.entities.values() if e.type == "plane"]
        sq_lim = 30.0 ** 2  # Collision radius squared.
        
        for i, p1 in enumerate(ps):
            for p2 in ps[i + 1:]:
                ds = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.alt - p2.alt) ** 2
                if ds < sq_lim:
                    self.events.append({"type": "midair", "victim": p1.uid, "killer": p2.uid})
                    if p1.uid in self.entities: del self.entities[p1.uid]
                    if p2.uid in self.entities: del self.entities[p2.uid]
                    break