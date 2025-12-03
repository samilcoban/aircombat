# ================================================
# FILE: src/core_flat.py
# ================================================
"""
FLAT-EARTH AIR COMBAT PHYSICS ENGINE (UPDATED FOR 3D & OPTIMIZED)

Updates:
1. Vectorized Spatial Cache: Replaces repeated N*N loops with numpy broadcasting.
2. 3D Missile Guidance: Full spherical Proportional Navigation (Yaw + Pitch).
3. 3D Physics Fix: Velocity projected correctly onto horizontal plane.
4. Optimized Logic: Squared distances for collisions, pre-calculated AI actions.
"""
import numpy as np
import math
from dataclasses import dataclass
from config import Config


# ================================================
# GEOMETRY UTILITIES
# ================================================

def dist_2d(x1, y1, x2, y2):
    """Euclidean distance between two points in 2D plane (Ground Range)."""
    return math.hypot(x2 - x1, y2 - y1)


def bearing_deg(x1, y1, x2, y2):
    """Calculate bearing from point 1 to point 2 in degrees (0=North, 90=East)."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0


def angle_between_vectors_degrees(v1, v2):
    """Returns the angle in degrees between two 3D vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    dot_product = np.dot(v1, v2)
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


@dataclass
class Entity:
    uid: int
    team: str
    type: str

    # Cartesian Coordinates (Meters)
    x: float
    y: float
    alt: float

    # Attitude
    heading: float  # Degrees (0-360)
    roll: float = 0.0  # Radians
    pitch: float = 0.0  # Radians

    # Physics State
    speed: float = 0.0  # Knots
    g_load: float = 1.0

    # Resources
    fuel: float = 1.0
    ammo: int = 4
    chaff: int = 20
    cm_active: bool = False

    # Logic
    target_id: int = None
    time_alive: float = 0.0
    owner_id: int = None


class AirCombatCore:
    """
    Core simulation engine. Manages entities, physics integration, and combat events.
    """

    def __init__(self):
        self.cfg = Config
        self.entities = {}
        self.next_uid = 1
        self.events = []
        self.time = 0.0

        # Spatial Cache Containers
        self.cached_step = -1
        self.dist_matrix = None
        self.rel_pos_matrix = None
        self.rel_vel_matrix = None
        self.uid_to_index = {}

    def spawn(self, x, y, alt, heading, speed, team, etype):
        # FIX 1.1: Accept 'alt' argument instead of hardcoding 10000.0
        e = Entity(
            uid=self.next_uid, team=team, type=etype,
            x=x, y=y, alt=alt,
            heading=heading, speed=speed
        )
        e.ammo = self.cfg.MAX_MISSILES if etype == "plane" else 0
        e.chaff = self.cfg.MAX_CHAFF if etype == "plane" else 0
        e.fuel = 1.0

        self.entities[self.next_uid] = e
        self.next_uid += 1
        return e.uid

    def update_spatial_cache(self):
        """
        Calculates N x N relative metrics once per step using Numpy broadcasting.
        FIX 2.2: Spatial Caching.
        """
        # Check if already updated this step
        if self.cached_step == self.time: return

        # 1. Snapshot current entities and build index map
        current_uids = list(self.entities.keys())
        if not current_uids: return

        self.uid_to_index = {uid: i for i, uid in enumerate(current_uids)}

        # 2. Build Arrays (N, 3)
        pos_list = []
        vel_list = []

        for uid in current_uids:
            e = self.entities[uid]
            pos_list.append([e.x, e.y, e.alt])
            vel_list.append(self._get_velocity_vector(e))

        pos_arr = np.array(pos_list)
        vel_arr = np.array(vel_list)

        # 3. Broadcasting (N, N, 3)
        # Relative Position: Target - Ego
        self.rel_pos_matrix = pos_arr[:, None, :] - pos_arr[None, :, :]

        # 4. Distance Matrix (N, N)
        self.dist_matrix = np.linalg.norm(self.rel_pos_matrix, axis=2)

        # 5. Relative Velocity (N, N, 3)
        self.rel_vel_matrix = vel_arr[:, None, :] - vel_arr[None, :, :]

        self.cached_step = self.time

    def get_relative_data(self, uid_a, uid_b):
        """O(1) retrieval of pre-calculated relative data from cache."""
        if uid_a not in self.uid_to_index or uid_b not in self.uid_to_index:
            return None, None

        idx_a = self.uid_to_index[uid_a]
        idx_b = self.uid_to_index[uid_b]

        # Returns: (Distance, Relative_Position_Vector, Relative_Velocity_Vector)
        return (
            self.dist_matrix[idx_a, idx_b],
            self.rel_pos_matrix[idx_a, idx_b],
            self.rel_vel_matrix[idx_a, idx_b]
        )

    def _get_velocity_vector(self, ent):
        """Helper to get 3D velocity vector."""
        h_r = math.radians(ent.heading)
        p_r = ent.pitch
        spd_ms = ent.speed * 0.514444
        return np.array([
            spd_ms * math.cos(p_r) * math.cos(h_r),
            spd_ms * math.cos(p_r) * math.sin(h_r),
            spd_ms * math.sin(p_r)
        ])

    def step(self, actions, kappa=0.0):
        """
        Advance simulation by one timestep (DT) using physics sub-stepping.
        FIX 2.1: Optimized Step Logic.
        """
        self.events = []

        # 1. PRE-CALCULATE AI ACTIONS (Do this ONCE per step)
        ai_actions = {}
        for uid, ent in self.entities.items():
            if ent.type == "plane" and uid not in actions:
                ai_actions[uid] = self._calculate_ai_action(ent, kappa)

        # 2. Physics Sub-stepping
        for substep in range(self.cfg.PHYSICS_SUBSTEPS):
            is_first_substep = (substep == 0)

            # Update Planes
            for uid, ent in list(self.entities.items()):
                if ent.type == "plane":
                    # Get action from RL agent OR pre-calculated AI
                    current_action = actions.get(uid, ai_actions.get(uid))

                    if current_action is not None:
                        self._update_plane_physics(ent, current_action, is_first_substep)

            # Update Missiles (Keep inside loop, physics needs high freq)
            for uid, ent in list(self.entities.items()):
                if ent.type == "missile":
                    self._update_missile(ent)

            # Solve Collisions (Keep inside loop to prevent tunneling)
            self._resolve_collisions()
            self._check_midair_collisions()

        self.time += self.cfg.DT

    def get_sensor_state(self, observer_uid, target_uid):
        """
        Calculates if the observer can SEE and LOCK the target using 3D geometry.
        """
        obs = self.entities[observer_uid]
        tgt = self.entities[target_uid]

        # 1. 3D Position Vectors
        pos_obs = np.array([obs.x, obs.y, obs.alt])
        pos_tgt = np.array([tgt.x, tgt.y, tgt.alt])
        vec_to_target = pos_tgt - pos_obs
        dist_3d = np.linalg.norm(vec_to_target)

        # 2. Observer Boresight Vector
        h_rad = math.radians(obs.heading)
        p_rad = obs.pitch
        obs_boresight = np.array([
            math.cos(p_rad) * math.cos(h_rad),
            math.cos(p_rad) * math.sin(h_rad),
            math.sin(p_rad)
        ])

        # 3. 3D Angle Off Boresight
        angle_off_3d = angle_between_vectors_degrees(obs_boresight, vec_to_target)

        # 4. Doppler Notch
        vel_obs = self._get_velocity_vector(obs)
        vel_tgt = self._get_velocity_vector(tgt)
        rel_vel = vel_tgt - vel_obs

        if dist_3d > 0:
            u_los = vec_to_target / dist_3d
            closure_speed = -np.dot(rel_vel, u_los)
            is_notched = abs(closure_speed) < (self.cfg.RADAR_NOTCH_SPEED_KNOTS * 0.514444)
        else:
            is_notched = False

        # FIX 1.5: Use half-angle for FOV check
        VISUAL_RANGE = 5000.0
        is_visual = (dist_3d < VISUAL_RANGE)
        half_fov = self.cfg.RADAR_FOV_DEG / 2.0

        is_radar_detect = (
                (dist_3d < self.cfg.RADAR_RANGE_KM * 1000.0) and
                (angle_off_3d < half_fov) and
                (not is_notched)
        )

        is_radar_lock = (
                is_radar_detect and
                (dist_3d < (self.cfg.RADAR_RANGE_KM * 1000.0) * 0.75) and
                (angle_off_3d < half_fov * 0.80)
        )

        return (is_visual or is_radar_detect), is_radar_lock

    def _get_air_density(self, alt):
        return math.exp(-alt / self.cfg.SCALE_HEIGHT)

    def _update_plane_physics(self, ent, action, execute_discrete_actions=True):
        """
        6-DOF Lite Flight Physics with Hyper-Speed Fix.
        """
        dt = self.cfg.PHYSICS_DT
        g = self.cfg.GRAVITY
        KNOTS_TO_MS = 0.514444
        MS_TO_KNOTS = 1.94384

        # Inputs
        roll_rate = np.clip(action[0], -1, 1) * math.radians(90.0)
        g_norm = np.clip(action[1], -1, 1)
        target_g = 1.0 + (g_norm * (self.cfg.MAX_G - 1.0))
        if g_norm < 0: target_g = 1.0 + (g_norm * 2.0)
        throttle = (np.clip(action[2], -1, 1) + 1.0) / 2.0

        if execute_discrete_actions:
            if action[3] > 0.0: self._handle_weapons_system(ent)
            ent.cm_active = False
            if len(action) > 4 and action[4] > 0.5 and ent.chaff > 0:
                ent.cm_active = True
                if np.random.rand() < 0.1: ent.chaff -= 1

        # Attitude Updates
        ent.roll += roll_rate * dt
        ent.roll = (ent.roll + math.pi) % (2 * math.pi) - math.pi

        # G-Limits
        safe_speed = max(ent.speed, 10.0)
        max_aero_g = (safe_speed / 200.0) ** 2
        actual_g = min(target_g, max_aero_g)
        ent.g_load = actual_g

        # Stall
        STALL_SPEED = 150.0
        STALL_ONSET = 180.0
        stall_ratio = np.clip((ent.speed - 100.0) / (STALL_ONSET - 100.0), 0.0, 1.0)
        control_authority = 0.2 + (0.8 * stall_ratio)

        # Turning
        horizontal_g = actual_g * math.sin(ent.roll)
        turn_rate = ((horizontal_g * g) / (ent.speed * KNOTS_TO_MS + 1e-5)) * control_authority
        ent.heading = (ent.heading + math.degrees(turn_rate * dt)) % 360.0

        # Pitching
        vertical_g = actual_g * math.cos(ent.roll) - 1.0
        pitch_rate = ((vertical_g * g) / (ent.speed * KNOTS_TO_MS + 1e-5)) * control_authority
        ent.pitch += pitch_rate * dt
        ent.pitch = np.clip(ent.pitch, -1.4, 1.4)

        # Forces
        rho_ratio = self._get_air_density(ent.alt)
        speed_ms = ent.speed * KNOTS_TO_MS

        drag_p = self.cfg.DRAG_PARASITIC_SL * rho_ratio * (speed_ms ** 2)
        drag_i = self.cfg.DRAG_INDUCED_SL * rho_ratio * (actual_g ** 2)
        drag_stall = (1.0 - stall_ratio) * 50.0

        available_thrust = throttle * self.cfg.THRUST_WEIGHT * g * (rho_ratio ** 0.7)
        if ent.fuel > 0:
            burn_rate = throttle / self.cfg.MAX_FUEL_SEC
            ent.fuel -= burn_rate * dt
        else:
            available_thrust = 0.0

        gravity_force = g * math.sin(ent.pitch)
        accel_ms = available_thrust - (drag_p + drag_i + drag_stall) - gravity_force
        ent.speed = ent.speed + (accel_ms * MS_TO_KNOTS) * dt

        if ent.speed < STALL_SPEED:
            nose_drop_rate = 0.5 * (1.0 - stall_ratio)
            ent.pitch -= nose_drop_rate * dt
        ent.speed = max(ent.speed, 0.0)

        # FIX 1.3: Hyper-Speed Movement Fix (3D Projection)
        speed_ms = ent.speed * KNOTS_TO_MS
        v_horizontal = speed_ms * math.cos(ent.pitch)
        v_vertical = speed_ms * math.sin(ent.pitch)

        dist_h = v_horizontal * dt
        ent.x += dist_h * math.cos(math.radians(ent.heading))
        ent.y += dist_h * math.sin(math.radians(ent.heading))

        lift_factor = stall_ratio
        gravity_drop = -9.81 * (1.0 - lift_factor)
        ent.alt += v_vertical * lift_factor * dt + 0.5 * gravity_drop * (dt ** 2)

        if ent.alt <= 0:
            self.events.append({"killer": -1, "victim": ent.uid, "type": "crash"})
            del self.entities[ent.uid]

    def _handle_weapons_system(self, ent):
        """Updated to use 3D geometry and logic decoupling."""
        targets = [e for e in self.entities.values() if e.team != ent.team and e.type == "plane"]
        if not targets: return

        # Sort by 3D distance
        targets.sort(key=lambda t: (ent.x - t.x) ** 2 + (ent.y - t.y) ** 2 + (ent.alt - t.alt) ** 2)

        cannon_range_m = getattr(self.cfg, 'CANNON_RANGE_KM', 1.5) * 1000.0
        cannon_fov_deg = getattr(self.cfg, 'CANNON_FOV_DEG', 10.0)
        missile_min_range_m = 500.0

        for target in targets:
            dx = target.x - ent.x
            dy = target.y - ent.y
            dz = target.alt - ent.alt
            dist_m = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Geometry
            h_rad = math.radians(ent.heading)
            p_rad = ent.pitch
            ego_vec = np.array([
                math.cos(p_rad) * math.cos(h_rad),
                math.cos(p_rad) * math.sin(h_rad),
                math.sin(p_rad)
            ])
            tgt_vec = np.array([dx, dy, dz]) / (dist_m + 1e-5)
            angle = angle_between_vectors_degrees(ego_vec, tgt_vec)

            # FIX 1.4: Decoupled Checks
            # 1. Try Cannon
            if dist_m < cannon_range_m and angle < (cannon_fov_deg / 2.0):
                self._fire_cannon(ent, target, dist_m)
                return

            # 2. Try Missile
            if ent.ammo > 0 and dist_m > missile_min_range_m:
                visible, locking = self.get_sensor_state(ent.uid, target.uid)
                if locking:
                    active_missiles = sum(
                        1 for m in self.entities.values() if m.type == 'missile' and m.owner_id == ent.uid)
                    if active_missiles < self.cfg.MAX_ACTIVE_MISSILES:
                        self._fire_missile(ent, target)
                        return

    def _fire_cannon(self, ent, target, dist):
        self.events.append({"killer": ent.uid, "victim": target.uid, "type": "kill"})
        if target.uid in self.entities: del self.entities[target.uid]

    def _fire_missile(self, ent, target):
        # FIX 1.1: Pass ent.alt to spawn
        m_uid = self.spawn(ent.x, ent.y, ent.alt, ent.heading, ent.speed, ent.team, "missile")
        self.entities[m_uid].target_id = target.uid
        self.entities[m_uid].owner_id = ent.uid
        self.entities[m_uid].time_alive = 0.0

        # FIX: Inherit pitch for 3D shots
        self.entities[m_uid].pitch = ent.pitch

        ent.ammo -= 1
        self.events.append({"shooter": ent.uid, "target": target.uid, "type": "missile_fired"})

    def _calculate_ai_action(self, ent, kappa=0.0):
        """Simple AI Opponent using 3D Distance."""
        targets = [e for e in self.entities.values() if e.team != ent.team and e.type == "plane"]
        if not targets: return [0.0, 0.0, 0.0, 0.0, 0.0]

        # FIX 1.7: Sort by 3D Distance
        target = min(targets, key=lambda t: (ent.x - t.x) ** 2 + (ent.y - t.y) ** 2 + (ent.alt - t.alt) ** 2)

        dx, dy, dz = target.x - ent.x, target.y - ent.y, target.alt - ent.alt
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Steering
        desired_heading = math.degrees(math.atan2(dy, dx)) % 360.0
        heading_err = (desired_heading - ent.heading + 180) % 360 - 180

        if np.random.rand() < kappa:
            return [np.random.uniform(-1, 1), np.random.uniform(-0.5, 1), np.random.uniform(0.5, 1), 0.0, 0.0]

        desired_roll = np.clip(math.radians(heading_err * 2.0), -1.4, 1.4)
        roll_err = desired_roll - ent.roll
        roll_cmd = np.clip(roll_err * 2.0, -1.0, 1.0)

        desired_g = 1.0 / max(0.2, math.cos(ent.roll))
        alt_err = 10000.0 - ent.alt
        desired_g += np.clip(alt_err * 0.001, -0.5, 2.0)
        g_cmd = np.clip((desired_g - 1.0) / (self.cfg.MAX_G - 1.0), -0.2, 1.0)

        roll_cmd += np.random.normal(0, kappa * 0.5)
        g_cmd += np.random.normal(0, kappa * 0.2)
        throttle = 1.0
        fire = 0.0

        if kappa < 0.5:
            # FIX 1.7: Check firing with 3D dist and half FOV
            if dist_3d < self.cfg.RADAR_RANGE_KM * 1000.0 and abs(heading_err) < (self.cfg.RADAR_FOV_DEG / 2.0):
                if np.random.rand() < 0.05: fire = 1.0

        cm = 1.0 if np.random.rand() < 0.01 else 0.0
        return [np.clip(roll_cmd, -1, 1), np.clip(g_cmd, -0.2, 1), throttle, fire, cm]

    def _update_missile(self, ent):
        """
        FIX 1.2: 3D Missile Logic with Pitch/Yaw Guidance.
        """
        dt = self.cfg.PHYSICS_DT
        ent.time_alive += dt
        KNOTS_TO_MS = 0.514444
        MS_TO_KNOTS = 1.94384

        if ent.target_id not in self.entities:
            del self.entities[ent.uid]
            return
        target = self.entities[ent.target_id]

        if target.cm_active and np.random.rand() < self.cfg.CM_SPOOF_PROB:
            del self.entities[ent.uid]
            return

        # 1. 3D Vector to Target
        dx = target.x - ent.x
        dy = target.y - ent.y
        dz = target.alt - ent.alt
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        # 2. Guidance (Pitch & Yaw)
        desired_pitch = math.asin(np.clip(dz / (dist_3d + 1e-5), -1.0, 1.0))
        desired_heading = math.degrees(math.atan2(dy, dx)) % 360.0

        speed_ms = ent.speed * KNOTS_TO_MS
        max_turn_rate = (self.cfg.MISSILE_MAX_G * 9.81) / (speed_ms + 1e-5)
        max_angle_step = max_turn_rate * dt

        # Yaw
        h_diff = (desired_heading - ent.heading + 180) % 360 - 180
        ent.heading += math.degrees(np.clip(math.radians(h_diff), -max_angle_step, max_angle_step))
        ent.heading %= 360.0

        # Pitch
        p_diff = desired_pitch - ent.pitch
        ent.pitch += np.clip(p_diff, -max_angle_step, max_angle_step)

        # 3. Physics
        thrust = 0.0
        if ent.time_alive < self.cfg.MISSILE_BOOST_SEC:
            thrust = self.cfg.MISSILE_BOOST_ACCEL

        drag_p = self.cfg.MISSILE_DRAG_PARASITIC * (speed_ms ** 2)
        # Simplified induced drag approx
        drag_i = self.cfg.MISSILE_DRAG_INDUCED * 1.0

        accel_ms = thrust - (drag_p + drag_i)
        ent.speed += (accel_ms * MS_TO_KNOTS) * dt

        if ent.speed < self.cfg.MISSILE_MIN_SPEED:
            del self.entities[ent.uid]
            return

        # 4. 3D Movement
        dist_h = speed_ms * math.cos(ent.pitch) * dt
        ent.x += dist_h * math.cos(math.radians(ent.heading))
        ent.y += dist_h * math.sin(math.radians(ent.heading))
        ent.alt += speed_ms * math.sin(ent.pitch) * dt

    def _resolve_collisions(self):
        """Check for missile-target collisions using Squared 3D Distance."""
        missiles = [e for e in self.entities.values() if e.type == "missile"]
        PROXIMITY_SQ = 200.0 ** 2

        for m in missiles:
            if m.target_id in self.entities:
                t = self.entities[m.target_id]
                dx = m.x - t.x
                dy = m.y - t.y
                dz = m.alt - t.alt

                # FIX 1.6: Squared Distance
                dist_sq = dx * dx + dy * dy + dz * dz

                if dist_sq < PROXIMITY_SQ:
                    self.events.append({"killer": m.uid, "victim": t.uid, "type": "kill"})
                    if t.uid in self.entities: del self.entities[t.uid]
                    if m.uid in self.entities: del self.entities[m.uid]

    def _check_midair_collisions(self):
        """Check for plane-plane collisions using Squared 3D Distance."""
        planes = [e for e in self.entities.values() if e.type == "plane"]
        COLLISION_SQ = 30.0 ** 2

        for i, p1 in enumerate(planes):
            for p2 in planes[i + 1:]:
                dx = p1.x - p2.x
                dy = p1.y - p2.y
                dz = p1.alt - p2.alt

                dist_sq = dx * dx + dy * dy + dz * dz

                if dist_sq < COLLISION_SQ:
                    self.events.append({"type": "midair_collision", "victim": p1.uid, "killer": p2.uid})
                    if p1.uid in self.entities: del self.entities[p1.uid]
                    if p2.uid in self.entities: del self.entities[p2.uid]
                    break