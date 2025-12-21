# ================================================
# FILE: src/core_flat.py
# ================================================
import numpy as np
import math
from dataclasses import dataclass
from config import Config


def dist_3d(x1, y1, z1, x2, y2, z2):
    """Euclidean distance between two points in 3D space (Slant Range)."""
    return math.hypot(x2 - x1, y2 - y1, z2 - z1)


@dataclass
class Entity:
    uid: int
    team: str
    type: str

    # Cartesian Coordinates (Meters)
    x: float
    y: float
    alt: float

    # Attitude (RADIANS)
    # 0 = North (+X), PI/2 = East (+Y)
    heading: float
    roll: float = 0.0
    pitch: float = 0.0

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

    # --- NEW: Explicit Dynamics Tracking ---
    prev_heading: float = 0.0
    prev_pitch: float = 0.0
    prev_roll: float = 0.0
    prev_speed: float = 0.0

    d_heading: float = 0.0
    d_pitch: float = 0.0
    d_roll: float = 0.0
    d_speed: float = 0.0


class AirCombatCore:
    """
    Core simulation engine. Manages entities, physics integration, and combat events.
    Optimized with Vectorized Spatial Cache and Radian-based Physics.
    """

    def __init__(self):
        self.cfg = Config
        self.entities = {}
        self.next_uid = 1
        self.events = []
        self.time = 0.0

        self.missile_registry = {}

        # Spatial Cache Containers
        self.cached_step = -1
        self.dist_matrix = None
        self.rel_pos_matrix = None
        self.rel_vel_matrix = None
        self.ata_cos_matrix = None
        self.aa_cos_matrix = None
        self.local_pos_matrix = None
        self.uid_to_index = {}

    def spawn(self, x, y, alt, heading, speed, team, etype):
        e = Entity(
            uid=self.next_uid, team=team, type=etype,
            x=x, y=y, alt=alt,
            heading=heading, speed=speed
        )
        # Init prev states to current to avoid massive deltas on spawn
        e.prev_heading = heading
        e.prev_pitch = 0.0
        e.prev_roll = 0.0
        e.prev_speed = speed

        e.ammo = self.cfg.MAX_MISSILES if etype == "plane" else 0
        e.chaff = self.cfg.MAX_CHAFF if etype == "plane" else 0
        e.fuel = 1.0

        self.entities[self.next_uid] = e
        self.next_uid += 1
        return e.uid

    def update_spatial_cache(self):
        if self.cached_step == self.time: return

        current_uids = list(self.entities.keys())
        if not current_uids: return

        self.uid_to_index = {uid: i for i, uid in enumerate(current_uids)}
        n = len(current_uids)

        pos_arr = np.zeros((n, 3), dtype=np.float32)
        vel_arr = np.zeros((n, 3), dtype=np.float32)
        fwd_arr = np.zeros((n, 3), dtype=np.float32)
        rgt_arr = np.zeros((n, 3), dtype=np.float32)
        up_arr = np.zeros((n, 3), dtype=np.float32)

        for i, uid in enumerate(current_uids):
            e = self.entities[uid]
            pos_arr[i] = [e.x, e.y, e.alt]
            ch, sh = math.cos(e.heading), math.sin(e.heading)
            cp, sp = math.cos(e.pitch), math.sin(e.pitch)
            cr, sr = math.cos(e.roll), math.sin(e.roll)
            fx, fy, fz = cp * ch, cp * sh, sp
            fwd_arr[i] = [fx, fy, fz]
            spd_ms = e.speed * 0.514444
            vel_arr[i] = [fx * spd_ms, fy * spd_ms, fz * spd_ms]
            rx = ch * sp * sr - sh * cr
            ry = sh * sp * sr + ch * cr
            rz = -cp * sr
            rgt_arr[i] = [rx, ry, rz]
            ux = -ch * sp * cr - sh * sr
            uy = -sh * sp * cr + ch * sr
            uz = cp * cr
            up_arr[i] = [ux, uy, uz]

        self.rel_pos_matrix = pos_arr[None, :, :] - pos_arr[:, None, :]
        self.rel_vel_matrix = vel_arr[None, :, :] - vel_arr[:, None, :]
        self.dist_matrix = np.linalg.norm(self.rel_pos_matrix, axis=2)
        safe_dist = self.dist_matrix[:, :, None] + 1e-6
        u_los = self.rel_pos_matrix / safe_dist
        self.ata_cos_matrix = np.einsum('ijk,ijk->ij', fwd_arr[:, None, :], u_los)
        self.aa_cos_matrix = np.einsum('ijk,ijk->ij', fwd_arr[None, :, :], -u_los)
        self.ata_cos_matrix = np.clip(self.ata_cos_matrix, -1.0, 1.0)
        self.aa_cos_matrix = np.clip(self.aa_cos_matrix, -1.0, 1.0)
        local_x = np.einsum('ijk,ik->ij', self.rel_pos_matrix, fwd_arr)
        local_y = np.einsum('ijk,ik->ij', self.rel_pos_matrix, rgt_arr)
        local_z = np.einsum('ijk,ik->ij', self.rel_pos_matrix, up_arr)
        self.local_pos_matrix = np.stack([local_x, local_y, local_z], axis=2)
        self.cached_step = self.time

    def get_relative_data(self, uid_a, uid_b):
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
        self.events = []

        # 1. Update Deltas (Proprioception) BEFORE physics integration
        for uid, ent in self.entities.items():
            # Angle wrapping logic (-180 to 180 diff)
            def angle_diff(a, b):
                d = a - b
                return (d + math.pi) % (2 * math.pi) - math.pi

            ent.d_heading = angle_diff(ent.heading, ent.prev_heading)
            ent.d_pitch = ent.pitch - ent.prev_pitch
            ent.d_roll = angle_diff(ent.roll, ent.prev_roll)
            ent.d_speed = ent.speed - ent.prev_speed

            # Update history
            ent.prev_heading = ent.heading
            ent.prev_pitch = ent.pitch
            ent.prev_roll = ent.roll
            ent.prev_speed = ent.speed

        # 2. AI Logic & Physics Loop
        ai_actions = {}
        for uid, ent in self.entities.items():
            if ent.type == "plane" and uid not in actions:
                ai_actions[uid] = self._calculate_ai_action(ent, kappa)

        for substep in range(self.cfg.PHYSICS_SUBSTEPS):
            is_first_substep = (substep == 0)
            # Planes
            for uid, ent in list(self.entities.items()):
                if ent.type == "plane":
                    act = actions.get(uid, ai_actions.get(uid))
                    if act is not None:
                        self._update_plane_physics(ent, act, is_first_substep)
            # Missiles
            for uid, ent in list(self.entities.items()):
                if ent.type == "missile":
                    self._update_missile(ent)
            # Collisions
            self._resolve_collisions()
            self._check_midair_collisions()

        self.time += self.cfg.DT

    def get_sensor_state(self, observer_uid, target_uid):
        data = self.get_relative_data(observer_uid, target_uid)
        if data is None: return False, False
        dist, rel_pos, rel_vel, ata_cos, _, _ = data
        radar_range_m = self.cfg.RADAR_RANGE_KM * 1000.0
        fov_half_rad = math.radians(self.cfg.RADAR_FOV_DEG / 2.0)
        min_cos_detect = math.cos(fov_half_rad)
        min_cos_lock = math.cos(fov_half_rad * 0.8)
        is_notched = False
        if dist > 0:
            closure = -np.dot(rel_vel, rel_pos / dist)
            notch_limit = self.cfg.RADAR_NOTCH_SPEED_KNOTS * 0.514444
            is_notched = abs(closure) < notch_limit
        VISUAL_RANGE = 5000.0
        is_visual = (dist < VISUAL_RANGE)
        is_radar_detect = ((dist < radar_range_m) and (ata_cos > min_cos_detect) and (not is_notched))
        is_radar_lock = (is_radar_detect and (dist < radar_range_m * 0.75) and (ata_cos > min_cos_lock))
        return (is_visual or is_radar_detect), is_radar_lock

    def _get_air_density(self, alt):
        return math.exp(-alt / self.cfg.SCALE_HEIGHT)

    def _update_plane_physics(self, ent, action, execute_discrete_actions=True):
        dt = self.cfg.PHYSICS_DT
        g = self.cfg.GRAVITY
        KNOTS_TO_MS = 0.514444
        MS_TO_KNOTS = 1.94384
        roll_rate = np.clip(action[0], -1, 1) * (math.pi / 2.0)
        g_norm = np.clip(action[1], -1, 1)
        if g_norm >= 0:
            target_g = 1.0 + (g_norm * (self.cfg.MAX_G - 1.0))
        else:
            MIN_NEG_G = -3.0
            target_g = 1.0 + (g_norm * (1.0 - MIN_NEG_G))
        throttle = (np.clip(action[2], -1, 1) + 1.0) / 2.0
        if execute_discrete_actions:
            if action[3] > 0.0: self._handle_weapons_system(ent)
            ent.cm_active = False
            if len(action) > 4 and action[4] > 0.5 and ent.chaff > 0:
                ent.cm_active = True
                if np.random.rand() < 0.1: ent.chaff -= 1
        ent.roll += roll_rate * dt
        ent.roll = (ent.roll + math.pi) % (2 * math.pi) - math.pi
        safe_speed = max(ent.speed, 10.0)
        max_aero_g = (safe_speed / 200.0) ** 2
        actual_g = min(target_g, max_aero_g)
        ent.g_load = actual_g
        q_factor = np.clip(ent.speed / 100.0, 0.0, 1.0)
        v_denom = max(ent.speed * KNOTS_TO_MS, 10.0)
        horizontal_g = actual_g * math.sin(ent.roll)
        turn_rate = (horizontal_g * g) / v_denom
        turn_rate *= q_factor
        ent.heading = (ent.heading + turn_rate * dt) % (2 * math.pi)
        vertical_g = actual_g * math.cos(ent.roll) - 1.0
        pitch_rate = (vertical_g * g) / v_denom
        pitch_rate *= q_factor
        ent.pitch += pitch_rate * dt
        ent.pitch = np.clip(ent.pitch, -1.4, 1.4)
        STALL_SPEED = 150.0
        stall_ratio = np.clip((ent.speed - 100.0) / 80.0, 0.0, 1.0)
        rho = self._get_air_density(ent.alt)
        v_ms = ent.speed * KNOTS_TO_MS
        drag_p = self.cfg.DRAG_PARASITIC_SL * rho * (v_ms ** 2)
        drag_i = self.cfg.DRAG_INDUCED_SL * rho * (actual_g ** 2)
        drag_stall = (1.0 - stall_ratio) * 50.0
        thrust = throttle * self.cfg.THRUST_WEIGHT * g * (rho ** 0.7)
        if ent.fuel <= 0:
            thrust = 0.0
        else:
            ent.fuel -= (throttle / self.cfg.MAX_FUEL_SEC) * dt
        gravity_drag = g * math.sin(ent.pitch)
        accel = thrust - (drag_p + drag_i + drag_stall) - gravity_drag
        ent.speed += (accel * MS_TO_KNOTS) * dt
        if ent.speed < STALL_SPEED:
            ent.pitch -= 0.5 * (1.0 - stall_ratio) * dt
        ent.speed = max(ent.speed, 0.0)
        v_ms = ent.speed * KNOTS_TO_MS
        v_horiz = v_ms * math.cos(ent.pitch)
        v_vert = v_ms * math.sin(ent.pitch)
        dist_h = v_horiz * dt
        ent.x += dist_h * math.cos(ent.heading)
        ent.y += dist_h * math.sin(ent.heading)
        ent.alt += v_vert * dt
        if ent.alt <= 0:
            self.events.append({"killer": -1, "victim": ent.uid, "type": "crash"})
            del self.entities[ent.uid]

    def _handle_weapons_system(self, ent):
        cannon_range = getattr(self.cfg, 'CANNON_RANGE_KM', 1.5) * 1000.0
        cannon_cos = math.cos(math.radians(getattr(self.cfg, 'CANNON_FOV_DEG', 10.0) / 2.0))
        for tid, t in self.entities.items():
            if t.team == ent.team or t.type != "plane": continue
            data = self.get_relative_data(ent.uid, tid)
            if data is None: continue
            dist, _, _, ata_cos, _, _ = data
            if dist < cannon_range and ata_cos > cannon_cos:
                self._fire_cannon(ent, t, dist)
                return
            if ent.ammo > 0 and dist > 500.0:
                vis, lock = self.get_sensor_state(ent.uid, tid)
                if lock:
                    active = sum(1 for m in self.entities.values() if m.type == 'missile' and m.owner_id == ent.uid)
                    if active < self.cfg.MAX_ACTIVE_MISSILES:
                        self._fire_missile(ent, t)
                        return

    def _fire_cannon(self, ent, target, dist):
        self.events.append({"killer": ent.uid, "victim": target.uid, "type": "kill"})
        if target.uid in self.entities: del self.entities[target.uid]

    def _fire_missile(self, ent, target):
        m_uid = self.spawn(ent.x, ent.y, ent.alt, ent.heading, ent.speed, ent.team, "missile")
        m_ent = self.entities[m_uid]
        m_ent.target_id = target.uid
        m_ent.owner_id = ent.uid
        m_ent.pitch = ent.pitch
        self.missile_registry[m_uid] = ent.uid
        ent.ammo -= 1
        self.events.append({"shooter": ent.uid, "target": target.uid, "type": "missile_fired"})

    def _calculate_ai_action(self, ent, kappa=0.0):
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
        if np.random.rand() < kappa:
            return [np.random.uniform(-1, 1), np.random.uniform(-0.5, 1), 0.5, 0, 0]
        des_roll = np.clip(h_err * 2.0, -1.4, 1.4)
        roll_cmd = np.clip((des_roll - ent.roll) * 2.0, -1.0, 1.0)
        bank_factor = 1.0 / max(0.2, math.cos(ent.roll))
        alt_err = 10000.0 - ent.alt
        g_cmd = np.clip(((bank_factor + np.clip(alt_err * 0.0005, -0.5, 1.0)) - 1.0) / 8.0, -0.2, 1.0)
        fire = 0.0
        if kappa < 0.5 and min_dist < 20000.0 and abs(h_err) < 0.2:
            if np.random.rand() < 0.05: fire = 1.0
        return [roll_cmd, g_cmd, 1.0, fire, 0.0]

    def _update_missile(self, ent):
        dt = self.cfg.PHYSICS_DT
        ent.time_alive += dt
        if ent.target_id not in self.entities:
            del self.entities[ent.uid];
            return
        t = self.entities[ent.target_id]
        if t.cm_active and np.random.rand() < self.cfg.CM_SPOOF_PROB:
            del self.entities[ent.uid];
            return
        dx, dy, dz = t.x - ent.x, t.y - ent.y, t.alt - ent.alt
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        des_pitch = math.asin(np.clip(dz / (dist + 1e-5), -1, 1))
        des_head = math.atan2(dy, dx)
        spd = ent.speed * 0.514444
        max_turn = (self.cfg.MISSILE_MAX_G * 9.81) / (spd + 1e-5)
        max_step = max_turn * dt
        h_diff = (des_head - ent.heading + math.pi) % (2 * math.pi) - math.pi
        ent.heading = (ent.heading + np.clip(h_diff, -max_step, max_step)) % (2 * math.pi)
        p_diff = des_pitch - ent.pitch
        ent.pitch += np.clip(p_diff, -max_step, max_step)
        thrust = self.cfg.MISSILE_BOOST_ACCEL if ent.time_alive < self.cfg.MISSILE_BOOST_SEC else 0.0
        drag = self.cfg.MISSILE_DRAG_PARASITIC * spd ** 2
        ent.speed += (thrust - drag) * 1.944 * dt
        if ent.speed < self.cfg.MISSILE_MIN_SPEED:
            del self.entities[ent.uid];
            return
        v_h = spd * math.cos(ent.pitch)
        ent.x += v_h * math.cos(ent.heading) * dt
        ent.y += v_h * math.sin(ent.heading) * dt
        ent.alt += spd * math.sin(ent.pitch) * dt

    def _resolve_collisions(self):
        ms = [e for e in self.entities.values() if e.type == "missile"]
        sq_lim = 200.0 ** 2
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
        ps = [e for e in self.entities.values() if e.type == "plane"]
        sq_lim = 30.0 ** 2
        for i, p1 in enumerate(ps):
            for p2 in ps[i + 1:]:
                ds = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.alt - p2.alt) ** 2
                if ds < sq_lim:
                    self.events.append({"type": "midair", "victim": p1.uid, "killer": p2.uid})
                    if p1.uid in self.entities: del self.entities[p1.uid]
                    if p2.uid in self.entities: del self.entities[p2.uid]
                    break