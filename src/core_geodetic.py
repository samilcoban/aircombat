# ================================================
# FILE: src/core_geodetic.py
# ================================================
import numpy as np
import math
from dataclasses import dataclass
from config import Config
from src.utils.geodesics import geodetic_direct, geodetic_distance_km, geodetic_bearing_deg


def angle_between_vectors_degrees(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    dot = np.dot(v1, v2)
    cos_angle = np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def lla_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    R_earth = 6371000.0
    d_lat = math.radians(lat - ref_lat)
    d_lon = math.radians(lon - ref_lon)
    lat_avg = math.radians((lat + ref_lat) / 2.0)
    x = R_earth * d_lon * math.cos(lat_avg)
    y = R_earth * d_lat
    z = alt - ref_alt
    return np.array([x, y, z])


@dataclass
class Entity:
    uid: int
    team: str
    type: str
    lat: float
    lon: float
    alt: float
    heading: float
    speed: float
    roll: float = 0.0
    pitch: float = 0.0
    g_load: float = 1.0
    fuel: float = 1.0
    ammo: int = 4
    chaff: int = 20
    cm_active: bool = False
    target_id: int = None
    time_alive: float = 0.0
    owner_id: int = None


class AirCombatCore:
    def __init__(self):
        self.cfg = Config
        self.entities = {}
        self.next_uid = 1
        self.events = []
        self.time = 0.0

        # --- NEW: MISSILE REGISTRY ---
        self.missile_registry = {}

    def spawn(self, lat, lon, heading, speed, team, etype):
        e = Entity(
            uid=self.next_uid, team=team, type=etype,
            lat=lat, lon=lon, alt=10000.0,
            heading=heading, speed=speed
        )
        e.ammo = self.cfg.MAX_MISSILES if etype == "plane" else 0
        e.chaff = self.cfg.MAX_CHAFF if etype == "plane" else 0
        self.entities[self.next_uid] = e
        self.next_uid += 1
        return e.uid

    def step(self, actions, kappa=0.0):
        self.events = []
        for substep in range(self.cfg.PHYSICS_SUBSTEPS):
            is_first = (substep == 0)

            for uid, ent in list(self.entities.items()):
                if ent.type == "plane":
                    if uid in actions:
                        self._update_plane(ent, actions[uid], is_first)
                    else:
                        self._update_plane(ent, self._ai_action(ent, kappa), is_first)

            for uid, ent in list(self.entities.items()):
                if ent.type == "missile": self._update_missile(ent)

            self._resolve_collisions()
            self._check_midair_collisions()

        self.time += self.cfg.DT

    def get_sensor_state(self, observer_uid, target_uid):
        obs = self.entities[observer_uid]
        tgt = self.entities[target_uid]

        rel_pos = lla_to_enu(tgt.lat, tgt.lon, tgt.alt, obs.lat, obs.lon, obs.alt)
        dist_3d = np.linalg.norm(rel_pos)

        h_rad = math.radians(obs.heading)
        p_rad = obs.pitch

        obs_boresight = np.array([
            math.cos(p_rad) * math.sin(h_rad),  # East
            math.cos(p_rad) * math.cos(h_rad),  # North
            math.sin(p_rad)  # Up
        ])

        angle_off = angle_between_vectors_degrees(obs_boresight, rel_pos)
        is_notched = False

        VISUAL_RANGE = 5000.0
        is_visual = (dist_3d < VISUAL_RANGE)

        is_radar_detect = (
                (dist_3d < self.cfg.RADAR_RANGE_KM * 1000.0) and
                (angle_off < self.cfg.RADAR_FOV_DEG) and
                (not is_notched)
        )

        is_radar_lock = (
                is_radar_detect and
                (dist_3d < (self.cfg.RADAR_RANGE_KM * 1000.0) * 0.75) and
                (angle_off < self.cfg.RADAR_FOV_DEG * 0.80)
        )

        return (is_visual or is_radar_detect), is_radar_lock

    def _get_air_density(self, alt):
        return math.exp(-alt / self.cfg.SCALE_HEIGHT)

    def _update_plane(self, ent, action, discrete=True):
        dt = self.cfg.PHYSICS_DT
        g = self.cfg.GRAVITY

        # Inputs
        roll_rate = np.clip(action[0], -1, 1) * math.radians(90.0)
        g_cmd = np.clip(action[1], -1, 1)

        # --- MODIFIED: NEGATIVE Gs ---
        if g_cmd >= 0:
            target_g = 1.0 + g_cmd * (self.cfg.MAX_G - 1.0)
        else:
            MIN_NEG_G = -3.0
            target_g = 1.0 + (g_cmd * (1.0 - MIN_NEG_G))
        # -----------------------------

        throttle = (action[2] + 1.0) / 2.0

        if discrete:
            if action[3] > 0.0: self._fire_logic(ent)
            ent.cm_active = (len(action) > 4 and action[4] > 0.5)
            if ent.cm_active and ent.chaff > 0 and np.random.rand() < 0.1: ent.chaff -= 1

        ent.roll = (ent.roll + roll_rate * dt + math.pi) % (2 * math.pi) - math.pi

        max_aero_g = (max(ent.speed, 10) / 200.0) ** 2
        ent.g_load = min(target_g, max_aero_g)

        stall_ratio = np.clip((ent.speed - 100) / 80.0, 0.0, 1.0)

        horizontal_g = ent.g_load * math.sin(ent.roll)
        turn_rate = (horizontal_g * g) / (ent.speed * 0.514 + 1e-5) * (0.2 + 0.8 * stall_ratio)
        ent.heading = (ent.heading + math.degrees(turn_rate * dt)) % 360.0

        vertical_g = ent.g_load * math.cos(ent.roll) - 1.0
        pitch_rate = (vertical_g * g) / (ent.speed * 0.514 + 1e-5) * (0.2 + 0.8 * stall_ratio)
        ent.pitch = np.clip(ent.pitch + pitch_rate * dt, -1.4, 1.4)

        rho = self._get_air_density(ent.alt)
        v_ms = ent.speed * 0.514
        drag = (0.0002 * rho * v_ms ** 2) + (0.1 * rho * ent.g_load ** 2) + ((1 - stall_ratio) * 50)
        thrust = throttle * 1.5 * g * (rho ** 0.7) if ent.fuel > 0 else 0
        gravity = g * math.sin(ent.pitch)

        accel = thrust - drag - gravity
        ent.speed = max(0, ent.speed + (accel * 1.944) * dt)
        if ent.fuel > 0: ent.fuel -= (throttle / 300.0) * dt

        if ent.speed < 150: ent.pitch -= 0.5 * (1 - stall_ratio) * dt

        dist = ent.speed * 0.514 * dt
        ent.lat, ent.lon = geodetic_direct(ent.lat, ent.lon, ent.heading, dist)

        lift = stall_ratio
        vert_spd = v_ms * math.sin(ent.pitch) * lift
        grav_drop = -9.81 * (1 - lift)
        ent.alt += vert_spd * dt + 0.5 * grav_drop * dt ** 2

        if ent.alt <= 0:
            self.events.append({"killer": -1, "victim": ent.uid, "type": "crash"})
            del self.entities[ent.uid]

    def _fire_logic(self, ent):
        targets = [e for e in self.entities.values() if e.team != ent.team and e.type == "plane"]
        if not targets: return

        for t in targets:
            vis, lock = self.get_sensor_state(ent.uid, t.uid)
            if lock and ent.ammo > 0:
                active = sum(1 for m in self.entities.values() if m.type == 'missile' and m.owner_id == ent.uid)
                if active < self.cfg.MAX_ACTIVE_MISSILES:
                    self._spawn_missile(ent, t)
                    return

    def _spawn_missile(self, ent, target):
        m = self.spawn(ent.lat, ent.lon, ent.heading, ent.speed, ent.team, "missile")

        # --- MODIFIED: REGISTER OWNER ---
        self.entities[m].target_id = target.uid
        self.entities[m].owner_id = ent.uid
        self.entities[m].pitch = ent.pitch
        self.missile_registry[m] = ent.uid
        # --------------------------------

        ent.ammo -= 1
        self.events.append({"shooter": ent.uid, "target": target.uid, "type": "missile_fired"})

    def _update_missile(self, ent):
        dt = self.cfg.PHYSICS_DT
        ent.time_alive += dt
        if ent.target_id not in self.entities:
            del self.entities[ent.uid];
            return

        tgt = self.entities[ent.target_id]
        if tgt.cm_active and np.random.rand() < 0.1:
            del self.entities[ent.uid];
            return

        bearing = geodetic_bearing_deg(ent.lat, ent.lon, tgt.lat, tgt.lon)
        diff = (bearing - ent.heading + 180) % 360 - 180
        turn = np.clip(diff, -5.0, 5.0)
        ent.heading = (ent.heading + turn) % 360

        accel = 500.0 if ent.time_alive < 6.0 else -10.0
        ent.speed += accel * 1.944 * dt

        if ent.speed < 200:
            del self.entities[ent.uid];
            return

        dist = ent.speed * 0.514 * dt
        ent.lat, ent.lon = geodetic_direct(ent.lat, ent.lon, ent.heading, dist)

    def _resolve_collisions(self):
        missiles = [e for e in self.entities.values() if e.type == "missile"]
        for m in missiles:
            if m.target_id in self.entities:
                t = self.entities[m.target_id]
                rel = lla_to_enu(t.lat, t.lon, t.alt, m.lat, m.lon, m.alt)
                if np.linalg.norm(rel) < 200.0:
                    # --- MODIFIED: OWNER LOOKUP ---
                    owner_id = self.missile_registry.get(m.uid, -1)
                    self.events.append({
                        "killer": m.uid,
                        "victim": t.uid,
                        "type": "kill",
                        "owner_id": owner_id
                    })
                    # ------------------------------
                    del self.entities[t.uid]
                    del self.entities[m.uid]

    def _check_midair_collisions(self):
        planes = [e for e in self.entities.values() if e.type == "plane"]
        for i, p1 in enumerate(planes):
            for p2 in planes[i + 1:]:
                rel = lla_to_enu(p2.lat, p2.lon, p2.alt, p1.lat, p1.lon, p1.alt)
                if np.linalg.norm(rel) < 30.0:
                    self.events.append({"type": "midair", "victim": p1.uid, "killer": p2.uid})
                    del self.entities[p1.uid]
                    del self.entities[p2.uid]
                    break

    def _ai_action(self, ent, kappa):
        return [0.0, 0.0, 1.0, 0.0, 0.0]