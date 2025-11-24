import numpy as np
import math
from dataclasses import dataclass
from config import Config
from aircombat_sim.utils.geodesics import geodetic_direct, geodetic_distance_km, geodetic_bearing_deg


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

    # Physics
    roll: float = 0.0
    pitch: float = 0.0
    g_load: float = 1.0

    # Logistics
    fuel: float = 1.0  # 1.0 = 100%
    ammo: int = 4
    chaff: int = 20
    cm_active: bool = False  # Is dropping chaff right now?

    # Missile Specific
    target_id: int = None
    time_alive: float = 0.0


class AirCombatCore:
    def __init__(self):
        self.cfg = Config
        self.entities = {}
        self.next_uid = 1
        self.events = []
        self.time = 0.0

    def spawn(self, lat, lon, heading, speed, team, etype):
        e = Entity(
            uid=self.next_uid, team=team, type=etype,
            lat=lat, lon=lon, alt=10000.0,
            heading=heading, speed=speed
        )
        # Init Logistics
        e.ammo = self.cfg.MAX_MISSILES if etype == "plane" else 0
        e.chaff = self.cfg.MAX_CHAFF if etype == "plane" else 0
        e.fuel = 1.0  # Full tank

        self.entities[self.next_uid] = e
        self.next_uid += 1
        return e.uid

    def step(self, actions):
        self.events = []
        self.time += self.cfg.DT

        for uid, ent in list(self.entities.items()):
            if ent.type == "plane":
                if uid in actions:
                    self._update_plane_physics(ent, actions[uid])
                else:
                    ai_action = self._calculate_ai_action(ent)
                    self._update_plane_physics(ent, ai_action)

        for uid, ent in list(self.entities.items()):
            if ent.type == "missile":
                self._update_missile(ent)

        self._resolve_collisions()

    def get_sensor_state(self, observer_uid, target_uid):
        # ... (Same as Phase 4 - Doppler Logic) ...
        # Re-paste the exact Doppler logic from previous step here
        obs = self.entities[observer_uid]
        tgt = self.entities[target_uid]
        dist = geodetic_distance_km(obs.lat, obs.lon, tgt.lat, tgt.lon)
        if dist > self.cfg.RADAR_RANGE_KM: return False, False
        bearing = geodetic_bearing_deg(obs.lat, obs.lon, tgt.lat, tgt.lon)
        angle_off = abs((bearing - obs.heading + 180) % 360 - 180)
        if angle_off > self.cfg.RADAR_FOV_DEG: return False, False

        # Doppler Notch
        bearing_to_obs = (bearing + 180) % 360
        aspect_angle = abs((bearing_to_obs - tgt.heading + 180) % 360 - 180)
        radial_speed_tgt = tgt.speed * math.cos(math.radians(aspect_angle))
        if abs(radial_speed_tgt) < self.cfg.RADAR_NOTCH_SPEED_KNOTS:
            return False, False

        return True, True

    def _get_air_density(self, alt):
        # Exponential atmosphere model
        # rho_ratio = exp(-h / scale_height)
        return math.exp(-alt / self.cfg.SCALE_HEIGHT)

    def _update_plane_physics(self, ent, action):
        dt = self.cfg.DT
        g = self.cfg.GRAVITY

        # --- Decode Actions ---
        roll_rate = np.clip(action[0], -1, 1) * math.radians(90.0)

        g_norm = np.clip(action[1], -1, 1)
        target_g = 1.0 + (g_norm * (self.cfg.MAX_G - 1.0))
        if g_norm < 0: target_g = 1.0 + (g_norm * 2.0)

        throttle = (np.clip(action[2], -1, 1) + 1.0) / 2.0

        # Fire (Changed threshold from 0.5 to 0.0 for Normal distribution)
        if action[3] > 0.0 and ent.ammo > 0:
            self._try_fire(ent)

        # Countermeasures (Action 4)
        ent.cm_active = False
        if len(action) > 4 and action[4] > 0.5 and ent.chaff > 0:
            ent.cm_active = True
            # Burn 1 chaff per second (0.5 per tick)
            # Just randomize logic: 10% chance to decrement counter to simulate usage
            if np.random.rand() < 0.1: ent.chaff -= 1

        # --- Dynamics ---
        ent.roll += roll_rate * dt
        ent.roll = (ent.roll + math.pi) % (2 * math.pi) - math.pi

        max_aero_g = (ent.speed / 200.0) ** 2
        actual_g = min(target_g, max_aero_g)
        ent.g_load = actual_g

        # Turn
        horizontal_g = actual_g * math.sin(ent.roll)
        turn_rate = (horizontal_g * g) / (ent.speed * 0.5144 + 1e-5)
        ent.heading = (ent.heading + math.degrees(turn_rate * dt)) % 360.0

        vertical_g = actual_g * math.cos(ent.roll) - 1.0
        pitch_rate = (vertical_g * g) / (ent.speed * 0.5144 + 1e-5)
        ent.pitch += pitch_rate * dt
        ent.pitch = np.clip(ent.pitch, -1.4, 1.4)

        # --- ATMOSPHERIC FORCES ---
        rho_ratio = self._get_air_density(ent.alt)

        # Drag scales with density
        # Induced Drag increases at altitude due to lower lift efficiency?
        # Simplified: Drag forces scale with rho * v^2

        drag_p = self.cfg.DRAG_PARASITIC_SL * rho_ratio * (ent.speed ** 2)
        drag_i = self.cfg.DRAG_INDUCED_SL * rho_ratio * (actual_g ** 2)

        # Thrust Logic (Fuel Burn)
        # Turbofans lose thrust with altitude, approx prop to density
        available_thrust = throttle * self.cfg.THRUST_WEIGHT * g * (rho_ratio ** 0.7)

        # Fuel Burn
        if ent.fuel > 0:
            # Base burn + Afterburner
            # Normalize: Full AB empties tank in MAX_FUEL_SEC
            burn_rate = throttle / self.cfg.MAX_FUEL_SEC
            ent.fuel -= burn_rate * dt
        else:
            available_thrust = 0.0  # Glider mode

        gravity_force = g * math.sin(ent.pitch)

        accel = (available_thrust - (drag_p + drag_i) - gravity_force) * 1.94384
        ent.speed = ent.speed + accel * dt
        
        # --- STALL PHYSICS ---
        STALL_SPEED = 150.0 # Knots
        if ent.speed < STALL_SPEED:
            # STALL! Loss of lift.
            # 1. Nose drops (Pitch down)
            ent.pitch -= 1.0 * dt # Pitch down rate
            
            # 2. Loss of control (Dampen inputs)
            # (Handled implicitly as lift depends on speed)
            
            # 3. Gravity dominates (Falling)
            # If stalled, lift is drastically reduced. 
            # We simulate this by overriding the vertical movement logic below.
            pass
        
        # Hard floor at 0 speed (stopped)
        ent.speed = max(ent.speed, 0.0)

        dist = (ent.speed * 0.5144) * dt
        ent.lat, ent.lon = geodetic_direct(ent.lat, ent.lon, ent.heading, dist)
        
        # Vertical Movement
        # If stalled, we fall like a rock regardless of pitch
        if ent.speed < STALL_SPEED:
            # Fall speed approx gravity * time (simplified)
            # Just force a descent rate
            descent_rate = -50.0 # m/s
            ent.alt += descent_rate * dt
        else:
            ent.alt += (ent.speed * 0.5144) * math.sin(ent.pitch) * dt

        if ent.alt <= 0:
            self.events.append({"killer": -1, "victim": ent.uid, "type": "crash"})
            del self.entities[ent.uid]

    def _calculate_ai_action(self, ent):
        # Same PID as Phase 3, but add CM usage
        # ... (Copy PID logic) ...
        # 1. Find Target
        targets = [e for e in self.entities.values() if e.team != ent.team and e.type == "plane"]
        if not targets: return [0.0, 0.0, 0.0, 0.0, 0.0]
        target = min(targets, key=lambda t: geodetic_distance_km(ent.lat, ent.lon, t.lat, t.lon))
        desired_heading = geodetic_bearing_deg(ent.lat, ent.lon, target.lat, target.lon)
        heading_err = (desired_heading - ent.heading + 180) % 360 - 180
        dist_km = geodetic_distance_km(ent.lat, ent.lon, target.lat, target.lon)

        desired_roll = np.clip(math.radians(heading_err * 2.0), -1.4, 1.4)
        roll_err = desired_roll - ent.roll
        roll_cmd = np.clip(roll_err * 2.0, -1.0, 1.0)

        desired_g = 1.0 / max(0.2, math.cos(ent.roll))
        target_alt = 10000.0
        alt_err = target_alt - ent.alt
        desired_g += np.clip(alt_err * 0.001, -0.5, 2.0)
        g_cmd = (desired_g - 1.0) / (self.cfg.MAX_G - 1.0)
        g_cmd = np.clip(g_cmd, -0.2, 1.0)

        throttle = 1.0
        fire = 0.0
        angle_off = abs(heading_err)
        if dist_km < self.cfg.RADAR_RANGE_KM and angle_off < self.cfg.RADAR_FOV_DEG:
            if np.random.rand() < 0.05: fire = 1.0

        # AI Countermeasures?
        cm = 0.0
        # If locked (simple cheat check for now, or random)
        if np.random.rand() < 0.01: cm = 1.0

        return [roll_cmd, g_cmd, throttle, fire, cm]

    def _try_fire(self, ent):
        targets = [e for e in self.entities.values() if e.team != ent.team and e.type == "plane"]
        for t in targets:
            visible, locking = self.get_sensor_state(ent.uid, t.uid)
            if locking:
                m_uid = self.spawn(ent.lat, ent.lon, ent.heading, ent.speed, ent.team, "missile")
                self.entities[m_uid].target_id = t.uid
                self.entities[m_uid].time_alive = 0.0
                ent.ammo -= 1
                # Log missile fired event for reward tracking
                self.events.append({"shooter": ent.uid, "target": t.uid, "type": "missile_fired"})
                break

    def _update_missile(self, ent):
        dt = self.cfg.DT
        ent.time_alive += dt

        if ent.target_id not in self.entities:
            del self.entities[ent.uid]
            return
        target = self.entities[ent.target_id]

        # --- Countermeasures Check ---
        if target.cm_active:
            if np.random.rand() < self.cfg.CM_SPOOF_PROB:
                # Spoofed! Missile loses lock
                del self.entities[ent.uid]
                return

        # ... (Standard Missile Physics from Phase 3) ...
        g = self.cfg.GRAVITY
        thrust = 0.0
        if ent.time_alive < self.cfg.MISSILE_BOOST_SEC:
            thrust = self.cfg.MISSILE_BOOST_ACCEL
        drag_p = self.cfg.MISSILE_DRAG_PARASITIC * (ent.speed ** 2)
        bearing = geodetic_bearing_deg(ent.lat, ent.lon, target.lat, target.lon)
        diff = (bearing - ent.heading + 180) % 360 - 180
        req_turn_rate_rad = math.radians(diff / dt)
        req_accel = (ent.speed * 0.5144) * abs(req_turn_rate_rad)
        req_g = req_accel / g
        actual_g = min(req_g, self.cfg.MISSILE_MAX_G)
        valid_turn_rate_deg = math.degrees((actual_g * g) / (ent.speed * 0.5144 + 1e-5))
        turn_step = valid_turn_rate_deg * dt
        if abs(diff) < turn_step:
            ent.heading = bearing
        else:
            ent.heading += math.copysign(turn_step, diff)
        ent.heading %= 360.0
        drag_i = self.cfg.MISSILE_DRAG_INDUCED * (actual_g ** 2)
        ent.speed += (thrust - (drag_p + drag_i) * 100.0) * dt
        if ent.speed < self.cfg.MISSILE_MIN_SPEED:
            del self.entities[ent.uid];
            return
        dist = (ent.speed * 0.5144) * dt
        ent.lat, ent.lon = geodetic_direct(ent.lat, ent.lon, ent.heading, dist)

    def _resolve_collisions(self):
        missiles = [e for e in self.entities.values() if e.type == "missile"]
        for m in missiles:
            if m.target_id in self.entities:
                t = self.entities[m.target_id]
                dist = geodetic_distance_km(m.lat, m.lon, t.lat, t.lon)
                if dist < 1.0:
                    self.events.append({"killer": m.uid, "victim": t.uid, "type": "kill"})
                    if t.uid in self.entities: del self.entities[t.uid]
                    if m.uid in self.entities: del self.entities[m.uid]