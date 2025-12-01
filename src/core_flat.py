# ================================================
# FILE: src/core_flat.py
# ================================================
import numpy as np
import math
from dataclasses import dataclass
from config import Config


# === CARTESIAN MATH HELPERS ===
def dist_2d(x1, y1, x2, y2):
    """Euclidean distance between two points in 2D plane."""
    return math.hypot(x2 - x1, y2 - y1)


def bearing_deg(x1, y1, x2, y2):
    """
    Calculate bearing from point 1 to point 2 in degrees.
    0 = North (+X), 90 = East (+Y).
    """
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0


@dataclass
class Entity:
    """
    Represents a single entity (aircraft or missile) in the simulation.
    """
    # Core Identification
    uid: int  # Unique identifier for this entity
    team: str  # Team affiliation ("blue" or "red")
    type: str  # Entity type ("plane" or "missile")

    # Position and Orientation (Cartesian coordinates)
    x: float  # North position in meters
    y: float  # East position in meters
    alt: float  # Altitude in meters above sea level
    heading: float  # Heading in degrees (0-360, 0=North, 90=East)
    speed: float  # Speed in knots

    # Physics State (Aircraft attitude and forces)
    roll: float = 0.0  # Roll angle in radians (-π to π, negative=left wing down)
    pitch: float = 0.0  # Pitch angle in radians (-1.4 to 1.4, positive=nose up)
    g_load: float = 1.0  # Current G-force being experienced (1.0 = level flight)

    # Logistics (Consumable resources)
    fuel: float = 1.0  # Fuel remaining as fraction (1.0 = 100% full tank)
    ammo: int = 4  # Number of missiles remaining
    chaff: int = 20  # Number of chaff countermeasures remaining
    cm_active: bool = False  # Whether countermeasures are currently being deployed

    # Missile-Specific Attributes (only used when type="missile")
    target_id: int = None  # UID of the target this missile is tracking
    time_alive: float = 0.0  # Time in seconds since missile launch
    owner_id: int = None  # NEW: Tracks who fired this missile (for active limits)


class AirCombatCore:
    """
    Core simulation engine that manages all entities, physics updates, and combat logic.
    Handles aircraft flight dynamics, missile guidance, sensor simulation, and collision detection.
    """

    def __init__(self):
        """
        Initialize the simulation core with empty state.
        Sets up entity tracking, event logging, and time management.
        """
        self.cfg = Config  # Reference to global configuration parameters
        self.entities = {}  # Dictionary mapping UID -> Entity for all active entities
        self.next_uid = 1  # Counter for generating unique entity IDs
        self.events = []  # List of events (crashes, kills, missile fires) that occurred this tick
        self.time = 0.0  # Simulation time in seconds since start

    def spawn(self, x, y, heading, speed, team, etype):
        """
        Create and spawn a new entity (aircraft or missile) in the simulation.
        """
        # Create new entity with basic parameters
        e = Entity(
            uid=self.next_uid, team=team, type=etype,
            x=x, y=y, alt=10000.0,  # Default spawn altitude at 10km
            heading=heading, speed=speed
        )

        # Initialize logistics based on entity type
        # Planes get full loadout, missiles get none
        e.ammo = self.cfg.MAX_MISSILES if etype == "plane" else 0
        e.chaff = self.cfg.MAX_CHAFF if etype == "plane" else 0
        e.fuel = 1.0  # Always spawn with full fuel tank

        # Add entity to tracking dictionary and increment UID counter
        self.entities[self.next_uid] = e
        self.next_uid += 1
        return e.uid

    def step(self, actions, kappa=0.0):
        """
        Advance simulation by one timestep (DT seconds) using physics sub-stepping.
        """
        # Clear previous frame's events
        self.events = []

        # === PHYSICS SUB-STEPPING LOOP ===
        # Run physics update internally with smaller timestep
        for substep in range(self.cfg.PHYSICS_SUBSTEPS):
            is_first_substep = (substep == 0)

            # Update all aircraft (player-controlled or AI-controlled)
            for uid, ent in list(self.entities.items()):  # Use list() to avoid dict modification during iteration
                if ent.type == "plane":
                    if uid in actions:
                        # Player/RL-agent controlled: apply provided action
                        self._update_plane_physics(ent, actions[uid], is_first_substep)
                    else:
                        # No action provided: generate AI action and apply it
                        ai_action = self._calculate_ai_action(ent, kappa)
                        self._update_plane_physics(ent, ai_action, is_first_substep)

            # Update all missiles (always autopilot)
            for uid, ent in list(self.entities.items()):
                if ent.type == "missile":
                    self._update_missile(ent)

            # Check for missile hits and remove destroyed entities
            # Run collision checks every sub-step for accurate detection
            self._resolve_collisions()

            # Check for midair collisions between aircraft
            self._check_midair_collisions()

        # Advance simulation time by full environment timestep
        self.time += self.cfg.DT

    def get_sensor_state(self, observer_uid, target_uid):
        """
        Simulate radar/sensor detection and lock capabilities.
        """
        # Get entity references
        obs = self.entities[observer_uid]
        tgt = self.entities[target_uid]

        # === VISIBILITY CHECKS (Detection) ===

        # Range Check: Target must be within radar maximum range
        dist = dist_2d(obs.x, obs.y, tgt.x, tgt.y)
        if dist > self.cfg.RADAR_RANGE_KM * 1000.0:
            return False, False  # Out of range - can't see or lock

        # Field of View Check: Target must be within radar cone angle
        bearing = bearing_deg(obs.x, obs.y, tgt.x, tgt.y)
        angle_off = abs((bearing - obs.heading + 180) % 360 - 180)  # Normalize to [-180, 180]
        if angle_off > self.cfg.RADAR_FOV_DEG:
            return False, False  # Outside FOV - can't see or lock

        # Doppler Notch Check: Target must be moving with sufficient radial velocity
        # Radar cannot detect targets moving perpendicular (beam aspect) due to Doppler filtering
        bearing_to_obs = (bearing + 180) % 360  # Reverse bearing (target's perspective)
        aspect_angle = abs((bearing_to_obs - tgt.heading + 180) % 360 - 180)  # Target's angle relative to observer
        radial_speed_tgt = tgt.speed * math.cos(
            math.radians(aspect_angle))  # Component of speed toward/away from observer
        if abs(radial_speed_tgt) < self.cfg.RADAR_NOTCH_SPEED_KNOTS:
            return False, False  # Target in notch filter - invisible to radar

        # Target is VISIBLE (passed all detection checks)
        is_visible = True

        # === LOCKING CHECKS (Tracking Quality) ===
        # Stricter constraints for weapons-grade tracking

        # Locking Range: Must be within 75% of max range for reliable track
        lock_max_range = (self.cfg.RADAR_RANGE_KM * 1000.0) * 0.75
        if dist > lock_max_range:
            return True, False  # Can see, but too far for solid lock

        # Locking FOV: Must be within 80% of FOV for centered track
        lock_max_angle = self.cfg.RADAR_FOV_DEG * 0.80
        if angle_off > lock_max_angle:
            return True, False  # Can see, but too far off-axis for good lock

        # All checks passed: target is both visible AND locked
        return True, True

    def _get_air_density(self, alt):
        """
        Calculate atmospheric density ratio at given altitude using exponential atmosphere model.
        """
        return math.exp(-alt / self.cfg.SCALE_HEIGHT)

    def _update_plane_physics(self, ent, action, execute_discrete_actions=True):
        """
        Update aircraft physics for one sub-timestep based on pilot/AI actions.
        """
        dt = self.cfg.PHYSICS_DT  # Use physics sub-timestep (0.04s)
        g = self.cfg.GRAVITY  # Gravitational acceleration (m/s²)

        # Unit Conversion Constants
        KNOTS_TO_MS = 0.514444
        MS_TO_KNOTS = 1.94384

        # === DECODE ACTIONS ===
        # Action indices: [0]=Roll Rate, [1]=G-Pull, [2]=Throttle, [3]=Fire, [4]=Countermeasures

        # Roll Rate: Convert normalized action [-1,1] to angular velocity (±90°/s max)
        roll_rate = np.clip(action[0], -1, 1) * math.radians(90.0)

        # G-Pull: Convert normalized action [-1,1] to target G-load
        g_norm = np.clip(action[1], -1, 1)
        target_g = 1.0 + (g_norm * (self.cfg.MAX_G - 1.0))  # Positive: 1.0 to MAX_G (pull)
        if g_norm < 0: target_g = 1.0 + (g_norm * 2.0)  # Negative: 1.0 to -1.0 (push)

        # Throttle: Convert normalized action [-1,1] to throttle setting [0,1]
        throttle = (np.clip(action[2], -1, 1) + 1.0) / 2.0

        # === DISCRETE ACTIONS (Execute only on first sub-step) ===
        if execute_discrete_actions:
            # Fire Weapon: Context sensitive (Cannon or Missile)
            if action[3] > 0.0:
                self._handle_weapons_system(ent)

            # Countermeasures: Activate chaff/flare dispensing
            ent.cm_active = False
            if len(action) > 4 and action[4] > 0.5 and ent.chaff > 0:
                ent.cm_active = True
                if np.random.rand() < 0.1: ent.chaff -= 1
        else:
            pass

        # === ATTITUDE DYNAMICS ===

        # Update Roll: Integrate roll rate over timestep
        ent.roll += roll_rate * dt
        # Wrap roll to [-π, π] range
        ent.roll = (ent.roll + math.pi) % (2 * math.pi) - math.pi

        # Aerodynamic G-Limit: Maximum achievable G depends on speed
        # Corner velocity concept: G capability = (V/200)²
        # FIX: Ensure we don't divide by zero
        safe_speed = max(ent.speed, 10.0)
        max_aero_g = (safe_speed / 200.0) ** 2
        actual_g = min(target_g, max_aero_g)  # Can't pull more G than airframe/speed allows
        ent.g_load = actual_g

        # === SMOOTH STALL PHYSICS ===
        # Compute stall ratio: 0.0 when fully stalled (100 kts), 1.0 when flying normally (≥150 kts)
        STALL_SPEED = 150.0  # Full stall below this speed
        STALL_ONSET = 180.0  # Stall begins at this speed (Raised slightly)
        stall_ratio = np.clip((ent.speed - 100.0) / (STALL_ONSET - 100.0), 0.0, 1.0)

        # Control Authority: Linear degradation with stall ratio
        control_authority = 0.2 + (0.8 * stall_ratio)

        # === TURN DYNAMICS ===

        # Horizontal Component: G-force in horizontal plane causes heading change (turn)
        horizontal_g = actual_g * math.sin(ent.roll)

        # Turn Rate = (a / v).
        turn_rate = ((horizontal_g * g) / (ent.speed * KNOTS_TO_MS + 1e-5)) * control_authority
        ent.heading = (ent.heading + math.degrees(turn_rate * dt)) % 360.0

        # Vertical Component: G-force in vertical plane causes pitch change
        vertical_g = actual_g * math.cos(ent.roll) - 1.0
        pitch_rate = ((vertical_g * g) / (ent.speed * KNOTS_TO_MS + 1e-5)) * control_authority

        ent.pitch += pitch_rate * dt
        ent.pitch = np.clip(ent.pitch, -1.4, 1.4)

        # === ATMOSPHERIC FORCES ===
        rho_ratio = self._get_air_density(ent.alt)
        speed_ms = ent.speed * KNOTS_TO_MS

        # Parasitic Drag: Scales with ρ×V²
        # FIX: Use Speed in m/s for drag calculation
        drag_p = self.cfg.DRAG_PARASITIC_SL * rho_ratio * (speed_ms ** 2)

        # Induced Drag: Scales with ρ×G²
        drag_i = self.cfg.DRAG_INDUCED_SL * rho_ratio * (actual_g ** 2)

        # FIX: Stall Drag reduced to 50.0 (from 1000.0) to allow recovery
        drag_stall = (1.0 - stall_ratio) * 50.0

        # Thrust: Turbofan engines lose thrust with altitude
        available_thrust = throttle * self.cfg.THRUST_WEIGHT * g * (rho_ratio ** 0.7)

        # Fuel Consumption
        if ent.fuel > 0:
            burn_rate = throttle / self.cfg.MAX_FUEL_SEC
            ent.fuel -= burn_rate * dt
        else:
            available_thrust = 0.0

        # Gravity Component (m/s^2 along flight path)
        gravity_force = g * math.sin(ent.pitch)

        # Net Acceleration (m/s^2)
        # Accel = (Thrust - Drags) - Gravity
        accel_ms = available_thrust - (drag_p + drag_i + drag_stall) - gravity_force

        # Convert back to Knots
        ent.speed = ent.speed + (accel_ms * MS_TO_KNOTS) * dt

        # Smooth stall nose drop: Force nose down if stalled to recover speed
        if ent.speed < STALL_SPEED:
            nose_drop_rate = 0.5 * (1.0 - stall_ratio)
            ent.pitch -= nose_drop_rate * dt

        ent.speed = max(ent.speed, 0.0)

        # === HORIZONTAL MOVEMENT (CARTESIAN) ===
        dist = (ent.speed * KNOTS_TO_MS) * dt
        dx = dist * math.cos(math.radians(ent.heading))
        dy = dist * math.sin(math.radians(ent.heading))
        ent.x += dx
        ent.y += dy

        # === VERTICAL MOVEMENT ===
        lift_factor = stall_ratio
        vertical_from_pitch = (ent.speed * KNOTS_TO_MS) * math.sin(ent.pitch) * lift_factor
        gravity_drop = -9.81 * (1.0 - lift_factor)

        ent.alt += vertical_from_pitch * dt + 0.5 * gravity_drop * (dt ** 2)

        # === GROUND COLLISION CHECK ===
        if ent.alt <= 0:
            self.events.append({"killer": -1, "victim": ent.uid, "type": "crash"})
            del self.entities[ent.uid]

    def _handle_weapons_system(self, ent):
        """
        Smart Weapon Selector:
        1. Iterates through targets to avoid "Tunnel Vision" on close-but-unshootable enemies.
        2. Enforces MAX_ACTIVE_MISSILES limit per agent.
        """
        # Find all enemy planes
        targets = [e for e in self.entities.values() if e.team != ent.team and e.type == "plane"]
        if not targets: return

        # Sort by distance (preferred)
        targets.sort(key=lambda t: dist_2d(ent.x, ent.y, t.x, t.y))

        # Get params safely
        cannon_range_km = getattr(self.cfg, 'CANNON_RANGE_KM', 1.5)
        cannon_fov_deg = getattr(self.cfg, 'CANNON_FOV_DEG', 10.0)

        # Iterate through targets until we find one we can shoot
        for target in targets:
            dist_m = dist_2d(ent.x, ent.y, target.x, target.y)

            # === CANNON LOGIC (Short Range) ===
            if dist_m < cannon_range_km * 1000.0:
                # Check Angle
                bearing = bearing_deg(ent.x, ent.y, target.x, target.y)
                angle_off = abs((bearing - ent.heading + 180) % 360 - 180)

                # Cone Check (Half width)
                if angle_off < (cannon_fov_deg / 2.0):
                    self._fire_cannon(ent, target, dist_m)
                    return  # Fired, action complete

            # === MISSILE LOGIC (Long Range) ===
            elif ent.ammo > 0:
                # Check lock
                visible, locking = self.get_sensor_state(ent.uid, target.uid)
                if locking:
                    # NEW: Active Missile Limit
                    active_missiles = sum(
                        1 for m in self.entities.values() if m.type == 'missile' and m.owner_id == ent.uid)

                    if active_missiles < self.cfg.MAX_ACTIVE_MISSILES:
                        self._fire_missile(ent, target)
                        return  # Fired, action complete

    def _fire_cannon(self, ent, target, dist):
        """
        Hit-scan logic for the cannon.
        """
        self.events.append({"killer": ent.uid, "victim": target.uid, "type": "kill"})
        if target.uid in self.entities:
            del self.entities[target.uid]

    def _fire_missile(self, ent, target):
        """
        Attempts to launch a missile at the specific target.
        """
        # Spawn missile
        m_uid = self.spawn(ent.x, ent.y, ent.heading, ent.speed, ent.team, "missile")
        self.entities[m_uid].target_id = target.uid
        self.entities[m_uid].owner_id = ent.uid  # Track owner for limits
        self.entities[m_uid].time_alive = 0.0
        ent.ammo -= 1
        self.events.append({"shooter": ent.uid, "target": target.uid, "type": "missile_fired"})

    def _calculate_ai_action(self, ent, kappa=0.0):
        """
        Generate AI opponent action using pursuit guidance with curriculum learning.
        """
        # === TARGET SELECTION ===
        targets = [e for e in self.entities.values() if e.team != ent.team and e.type == "plane"]
        if not targets: return [0.0, 0.0, 0.0, 0.0, 0.0]

        target = min(targets, key=lambda t: dist_2d(ent.x, ent.y, t.x, t.y))

        desired_heading = bearing_deg(ent.x, ent.y, target.x, target.y)
        heading_err = (desired_heading - ent.heading + 180) % 360 - 180
        dist_m = dist_2d(ent.x, ent.y, target.x, target.y)

        # === CURRICULUM LEARNING (Decision Noise) ===
        if np.random.rand() < kappa:
            return [np.random.uniform(-1, 1), np.random.uniform(-0.5, 1), np.random.uniform(0.5, 1), 0.0, 0.0]

        # === ROLL CONTROL (Bank toward target) ===
        desired_roll = np.clip(math.radians(heading_err * 2.0), -1.4, 1.4)
        roll_err = desired_roll - ent.roll
        roll_cmd = np.clip(roll_err * 2.0, -1.0, 1.0)

        # === G-PULL CONTROL (Turn and maintain altitude) ===
        desired_g = 1.0 / max(0.2, math.cos(ent.roll))
        target_alt = 10000.0
        alt_err = target_alt - ent.alt
        desired_g += np.clip(alt_err * 0.001, -0.5, 2.0)

        g_cmd = (desired_g - 1.0) / (self.cfg.MAX_G - 1.0)
        g_cmd = np.clip(g_cmd, -0.2, 1.0)

        # === CURRICULUM LEARNING (Execution Noise) ===
        roll_cmd += np.random.normal(0, kappa * 0.5)
        g_cmd += np.random.normal(0, kappa * 0.2)

        # === THROTTLE (Always maximum) ===
        throttle = 1.0

        # === WEAPON EMPLOYMENT ===
        fire = 0.0
        angle_off = abs(heading_err)

        if kappa < 0.5:
            # Basic AI firing logic
            if dist_m < self.cfg.RADAR_RANGE_KM * 1000.0 and angle_off < self.cfg.RADAR_FOV_DEG:
                # NEW: AI also respects missile limits
                active = sum(1 for m in self.entities.values() if m.type == 'missile' and m.owner_id == ent.uid)
                if active < self.cfg.MAX_ACTIVE_MISSILES:
                    if np.random.rand() < 0.05:
                        fire = 1.0

        # === COUNTERMEASURES ===
        cm = 0.0
        if np.random.rand() < 0.01:
            cm = 1.0

        return [np.clip(roll_cmd, -1, 1), np.clip(g_cmd, -0.2, 1), throttle, fire, cm]

    def _update_missile(self, ent):
        """
        Update missile physics and guidance for one sub-timestep.
        """
        dt = self.cfg.PHYSICS_DT
        ent.time_alive += dt
        KNOTS_TO_MS = 0.514444
        MS_TO_KNOTS = 1.94384

        if ent.target_id not in self.entities:
            del self.entities[ent.uid]
            return
        target = self.entities[ent.target_id]

        # === COUNTERMEASURES CHECK ===
        if target.cm_active:
            if np.random.rand() < self.cfg.CM_SPOOF_PROB:
                del self.entities[ent.uid]
                return

        # === MISSILE MOTOR MODEL ===
        g = self.cfg.GRAVITY
        thrust = 0.0
        if ent.time_alive < self.cfg.MISSILE_BOOST_SEC:
            thrust = self.cfg.MISSILE_BOOST_ACCEL

        # === DRAG FORCES ===
        # FIX: Use m/s for drag
        speed_ms = ent.speed * KNOTS_TO_MS
        drag_p = self.cfg.MISSILE_DRAG_PARASITIC * (speed_ms ** 2)

        # === PROPORTIONAL NAVIGATION GUIDANCE ===
        bearing = bearing_deg(ent.x, ent.y, target.x, target.y)
        diff = (bearing - ent.heading + 180) % 360 - 180

        req_turn_rate_rad = math.radians(diff / dt)
        req_accel = (speed_ms) * abs(req_turn_rate_rad)
        req_g = req_accel / g

        actual_g = min(req_g, self.cfg.MISSILE_MAX_G)

        valid_turn_rate_deg = math.degrees((actual_g * g) / (speed_ms + 1e-5))
        turn_step = valid_turn_rate_deg * dt

        if abs(diff) < turn_step:
            ent.heading = bearing
        else:
            ent.heading += math.copysign(turn_step, diff)
        ent.heading %= 360.0

        drag_i = self.cfg.MISSILE_DRAG_INDUCED * (actual_g ** 2)

        # === VELOCITY UPDATE ===
        # Scale back the drag scaler from 100.0 to 1.0 now that units are correct
        accel_ms = thrust - (drag_p + drag_i)
        ent.speed += (accel_ms * MS_TO_KNOTS) * dt

        if ent.speed < self.cfg.MISSILE_MIN_SPEED:
            del self.entities[ent.uid]
            return

        # === POSITION UPDATE ===
        dist = (ent.speed * KNOTS_TO_MS) * dt
        dx = dist * math.cos(math.radians(ent.heading))
        dy = dist * math.sin(math.radians(ent.heading))
        ent.x += dx
        ent.y += dy

    def _resolve_collisions(self):
        """
        Check for missile-target collisions and resolve hits.
        Reduced proximity fuse from 1000.0 to 200.0 to enforce better guidance.
        """
        missiles = [e for e in self.entities.values() if e.type == "missile"]

        for m in missiles:
            if m.target_id in self.entities:
                t = self.entities[m.target_id]
                dist = dist_2d(m.x, m.y, t.x, t.y)

                if dist < 200.0:
                    self.events.append({"killer": m.uid, "victim": t.uid, "type": "kill"})
                    if t.uid in self.entities: del self.entities[t.uid]
                    if m.uid in self.entities: del self.entities[m.uid]

    def _check_midair_collisions(self):
        """
        Check for plane-vs-plane collisions.
        Reduced threshold from 50.0 to 30.0.
        """
        planes = [e for e in self.entities.values() if e.type == "plane"]

        for i, p1 in enumerate(planes):
            for p2 in planes[i + 1:]:
                dist_horiz_m = dist_2d(p1.x, p1.y, p2.x, p2.y)
                alt_diff_m = abs(p1.alt - p2.alt)
                dist_3d_m = math.sqrt(dist_horiz_m ** 2 + alt_diff_m ** 2)

                if dist_3d_m < 30.0:
                    self.events.append({
                        "type": "midair_collision",
                        "victim": p1.uid,
                        "killer": p2.uid
                    })
                    if p1.uid in self.entities: del self.entities[p1.uid]
                    if p2.uid in self.entities: del self.entities[p2.uid]
                    break