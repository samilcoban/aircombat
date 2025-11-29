# Import required libraries for numerical operations, trigonometry, and data structures
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
    Uses dataclass for automatic initialization and representation.
    """
    # Core Identification
    uid: int          # Unique identifier for this entity
    team: str         # Team affiliation ("blue" or "red")
    type: str         # Entity type ("plane" or "missile")
    
    # Position and Orientation (Cartesian coordinates)
    x: float          # North position in meters
    y: float          # East position in meters
    alt: float        # Altitude in meters above sea level
    heading: float    # Heading in degrees (0-360, 0=North, 90=East)
    speed: float      # Speed in knots
    
    # Physics State (Aircraft attitude and forces)
    roll: float = 0.0      # Roll angle in radians (-π to π, negative=left wing down)
    pitch: float = 0.0     # Pitch angle in radians (-1.4 to 1.4, positive=nose up)
    g_load: float = 1.0    # Current G-force being experienced (1.0 = level flight)
    
    # Logistics (Consumable resources)
    fuel: float = 1.0      # Fuel remaining as fraction (1.0 = 100% full tank)
    ammo: int = 4          # Number of missiles remaining
    chaff: int = 20        # Number of chaff countermeasures remaining
    cm_active: bool = False  # Whether countermeasures are currently being deployed
    
    # Missile-Specific Attributes (only used when type="missile")
    target_id: int = None      # UID of the target this missile is tracking
    time_alive: float = 0.0    # Time in seconds since missile launch


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
        self.cfg = Config                # Reference to global configuration parameters
        self.entities = {}               # Dictionary mapping UID -> Entity for all active entities
        self.next_uid = 1                # Counter for generating unique entity IDs
        self.events = []                 # List of events (crashes, kills, missile fires) that occurred this tick
        self.time = 0.0                  # Simulation time in seconds since start

    def spawn(self, x, y, heading, speed, team, etype):
        """
        Create and spawn a new entity (aircraft or missile) in the simulation.
        
        Args:
            x: Initial North position in meters
            y: Initial East position in meters
            heading: Initial heading in degrees
            speed: Initial speed in knots
            team: Team affiliation ("blue" or "red")
            etype: Entity type ("plane" or "missile")
            
        Returns:
            int: The unique ID assigned to the spawned entity
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
        
        The environment step is DT=0.2s, but internally we run PHYSICS_SUBSTEPS=20
        iterations with dt=0.01s each. This prevents missiles from teleporting through
        targets and improves collision detection accuracy.
        
        Discrete actions (fire, countermeasures) execute only on the FIRST sub-step,
        while continuous controls (roll, g, throttle) apply every sub-step.
        
        Args:
            actions: Dictionary mapping entity UID -> action array [roll, g, throttle, fire, cm]
            kappa: Curriculum learning parameter (0.0-1.0). Higher values make AI opponents weaker/noisier.
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
        # (after all sub-steps complete)
        self.time += self.cfg.DT

    def get_sensor_state(self, observer_uid, target_uid):
        """
        Simulate radar/sensor detection and lock capabilities.
        
        Returns TWO distinct states:
        1. VISIBILITY (Detection): Can the radar see the target?
        2. LOCKING (Tracking): Can we achieve a weapons-quality lock?
        
        Checks based on:
        - Range (must be within radar range)
        - Field of View (target must be in radar cone)
        - Doppler Notch (target must have sufficient radial velocity to be detected)
        
        Locking requires STRICTER constraints than visibility:
        - Distance < 75% of max radar range (tighter geofencing for missile accuracy)
        - Angle < 80% of FOV (requires target more centered in radar cone)
        
        Args:
            observer_uid: UID of the observing entity
            target_uid: UID of the target entity
            
        Returns:
            tuple: (visible: bool, locking: bool)
                visible: Whether target can be detected by radar
                locking: Whether target can be locked for missile firing (stricter)
        """
        # Get entity references
        obs = self.entities[observer_uid]
        tgt = self.entities[target_uid]
        
        # === VISIBILITY CHECKS (Detection) ===
        
        # Range Check: Target must be within radar maximum range
        # Convert KM to Meters for comparison
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
        radial_speed_tgt = tgt.speed * math.cos(math.radians(aspect_angle))  # Component of speed toward/away from observer
        if abs(radial_speed_tgt) < self.cfg.RADAR_NOTCH_SPEED_KNOTS:
            return False, False  # Target in notch filter - invisible to radar
        
        # Target is VISIBLE (passed all detection checks)
        is_visible = True
        
        # === LOCKING CHECKS (Tracking Quality) ===
        # Stricter constraints for weapons-grade tracking
        
        # Locking Range: Must be within 75% of max range for reliable track
        # Rationale: Radar accuracy degrades at extreme range
        lock_max_range = (self.cfg.RADAR_RANGE_KM * 1000.0) * 0.75
        if dist > lock_max_range:
            return True, False  # Can see, but too far for solid lock
        
        # Locking FOV: Must be within 80% of FOV for centered track
        # Rationale: Target near edge of scan volume has poor track quality
        lock_max_angle = self.cfg.RADAR_FOV_DEG * 0.80
        if angle_off > lock_max_angle:
            return True, False  # Can see, but too far off-axis for good lock
        
        # All checks passed: target is both visible AND locked
        return True, True

    def _get_air_density(self, alt):
        """
        Calculate atmospheric density ratio at given altitude using exponential atmosphere model.
        Used for realistic drag and thrust calculations that vary with altitude.
        
        Args:
            alt: Altitude in meters
            
        Returns:
            float: Density ratio (1.0 at sea level, decreases exponentially with altitude)
        """
        # Exponential atmosphere model: ρ/ρ₀ = exp(-h/H)
        # where H is scale height (~7400m for Earth's atmosphere)
        return math.exp(-alt / self.cfg.SCALE_HEIGHT)

    def _update_plane_physics(self, ent, action, execute_discrete_actions=True):
        """
        Update aircraft physics for one sub-timestep based on pilot/AI actions.
        
        Implements 6-DOF flight dynamics including:
        - Roll/pitch/yaw control
        - Thrust and drag (altitude-dependent)
        - G-force limitations
        - Fuel consumption
        - Stall physics
        - Weapon/countermeasure deployment
        
        Args:
            ent: Entity object representing the aircraft
            action: Action array [roll_cmd, g_cmd, throttle, fire, cm]
            execute_discrete_actions: If True, execute discrete actions (fire, CM).
                                    Set to False for sub-steps after the first.
        """
        dt = self.cfg.PHYSICS_DT  # Use physics sub-timestep (0.01s)
        g = self.cfg.GRAVITY      # Gravitational acceleration (m/s²)

        # === DECODE ACTIONS ===
        # Action indices: [0]=Roll Rate, [1]=G-Pull, [2]=Throttle, [3]=Fire, [4]=Countermeasures
        
        # Roll Rate: Convert normalized action [-1,1] to angular velocity (±90°/s max)
        roll_rate = np.clip(action[0], -1, 1) * math.radians(45.0)

        # G-Pull: Convert normalized action [-1,1] to target G-load
        g_norm = np.clip(action[1], -1, 1)
        target_g = 1.0 + (g_norm * (self.cfg.MAX_G - 1.0))  # Positive: 1.0 to MAX_G (pull)
        if g_norm < 0: target_g = 1.0 + (g_norm * 2.0)      # Negative: 1.0 to -1.0 (push)

        # Throttle: Convert normalized action [-1,1] to throttle setting [0,1]
        # Agent must learn proper throttle control for effective training
        throttle = (np.clip(action[2], -1, 1) + 1.0) / 2.0

        # === DISCRETE ACTIONS (Execute only on first sub-step) ===
        if execute_discrete_actions:
            # Fire Weapon: Trigger threshold changed from 0.5 to 0.0 for compatibility with Normal distribution outputs
            if action[3] > 0.0 and ent.ammo > 0:
                self._try_fire(ent)

            # Countermeasures: Activate chaff/flare dispensing
            ent.cm_active = False
            if len(action) > 4 and action[4] > 0.5 and ent.chaff > 0:
                ent.cm_active = True
                # Simulate chaff consumption: 10% chance per tick to decrement counter
                # (Approximates burning 1 chaff per second at 2Hz update rate)
                if np.random.rand() < 0.1: ent.chaff -= 1
        else:
            # On subsequent sub-steps, maintain CM state but don't re-execute
            # (CM deployment persists across sub-steps within one environment step)
            pass

        # === ATTITUDE DYNAMICS ===
        
        # Update Roll: Integrate roll rate over timestep
        ent.roll += roll_rate * dt
        # Wrap roll to [-π, π] range
        ent.roll = (ent.roll + math.pi) % (2 * math.pi) - math.pi

        # Aerodynamic G-Limit: Maximum achievable G depends on speed
        # Corner velocity concept: G capability = (V/200)²
        max_aero_g = (ent.speed / 200.0) ** 2
        actual_g = min(target_g, max_aero_g)  # Can't pull more G than airframe/speed allows
        ent.g_load = actual_g

        # === TURN DYNAMICS ===
        
        # Horizontal Component: G-force in horizontal plane causes heading change (turn)
        horizontal_g = actual_g * math.sin(ent.roll)  # Roll angle determines horizontal G component
        turn_rate = (horizontal_g * g) / (ent.speed * 0.5144 + 1e-5)  # Turn rate = (G×g)/V, convert knots to m/s
        ent.heading = (ent.heading + math.degrees(turn_rate * dt)) % 360.0  # Update heading and wrap to [0,360]

        # Vertical Component: G-force in vertical plane causes pitch change (climb/dive)
        vertical_g = actual_g * math.cos(ent.roll) - 1.0  # Subtract 1.0 for gravity offset
        pitch_rate = (vertical_g * g) / (ent.speed * 0.5144 + 1e-5)  # Pitch rate calculation
        ent.pitch += pitch_rate * dt
        ent.pitch = np.clip(ent.pitch, -1.4, 1.4)  # Limit pitch to ±80° to prevent unrealistic vertical flight

        # === ATMOSPHERIC FORCES ===
        
        # Get density ratio at current altitude
        rho_ratio = self._get_air_density(ent.alt)

        # Parasitic Drag: Scales with ρ×V² (form drag, skin friction)
        drag_p = self.cfg.DRAG_PARASITIC_SL * rho_ratio * (ent.speed ** 2)
        
        # Induced Drag: Scales with ρ×G² (drag due to lift production)
        drag_i = self.cfg.DRAG_INDUCED_SL * rho_ratio * (actual_g ** 2)
        
        # === SMOOTH STALL PHYSICS ===
        # Compute stall ratio: 0.0 when fully stalled (100 kts), 1.0 when flying normally (≥150 kts)
        # This provides a smooth gradient instead of a discontinuous cliff
        STALL_SPEED = 150.0  # Full stall below this speed
        STALL_ONSET = 100.0  # Stall begins at this speed
        stall_ratio = np.clip((ent.speed - STALL_ONSET) / (STALL_SPEED - STALL_ONSET), 0.0, 1.0)
        
        # Stall Drag: Massive drag increase at low speeds (simulates separation drag)
        # REDUCED from 20000.0 to 2000.0 for PPO training stability
        # The original 20000 created a "gradient cliff" where slowing to 149 kts = instant death
        # PPO cannot learn from discontinuous penalties. 2000 is still severe but provides
        # smooth gradients that teach: "Slowing down = more drag = bad"
        drag_stall = (1.0 - stall_ratio) * 2000.0  # CHANGED: Was 20000.0

        # Thrust: Turbofan engines lose thrust with altitude
        # Thrust ∝ ρ^0.7 (empirical approximation for turbofans)
        available_thrust = throttle * self.cfg.THRUST_WEIGHT * g * (rho_ratio ** 0.7)

        # Fuel Consumption
        if ent.fuel > 0:
            # Fuel burn rate proportional to throttle setting
            # Normalized so full afterburner empties tank in MAX_FUEL_SEC seconds
            burn_rate = throttle / self.cfg.MAX_FUEL_SEC
            ent.fuel -= burn_rate * dt
        else:
            available_thrust = 0.0  # Out of fuel: engine flameout, become a glider
        # Gravity Component: Weight force along flight path (positive = climbing against gravity)
        gravity_force = g * math.sin(ent.pitch)

        # Net Acceleration: (Thrust - Drag - Gravity_component) × unit_conversion
        # Now includes stall drag for smooth physics
        accel = (available_thrust - (drag_p + drag_i + drag_stall) - gravity_force) * 1.94384
        ent.speed = ent.speed + accel * dt
        
        # Smooth stall nose drop: Pitch decreases progressively as speed drops
        # This simulates loss of lift on tail surfaces (pitch authority degradation)
        if ent.speed < STALL_SPEED:
            nose_drop_rate = 1.0 * (1.0 - stall_ratio)  # Stronger effect when more stalled
            ent.pitch -= nose_drop_rate * dt
        
        # Prevent negative speeds (aircraft cannot fly backwards)
        ent.speed = max(ent.speed, 0.0)

        # === HORIZONTAL MOVEMENT (CARTESIAN) ===
        # Calculate distance traveled this timestep
        dist = (ent.speed * 0.5144) * dt  # Convert knots to m/s, multiply by time
        
        # Update position using Cartesian trigonometry
        # Heading 0 = North (+X), 90 = East (+Y)
        dx = dist * math.cos(math.radians(ent.heading))
        dy = dist * math.sin(math.radians(ent.heading))
        ent.x += dx
        ent.y += dy
        
        # === VERTICAL MOVEMENT (with smooth lift transition) ===
        # Lift factor: Wings produce full lift at normal speeds, zero lift when stalled
        lift_factor = stall_ratio  # 0.0 = no lift, 1.0 = full lift
        
        # Vertical velocity from pitch (modulated by lift)
        vertical_from_pitch = (ent.speed * 0.5144) * math.sin(ent.pitch) * lift_factor
        
        # Gravity component: When lift is lost, gravity dominates
        # This creates smooth descent that increases as stall deepens
        gravity_drop = -9.81 * (1.0 - lift_factor)  # m/s² downward when stalled
        
        # Combined vertical movement
        ent.alt += vertical_from_pitch * dt + 0.5 * gravity_drop * (dt ** 2)

        # === GROUND COLLISION CHECK ===
        if ent.alt <= 0:
            # Aircraft hit the ground: record crash event and remove from simulation
            self.events.append({"killer": -1, "victim": ent.uid, "type": "crash"})  # killer=-1 indicates terrain
            del self.entities[ent.uid]

    def _calculate_ai_action(self, ent, kappa=0.0):
        """
        Generate AI opponent action using pursuit guidance with curriculum learning.
        Implements a simple but effective AI that:
        - Points toward nearest enemy (pure pursuit)
        - Maintains altitude
        - Fires when in firing position
        - Uses curriculum parameter κ to adjust difficulty
        
        Args:
            ent: The AI-controlled entity
            kappa: Curriculum parameter (0.0=expert, 1.0=random novice)
            
        Returns:
            list: Action array [roll_cmd, g_cmd, throttle, fire, cm]
        """
        # === TARGET SELECTION ===
        # Find all enemy aircraft
        targets = [e for e in self.entities.values() if e.team != ent.team and e.type == "plane"]
        if not targets: return [0.0, 0.0, 0.0, 0.0, 0.0]  # No targets: return neutral action
        
        # Select nearest enemy as target
        target = min(targets, key=lambda t: dist_2d(ent.x, ent.y, t.x, t.y))
        
        # Calculate bearing and distance to target
        desired_heading = bearing_deg(ent.x, ent.y, target.x, target.y)
        heading_err = (desired_heading - ent.heading + 180) % 360 - 180  # Normalize to [-180, 180]
        dist_m = dist_2d(ent.x, ent.y, target.x, target.y)

        # === CURRICULUM LEARNING (Decision Noise) ===
        # With probability κ, AI makes completely random decisions ("drunk pilot" mode)
        if np.random.rand() < kappa:
            # Random flailing - used to make opponent easier during early training
            roll_cmd = np.random.uniform(-1.0, 1.0)
            g_cmd = np.random.uniform(-0.5, 1.0)
            throttle = np.random.uniform(0.5, 1.0)
            fire = 0.0
            cm = 0.0
            return [roll_cmd, g_cmd, throttle, fire, cm]

        # === ROLL CONTROL (Bank toward target) ===
        # Calculate desired roll angle to turn toward target
        desired_roll = np.clip(math.radians(heading_err * 2.0), -1.4, 1.4)  # Proportional control with limits
        roll_err = desired_roll - ent.roll
        roll_cmd = np.clip(roll_err * 2.0, -1.0, 1.0)  # P-controller to achieve desired roll

        # === G-PULL CONTROL (Turn and maintain altitude) ===
        # Base G-load: Higher when banked (to maintain level turn)
        desired_g = 1.0 / max(0.2, math.cos(ent.roll))  # G = 1/cos(roll) to maintain altitude during turn
        
        # Altitude keeper: Add G adjustment to return to target altitude
        target_alt = 10000.0  # Preferred cruise altitude (10km)
        alt_err = target_alt - ent.alt
        desired_g += np.clip(alt_err * 0.001, -0.5, 2.0)  # Proportional altitude correction
        
        # Normalize to action space
        g_cmd = (desired_g - 1.0) / (self.cfg.MAX_G - 1.0)
        g_cmd = np.clip(g_cmd, -0.2, 1.0)

        # === CURRICULUM LEARNING (Execution Noise) ===
        # Even when AI makes correct decision, add jitter based on κ (imperfect execution)
        roll_cmd += np.random.normal(0, kappa * 0.5)  # Gaussian noise in roll command
        g_cmd += np.random.normal(0, kappa * 0.2)     # Gaussian noise in G command

        # Clip final values to valid action range
        roll_cmd = np.clip(roll_cmd, -1.0, 1.0)
        g_cmd = np.clip(g_cmd, -0.2, 1.0)

        # === THROTTLE (Always maximum) ===
        throttle = 1.0  # AI always uses full throttle (aggressive)

        # === WEAPON EMPLOYMENT ===
        fire = 0.0
        angle_off = abs(heading_err)
        
        # Only attempt firing if AI is competent (low κ)
        if kappa < 0.5:
            # Fire if: within radar range AND target is in radar FOV
            if dist_m < self.cfg.RADAR_RANGE_KM * 1000.0 and angle_off < self.cfg.RADAR_FOV_DEG:
                if np.random.rand() < 0.05:  # 5% chance per tick (probabilistic firing)
                    fire = 1.0

        # === COUNTERMEASURES ===
        cm = 0.0
        # Simple probabilistic countermeasure usage (could be improved with threat detection)
        if np.random.rand() < 0.01:  # 1% chance per tick
            cm = 1.0

        return [roll_cmd, g_cmd, throttle, fire, cm]

    def _try_fire(self, ent):
        """
        Attempt to fire a missile at a locked target.
        Only succeeds if:
        - There are enemy targets
        - Sensor has valid lock on at least one target
        - Ammunition is available (checked by caller)
        
        Args:
            ent: The firing entity (aircraft)
        """
        # Find all enemy aircraft
        targets = [e for e in self.entities.values() if e.team != ent.team and e.type == "plane"]
        
        # Check each potential target for valid lock
        for t in targets:
            visible, locking = self.get_sensor_state(ent.uid, t.uid)
            if locking:
                # Valid lock achieved: spawn missile at shooter's position
                m_uid = self.spawn(ent.x, ent.y, ent.heading, ent.speed, ent.team, "missile")
                
                # Configure missile to track this target
                self.entities[m_uid].target_id = t.uid
                self.entities[m_uid].time_alive = 0.0
                
                # Decrement ammunition count
                ent.ammo -= 1
                
                # Log event for reward system tracking
                self.events.append({"shooter": ent.uid, "target": t.uid, "type": "missile_fired"})
                
                # Only fire one missile per action (no multi-shot)
                break

    def _update_missile(self, ent):
        """
        Update missile physics and guidance for one sub-timestep.
        Implements:
        - Proportional navigation guidance
        - Boost-sustain motor model
        - Countermeasure susceptibility
        - Fuel/range limitations
        
        Args:
            ent: Entity object representing the missile
        """
        dt = self.cfg.PHYSICS_DT  # Use physics sub-timestep (0.01s)
        ent.time_alive += dt  # Track missile flight time

        # Check if target still exists (may have been destroyed)
        if ent.target_id not in self.entities:
            del self.entities[ent.uid]  # Target lost: missile self-destructs
            return
        target = self.entities[ent.target_id]

        # === COUNTERMEASURES CHECK ===
        if target.cm_active:
            # Target is deploying chaff/flares: probabilistic spoof check
            if np.random.rand() < self.cfg.CM_SPOOF_PROB:
                # Missile seeker fooled by countermeasures: loses lock and self-destructs
                del self.entities[ent.uid]
                return

        # === MISSILE MOTOR MODEL ===
        g = self.cfg.GRAVITY
        thrust = 0.0
        # Boost phase: High thrust for initial seconds
        if ent.time_alive < self.cfg.MISSILE_BOOST_SEC:
            thrust = self.cfg.MISSILE_BOOST_ACCEL
        # After boost phase: no thrust (coasting/sustain - simplified, no sustain motor modeled)
        
        # === DRAG FORCES ===
        # Parasitic drag: Scales with V²
        drag_p = self.cfg.MISSILE_DRAG_PARASITIC * (ent.speed ** 2)

        # === PROPORTIONAL NAVIGATION GUIDANCE ===
        # Calculate required heading to intercept target
        bearing = bearing_deg(ent.x, ent.y, target.x, target.y)
        diff = (bearing - ent.heading + 180) % 360 - 180  # Heading error
        
        # Calculate required turn rate to null heading error in one timestep (aggressive guidance)
        req_turn_rate_rad = math.radians(diff / dt)
        
        # Convert turn rate to required lateral acceleration
        req_accel = (ent.speed * 0.5144) * abs(req_turn_rate_rad)  # a = V × ω
        req_g = req_accel / g  # Convert to G-units
        
        # Limit to missile's maximum maneuverability
        actual_g = min(req_g, self.cfg.MISSILE_MAX_G)
        
        # Convert back to achievable turn rate
        valid_turn_rate_deg = math.degrees((actual_g * g) / (ent.speed * 0.5144 + 1e-5))
        turn_step = valid_turn_rate_deg * dt
        
        # Update heading toward target
        if abs(diff) < turn_step:
            ent.heading = bearing  # Close enough: point directly at target
        else:
            ent.heading += math.copysign(turn_step, diff)  # Turn in correct direction
        ent.heading %= 360.0  # Wrap to [0, 360]
        
        # Induced drag from maneuvering: Scales with G²
        drag_i = self.cfg.MISSILE_DRAG_INDUCED * (actual_g ** 2)
        
        # === VELOCITY UPDATE ===
        # Net acceleration: (Thrust - Drag) × scaling factor
        ent.speed += (thrust - (drag_p + drag_i) * 100.0) * dt
        
        # Minimum speed check: Missile runs out of energy and falls
        if ent.speed < self.cfg.MISSILE_MIN_SPEED:
            del self.entities[ent.uid]  # Missile expires (fuel exhausted or stalled)
            return
        
        # === POSITION UPDATE (CARTESIAN) ===
        # Calculate distance traveled this timestep
        dist = (ent.speed * 0.5144) * dt  # Convert knots to m/s
        
        # Update position using Cartesian trigonometry
        dx = dist * math.cos(math.radians(ent.heading))
        dy = dist * math.sin(math.radians(ent.heading))
        ent.x += dx
        ent.y += dy

    def _resolve_collisions(self):
        """
        Check for missile-target collisions and resolve hits.
        Uses simple proximity detection (1km radius).
        When hit occurs:
        - Logs kill event
        - Removes both missile and target from simulation
        """
        # Get all active missiles
        missiles = [e for e in self.entities.values() if e.type == "missile"]
        
        for m in missiles:
            # Check if missile's target still exists
            if m.target_id in self.entities:
                t = self.entities[m.target_id]
                
                # Calculate distance between missile and target
                dist = dist_2d(m.x, m.y, t.x, t.y)
                
                # Proximity fuse: Trigger if within 1km (1000m)
                if dist < 1000.0:
                    # HIT! Log kill event for scoring/rewards
                    self.events.append({"killer": m.uid, "victim": t.uid, "type": "kill"})
                    
                    # Remove both missile and target from simulation
                    if t.uid in self.entities: del self.entities[t.uid]  # Target destroyed
                    if m.uid in self.entities: del self.entities[m.uid]  # Missile expended
    
    def _check_midair_collisions(self):
        """
        Check for plane-vs-plane collisions (aircraft flying through each other).
        Uses proximity detection with 50m threshold.
        
        When collision occurs:
        - Logs midair collision event
        - Destroys both aircraft
        
        This prevents unrealistic scenarios where aircraft can fly through each other
        with no consequences during close-range maneuvering.
        """
        # Get all active aircraft
        planes = [e for e in self.entities.values() if e.type == "plane"]
        
        # Check all pairs of aircraft (avoid double-checking)
        for i, p1 in enumerate(planes):
            for p2 in planes[i+1:]:  # Only check each pair once
                # Calculate 3D distance between aircraft
                # Horizontal component
                dist_horiz_m = dist_2d(p1.x, p1.y, p2.x, p2.y)
                
                # Vertical component (altitude difference)
                alt_diff_m = abs(p1.alt - p2.alt)
                
                # 3D distance using Pythagorean theorem
                dist_3d_m = math.sqrt(dist_horiz_m**2 + alt_diff_m**2)
                
                # Collision threshold: 50 meters
                COLLISION_THRESHOLD = 50.0  # meters
                
                if dist_3d_m < COLLISION_THRESHOLD:
                    # MIDAIR COLLISION! Both aircraft destroyed
                    # Log event (arbitrarily pick p1 as "victim" for consistency)
                    self.events.append({
                        "type": "midair_collision", 
                        "victim": p1.uid, 
                        "killer": p2.uid
                    })
                    
                    # Destroy both aircraft
                    if p1.uid in self.entities: del self.entities[p1.uid]
                    if p2.uid in self.entities: del self.entities[p2.uid]
                    
                    # Break out of inner loop since p1 is now destroyed
                    break