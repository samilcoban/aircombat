# ================================================
# FILE: src/render_panda3d.py
# ================================================
"""
Real-time 3D visualization using Panda3D game engine.

This module provides a 3D renderer for visualizing air combat scenarios
in real-time. It is primarily used for debugging and demonstrations.

Features:
- 3D aircraft and missile rendering with team colors
- Flight path trails for trajectory visualization
- Dynamic camera with tactical (dogfight) and chase modes
- GLTF model support if available, fallback to procedural geometry

Coordinate Systems:
- Simulation: X=North, Y=East, Z=Up (Right-Handed Z-Up)
- Panda3D: X=Right, Y=Forward, Z=Up (Right-Handed Z-Up)
- Mapping: Sim X -> Panda Y (North), Sim Y -> Panda X (East)
"""
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import math
import os
import collections
import numpy as np

# Import GLTF loader for custom 3D models.
try:
    from panda3d_gltf import loader as gltf_loader

    GLTF_AVAILABLE = True
except ImportError:
    GLTF_AVAILABLE = False
    print("⚠️  panda3d-gltf not available, using procedural geometry only")


class TrailRenderer:
    """
    Draws a 3D ribbon behind an object to visualize flight path.
    
    Uses Panda3D LineSegs for efficient trail rendering with configurable
    length and appearance.
    """

    def __init__(self, parent_node, color, length=100, thickness=2.0):
        """
        Initialize trail renderer.
        
        Args:
            parent_node: Panda3D node to attach trail to.
            color: Trail color as Vec4 (r, g, b, a).
            length: Maximum number of trail points to keep.
            thickness: Line thickness in pixels.
        """
        self.points = collections.deque(maxlen=length)
        self.node_path = parent_node.attachNewNode("trail")
        self.color = color
        self.thickness = thickness
        self.segs = LineSegs()
        self.segs.setThickness(thickness)
        self.segs.setColor(color)

    def update(self, pos):
        """
        Add new position to trail and redraw.
        
        Args:
            pos: Current position as Vec3.
        """
        # Add new point.
        self.points.append(pos)

        # Redraw only if enough points exist.
        if len(self.points) < 2: return

        self.segs.reset()
        self.segs.setThickness(self.thickness)
        self.segs.setColor(self.color)

        self.segs.moveTo(self.points[0])
        for p in list(self.points)[1:]:
            self.segs.drawTo(p)

        # Remove old geometry and attach new.
        self.node_path.node().removeAllChildren()
        self.node_path.attachNewNode(self.segs.create())


class Panda3DRenderer(ShowBase):
    """
    Real-time 3D visualization using the Panda3D engine.

    Handles unit conversion from simulation (radians) to Panda3D (degrees)
    and coordinate system mapping between simulation and rendering.

    Coordinate System Mapping:
    - Simulation: X=North, Y=East, Z=Up (Right-Handed Z-Up)
    - Panda3D: X=Right, Y=Forward, Z=Up (Right-Handed Z-Up)
    - Mapping: Sim X -> Panda Y (North), Sim Y -> Panda X (East)
    """

    def __init__(self):
        """Initialize renderer with window, lights, and assets."""
        ShowBase.__init__(self)

        if GLTF_AVAILABLE:
            try:
                gltf_loader.patch_loader(self.loader)
            except:
                pass

        # Window & Camera setup.
        self.win_props = WindowProperties()
        self.win_props.setTitle("AirCombat 3.0: Tactical View")
        self.win_props.setSize(1280, 720)
        self.win.requestProperties(self.win_props)
        self.disableMouse()  # Disable default mouse camera control.
        self.setBackgroundColor(0.1, 0.1, 0.15, 1)  # Dark Blue Sky.

        self.setup_lights()
        self.setup_environment()

        # Load 3D model assets.
        self.model_assets = {}
        self.load_model_assets()

        # Entity tracking.
        self.nodes = {}   # Map UID -> NodePath.
        self.trails = {}  # Map UID -> TrailRenderer.

        self.camera_target = None
        self.camera_focus = None

        # Camera smoothing state.
        self.cam_pos_smooth = Vec3(0, -100, 50)
        self.cam_look_smooth = Vec3(0, 0, 0)

        self.is_running = True

    def setup_lights(self):
        """Setup ambient and directional lighting."""
        # Ambient light for general illumination.
        ambient = AmbientLight("ambient")
        ambient.setColor((0.4, 0.4, 0.5, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        # Sun (directional light) with shadows.
        sun = DirectionalLight("sun")
        sun.setColor((0.9, 0.9, 0.8, 1))
        sun.setShadowCaster(True, 2048, 2048)
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)  # Sun angle.
        self.render.setLight(sun_np)

    def setup_environment(self):
        """Create ground grid for spatial reference."""
        segs = LineSegs()
        segs.setColor(0.3, 0.3, 0.3, 0.5)
        
        # Draw a 100km x 100km grid.
        # Scaled down by 0.1 for rendering stability (1 unit = 10m).
        step = 500   # 5km grid lines.
        limit = 5000  # 50km in each direction.

        for i in range(-limit, limit + 1, step):
            # North-South lines (Along Y).
            segs.moveTo(i, -limit, 0)
            segs.drawTo(i, limit, 0)
            # East-West lines (Along X).
            segs.moveTo(-limit, i, 0)
            segs.drawTo(limit, i, 0)

        node = self.render.attachNewNode(segs.create())
        node.setPos(0, 0, -10)  # Slightly below origin.

    def load_model_assets(self):
        """Load or create 3D models for entities."""
        # Try load GLTF model, fallback to boxes.
        if GLTF_AVAILABLE and os.path.exists("assets/f16.gltf"):
            try:
                m = self.loader.loadModel("assets/f16.gltf")
                m.setScale(10)  # Adjust scale for visibility.
                self.model_assets['plane'] = m
            except:
                self._make_placeholder_plane()
        else:
            self._make_placeholder_plane()

        # Missile (simple elongated box).
        m = self.loader.loadModel("models/box")
        m.setScale(0.5, 2.0, 0.5)
        self.model_assets['missile'] = m

    def _make_placeholder_plane(self):
        """Create procedural low-poly plane geometry."""
        p = self.loader.loadModel("models/box")
        p.setScale(2, 5, 0.5)  # Stretched box as plane.
        self.model_assets['plane'] = p

    def update_entities(self, entities, map_limits):
        """
        Update all entity positions and rotations in the scene.
        
        Args:
            entities: Dict of UID -> Entity from simulation.
            map_limits: MapLimits object (unused but kept for API).
        """
        active_uids = set()

        # Scale factor: 1 unit = 10 meters.
        # Panda3D precision benefits from smaller coordinates.
        SCALE = 0.1

        blue_plane = None
        red_plane = None

        for uid, ent in entities.items():
            active_uids.add(uid)

            # 1. Create Node if missing.
            if uid not in self.nodes:
                if ent.type == "plane":
                    model = self.model_assets['plane'].copyTo(self.render)
                    color = (0, 0.5, 1, 1) if ent.team == "blue" else (1, 0.2, 0.2, 1)
                    model.setColor(*color)
                    self.trails[uid] = TrailRenderer(self.render, Vec4(*color), length=200)
                else:
                    model = self.model_assets['missile'].copyTo(self.render)
                    model.setColor(1, 1, 0, 1)  # Yellow for missiles.
                    self.trails[uid] = TrailRenderer(self.render, Vec4(1, 1, 0, 0.5), length=50)

                self.nodes[uid] = model

            # 2. Update Position/Rotation.
            # COORDINATE TRANSFORMATION:
            # Sim: X=North, Y=East -> Panda: Y=North, X=East (swap X/Y).
            pos = Vec3(ent.y * SCALE, ent.x * SCALE, ent.alt * SCALE)

            # ROTATION TRANSFORMATION:
            # Sim: Heading/Pitch/Roll in radians.
            # Panda: H/P/R in degrees.
            # Heading 0 (North) -> Panda 0 (North/+Y).
            # Sim heading increases clockwise, Panda counter-clockwise, so negate.
            h_deg = -math.degrees(ent.heading)
            p_deg = math.degrees(ent.pitch)
            r_deg = math.degrees(ent.roll)

            hpr = Vec3(h_deg, p_deg, r_deg)

            self.nodes[uid].setPos(pos)
            self.nodes[uid].setHpr(hpr)

            # Update trail.
            if uid in self.trails:
                self.trails[uid].update(pos)

            # Identify key actors for camera.
            if ent.type == "plane":
                if ent.team == "blue": blue_plane = self.nodes[uid]
                if ent.team == "red": red_plane = self.nodes[uid]

        # 3. Cleanup dead entities.
        for uid in list(self.nodes.keys()):
            if uid not in active_uids:
                self.nodes[uid].removeNode()
                if uid in self.trails:
                    self.trails[uid].node_path.removeNode()
                    del self.trails[uid]
                del self.nodes[uid]

        # 4. Update camera.
        self._update_camera(blue_plane, red_plane)
        self.taskMgr.step()

    def _update_camera(self, hero, enemy):
        """
        Update camera position and look-at target.
        
        Uses two modes:
        - TACTICAL: When enemy present, frame both aircraft.
        - CHASE: When solo, follow from behind.
        
        Args:
            hero: Blue team aircraft NodePath.
            enemy: Red team aircraft NodePath (or None).
        """
        if not hero: return

        hero_pos = hero.getPos()

        if enemy:
            # TACTICAL MODE: Frame the fight.
            enemy_pos = enemy.getPos()
            midpoint = (hero_pos + enemy_pos) * 0.5

            # Direction from enemy to hero (camera behind hero).
            diff = hero_pos - enemy_pos
            dist = diff.length()

            if dist < 1.0:  # Too close, default offset.
                direction = Vec3(0, -1, 0)
            else:
                direction = diff.normalized()

            # Dynamic zoom based on separation.
            zoom = max(100, dist * 1.5)
            cam_offset = direction * zoom + Vec3(0, 0, zoom * 0.4)

            target_cam_pos = hero_pos + cam_offset
            look_target = midpoint

        else:
            # CHASE MODE: Follow from behind.
            h_rad = math.radians(hero.getH())
            # Panda forward: H=0 -> +Y, H=90 -> -X (CCW).
            forward = Vec3(-math.sin(h_rad), math.cos(h_rad), 0)

            target_cam_pos = hero_pos - (forward * 80) + Vec3(0, 0, 30)
            look_target = hero_pos + (forward * 200)

        # Smooth camera movement.
        dt = 0.1  # Smoothing factor.
        self.cam_pos_smooth = self.cam_pos_smooth + (target_cam_pos - self.cam_pos_smooth) * dt
        self.cam_look_smooth = self.cam_look_smooth + (look_target - self.cam_look_smooth) * dt

        self.camera.setPos(self.cam_pos_smooth)
        self.camera.lookAt(self.cam_look_smooth)

    def check_running(self):
        """Check if renderer window is still open."""
        return self.is_running and self.win.isValid()

    def cleanup(self):
        """Clean up renderer resources."""
        self.destroy()