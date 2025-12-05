# ================================================
# FILE: src/render_panda3d.py
# ================================================
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import math
import os
import collections
import numpy as np

# Import GLTF loader
try:
    from panda3d_gltf import loader as gltf_loader

    GLTF_AVAILABLE = True
except ImportError:
    GLTF_AVAILABLE = False
    print("⚠️  panda3d-gltf not available, using procedural geometry only")


class TrailRenderer:
    """Draws a 3D ribbon behind an object to visualize flight path."""

    def __init__(self, parent_node, color, length=100, thickness=2.0):
        self.points = collections.deque(maxlen=length)
        self.node_path = parent_node.attachNewNode("trail")
        self.color = color
        self.thickness = thickness
        self.segs = LineSegs()
        self.segs.setThickness(thickness)
        self.segs.setColor(color)

    def update(self, pos):
        # Add new point
        self.points.append(pos)

        # Redraw
        # Optimization: Only redraw if enough points exist
        if len(self.points) < 2: return

        self.segs.reset()
        self.segs.setThickness(self.thickness)
        self.segs.setColor(self.color)

        self.segs.moveTo(self.points[0])
        for p in list(self.points)[1:]:
            self.segs.drawTo(p)

        # Remove old geometry and attach new
        self.node_path.node().removeAllChildren()
        self.node_path.attachNewNode(self.segs.create())


class Panda3DRenderer(ShowBase):
    """
    Real-time 3D visualization using the Panda3D engine.

    UPDATED: Handles Unit Conversion (Radians -> Degrees) for Phase 1-5 Refactor.

    Coordinate System Mapping:
    - Simulation: X=North, Y=East, Z=Up (Right-Handed Z-Up)
    - Panda3D: X=Right, Y=Forward, Z=Up (Right-Handed Z-Up)
    - Mapping: Sim X -> Panda Y (North), Sim Y -> Panda X (East)
    """

    def __init__(self):
        ShowBase.__init__(self)

        if GLTF_AVAILABLE:
            try:
                gltf_loader.patch_loader(self.loader)
            except:
                pass

        # Window & Camera
        self.win_props = WindowProperties()
        self.win_props.setTitle("AirCombat 3.0: Tactical View")
        self.win_props.setSize(1280, 720)
        self.win.requestProperties(self.win_props)
        self.disableMouse()
        self.setBackgroundColor(0.1, 0.1, 0.15, 1)  # Dark Blue Sky

        self.setup_lights()
        self.setup_environment()

        self.model_assets = {}
        self.load_model_assets()

        self.nodes = {}  # Map UID -> NodePath
        self.trails = {}  # Map UID -> TrailRenderer

        self.camera_target = None
        self.camera_focus = None

        # Camera Smoothing State
        self.cam_pos_smooth = Vec3(0, -100, 50)
        self.cam_look_smooth = Vec3(0, 0, 0)

        self.is_running = True

    def setup_lights(self):
        ambient = AmbientLight("ambient")
        ambient.setColor((0.4, 0.4, 0.5, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor((0.9, 0.9, 0.8, 1))
        sun.setShadowCaster(True, 2048, 2048)
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        self.render.setLight(sun_np)

    def setup_environment(self):
        # Grid Floor
        segs = LineSegs()
        segs.setColor(0.3, 0.3, 0.3, 0.5)
        # Draw a 100km x 100km grid
        # Scaled down by 0.1 for rendering stability (1 unit = 10m)
        step = 500  # 5km lines
        limit = 5000

        for i in range(-limit, limit + 1, step):
            # North-South lines (Along Y)
            segs.moveTo(i, -limit, 0)
            segs.drawTo(i, limit, 0)
            # East-West lines (Along X)
            segs.moveTo(-limit, i, 0)
            segs.drawTo(limit, i, 0)

        node = self.render.attachNewNode(segs.create())
        node.setPos(0, 0, -10)

    def load_model_assets(self):
        # Try load GLTF, fallback to boxes
        if GLTF_AVAILABLE and os.path.exists("assets/f16.gltf"):
            try:
                m = self.loader.loadModel("assets/f16.gltf")
                m.setScale(10)  # Adjust scale for visibility
                self.model_assets['plane'] = m
            except:
                self._make_placeholder_plane()
        else:
            self._make_placeholder_plane()

        # Missile
        m = self.loader.loadModel("models/box")
        m.setScale(0.5, 2.0, 0.5)
        self.model_assets['missile'] = m

    def _make_placeholder_plane(self):
        # Procedural low-poly plane (Triangle-ish box)
        p = self.loader.loadModel("models/box")
        p.setScale(2, 5, 0.5)
        self.model_assets['plane'] = p

    def update_entities(self, entities, map_limits):
        active_uids = set()

        # Scale factor: 1 unit = 10 meters
        # (Panda3D precision logic: keeping coordinates smaller helps Z-buffer)
        SCALE = 0.1

        blue_plane = None
        red_plane = None

        for uid, ent in entities.items():
            active_uids.add(uid)

            # 1. Create Node if missing
            if uid not in self.nodes:
                if ent.type == "plane":
                    model = self.model_assets['plane'].copyTo(self.render)
                    color = (0, 0.5, 1, 1) if ent.team == "blue" else (1, 0.2, 0.2, 1)
                    model.setColor(*color)
                    self.trails[uid] = TrailRenderer(self.render, Vec4(*color), length=200)
                else:
                    model = self.model_assets['missile'].copyTo(self.render)
                    model.setColor(1, 1, 0, 1)
                    self.trails[uid] = TrailRenderer(self.render, Vec4(1, 1, 0, 0.5), length=50)

                self.nodes[uid] = model

            # 2. Update Position/Rotation
            # COORDINATE TRANSFORMATION
            # Sim: X=North, Y=East
            # Panda: Y=North, X=East
            # Swap X/Y
            pos = Vec3(ent.y * SCALE, ent.x * SCALE, ent.alt * SCALE)

            # ROTATION TRANSFORMATION
            # Sim: Heading (Rad), Pitch (Rad), Roll (Rad)
            # Panda: H (Deg), P (Deg), R (Deg)
            # Heading 0 (North) -> Panda 0 (North, which is +Y)
            # Sim Heading increases CW (0->East->South) ?
            # Standard Math: 0=East, Counter-Clockwise.
            # Aviation/Sim Core: We implemented 0=North, PI/2=East (Clockwise-ish if using sin/cos map).
            # Core Logic: x = cos(p)*cos(h), y = cos(p)*sin(h).
            # If h=0, x=1 (North). If h=PI/2, y=1 (East).

            # Panda H: 0 is +Y axis (North).
            # Panda H increases Counter-Clockwise (Standard Right-Hand Rule Z-up).
            # Sim: 0 is North. PI/2 is East.
            # To map Sim (CW) to Panda (CCW), we negate Heading.
            # Offset: Sim 0 = Panda 0.

            h_deg = -math.degrees(ent.heading)
            p_deg = math.degrees(ent.pitch)
            r_deg = math.degrees(ent.roll)

            hpr = Vec3(h_deg, p_deg, r_deg)

            self.nodes[uid].setPos(pos)
            self.nodes[uid].setHpr(hpr)

            # Update Trail
            if uid in self.trails:
                self.trails[uid].update(pos)

            # Identify Key Actors for Camera
            if ent.type == "plane":
                if ent.team == "blue": blue_plane = self.nodes[uid]
                if ent.team == "red": red_plane = self.nodes[uid]

        # 3. Cleanup Dead
        for uid in list(self.nodes.keys()):
            if uid not in active_uids:
                self.nodes[uid].removeNode()
                if uid in self.trails:
                    self.trails[uid].node_path.removeNode()
                    del self.trails[uid]
                del self.nodes[uid]

        # 4. Update Camera Logic
        self._update_camera(blue_plane, red_plane)
        self.taskMgr.step()

    def _update_camera(self, hero, enemy):
        if not hero: return

        hero_pos = hero.getPos()

        if enemy:
            # TACTICAL MODE: Frame the fight
            enemy_pos = enemy.getPos()
            midpoint = (hero_pos + enemy_pos) * 0.5

            # Direction vector from enemy to hero (to place camera behind hero)
            diff = hero_pos - enemy_pos
            dist = diff.length()

            if dist < 1.0:  # Too close, default to simple offset
                direction = Vec3(0, -1, 0)
            else:
                direction = diff.normalized()

            # Dynamic Zoom
            zoom = max(100, dist * 1.5)
            # Lift camera up based on zoom
            cam_offset = direction * zoom + Vec3(0, 0, zoom * 0.4)

            target_cam_pos = hero_pos + cam_offset
            look_target = midpoint

        else:
            # CHASE MODE
            # Get hero heading vector from HPR
            h_rad = math.radians(hero.getH())
            # Panda Forward vector based on H
            # H=0 -> +Y. H=90 -> -X (CCW).
            # x = -sin(h), y = cos(h)
            forward = Vec3(-math.sin(h_rad), math.cos(h_rad), 0)

            target_cam_pos = hero_pos - (forward * 80) + Vec3(0, 0, 30)
            look_target = hero_pos + (forward * 200)

        # Smooth Damping
        dt = 0.1  # Smoothing factor
        self.cam_pos_smooth = self.cam_pos_smooth + (target_cam_pos - self.cam_pos_smooth) * dt
        self.cam_look_smooth = self.cam_look_smooth + (look_target - self.cam_look_smooth) * dt

        self.camera.setPos(self.cam_pos_smooth)
        self.camera.lookAt(self.cam_look_smooth)

    def check_running(self):
        return self.is_running and self.win.isValid()

    def cleanup(self):
        self.destroy()