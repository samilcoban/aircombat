# ================================================
# FILE: src/render_panda3d.py
# ================================================
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import math
import os
import collections

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
        # Intuition: Rebuilding the geometry every frame is inefficient for very long trails,
        # but acceptable for short tactical trails.
        self.segs.moveTo(self.points[0])
        for p in list(self.points)[1:]:
            self.segs.drawTo(p)

        # Replace the geometry
        self.node_path.node().removeAllChildren()
        self.node_path.attachNewNode(self.segs.create())


class Panda3DRenderer(ShowBase):
    """
    Real-time 3D visualization using the Panda3D engine.
    
    Coordinate System Mapping:
    - Simulation: X=North, Y=East, Z=Up (NED-like but Z-up)
    - Panda3D: Y=North, X=East, Z=Up
    - Conversion: Sim(X, Y) -> Panda(Y, X)
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

        self.camera_target = None  # The node we are following
        self.camera_focus = None  # The node we are looking AT (enemy)

        # Camera Smoothing State
        # Intuition: Smooth camera movement prevents motion sickness and jitter.
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
        # Intuition: Provides a visual reference for altitude and speed.
        segs = LineSegs()
        segs.setColor(0.3, 0.3, 0.3, 0.5)
        for i in range(-50, 51, 5):  # 5km lines
            segs.moveTo(i * 100, -5000, 0);
            segs.drawTo(i * 100, 5000, 0)
            segs.moveTo(-5000, i * 100, 0);
            segs.drawTo(5000, i * 100, 0)
        self.render.attachNewNode(segs.create()).setPos(0, 0, -10)

    def load_model_assets(self):
        # Try load GLTF, fallback to boxes
        if GLTF_AVAILABLE and os.path.exists("assets/f16.gltf"):
            try:
                m = self.loader.loadModel("assets/f16.gltf")
                m.setScale(10)  # Adjust scale
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
        # Procedural low-poly plane
        p = self.loader.loadModel("models/box")
        p.setScale(2, 5, 0.5)
        self.model_assets['plane'] = p

    def update_entities(self, entities, map_limits):
        active_uids = set()

        # Scale factor: 1 unit = 10 meters for better rendering depth precision
        # Original: Meters. Renderer: Decameters.
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
                    # Add Trail
                    self.trails[uid] = TrailRenderer(self.render, Vec4(*color), length=200)
                else:
                    model = self.model_assets['missile'].copyTo(self.render)
                    model.setColor(1, 1, 0, 1)
                    self.trails[uid] = TrailRenderer(self.render, Vec4(1, 1, 0, 0.5), length=50)

                self.nodes[uid] = model

            # 2. Update Position/Rotation
            # Map: X=North, Y=East, Z=Up.
            # Panda: Y=North, X=East, Z=Up.
            # So: ent.y -> Panda X, ent.x -> Panda Y.
            # Intuition: Coordinate swap is necessary because simulation uses NED-like (North-East-Down/Up)
            # while Panda3D uses Y-Forward (North), X-Right (East), Z-Up.
            pos = Vec3(ent.y * SCALE, ent.x * SCALE, ent.alt * SCALE)
            hpr = Vec3(ent.heading + 180, -math.degrees(ent.pitch), math.degrees(ent.roll))  # Panda H is reversed

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

        # 4. Update Camera Logic (Tactical View)
        self._update_camera(blue_plane, red_plane)
        self.taskMgr.step()

    def _update_camera(self, hero, enemy):
        if not hero: return

        # Target Position (where the plane is)
        hero_pos = hero.getPos()

        # Desired Camera Offset (Behind and Above)
        # We calculate this relative to the WORLD, not the plane's rotation,
        # to prevent the "wiggling" motion sickness.

        if enemy:
            # TACTICAL MODE: Keep both planes in view
            # Intuition: In a dogfight, the pilot (and the viewer) cares about the relative position
            # of the enemy. We position the camera to frame both combatants.
            enemy_pos = enemy.getPos()
            midpoint = (hero_pos + enemy_pos) * 0.5
            dist = (hero_pos - enemy_pos).length()

            # Position camera behind hero, but looking at midpoint
            # Vector from enemy to hero
            direction = (hero_pos - enemy_pos).normalized()

            # Zoom out based on distance
            zoom = max(100, dist * 0.8)
            target_cam_pos = hero_pos + (direction * zoom) + Vec3(0, 0, zoom * 0.4)
            look_target = midpoint

        else:
            # CHASE MODE: Just follow hero, but soft-locked
            # Get velocity vector approximation from heading
            h = math.radians(hero.getH())
            heading_vec = Vec3(-math.sin(h), math.cos(h), 0)

            target_cam_pos = hero_pos - (heading_vec * 80) + Vec3(0, 0, 30)
            look_target = hero_pos + (heading_vec * 200)

        # Smooth Damping (Lerp)
        # Adjust 'dt' factor for smoothness (0.1 = slow/cinematic, 0.5 = snappy)
        self.cam_pos_smooth = self.cam_pos_smooth + (target_cam_pos - self.cam_pos_smooth) * 0.1
        self.cam_look_smooth = self.cam_look_smooth + (look_target - self.cam_look_smooth) * 0.1

        self.camera.setPos(self.cam_pos_smooth)
        self.camera.lookAt(self.cam_look_smooth)

    def check_running(self):
        return self.is_running and self.win.isValid()

    def cleanup(self):
        self.destroy()