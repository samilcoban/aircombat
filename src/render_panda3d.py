from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import math
import os

# Import GLTF loader
try:
    from panda3d_gltf import loader as gltf_loader
    GLTF_AVAILABLE = True
except ImportError:
    GLTF_AVAILABLE = False
    print("⚠️  panda3d-gltf not available, using procedural geometry only")

class Panda3DRenderer(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Enable GLTF support if available
        if GLTF_AVAILABLE:
            try:
                gltf_loader.patch_loader(self.loader)
                print("✅ GLTF support enabled")
            except Exception as e:
                print(f"⚠️  GLTF loader patch failed: {e}")
        
        # Window setup
        self.win_props = WindowProperties()
        self.win_props.setTitle("AirCombat Simulation")
        self.win_props.setSize(1280, 720)
        self.win.requestProperties(self.win_props)
        
        # Disable default camera control
        self.disableMouse()
        
        # Setup lighting
        self.setup_lights()
        
        # Setup ground plane
        self.setup_ground()
        
        # Model asset cache
        self.model_assets = {}
        self.load_model_assets()
        
        # Entity cache: Map UID -> NodePath
        self.nodes = {}
        
        # Reference center for coordinate conversion
        self.ref_lat = None
        self.ref_lon = None
        
        # Camera follow target
        self.follow_target = None
        
        # Initialize camera position
        self.camera.setPos(0, -100, 50)
        self.camera.lookAt(0, 0, 0)
        
        # Running flag
        self.is_running = True
        
    def setup_lights(self):
        # Ambient light
        ambient = AmbientLight("ambient")
        ambient.setColor((0.3, 0.3, 0.3, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)
        
        # Directional light (sun)
        sun = DirectionalLight("sun")
        sun.setColor((0.8, 0.8, 0.8, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -45, 0)
        self.render.setLight(sun_np)
        
    def setup_ground(self):
        # Create a large ground plane for reference
        cm = CardMaker("ground")
        cm.setFrame(-5000, 5000, -5000, 5000)
        ground = self.render.attachNewNode(cm.generate())
        ground.setP(-90)  # Rotate to be horizontal
        ground.setPos(0, 0, -10)
        ground.setColor(0.2, 0.3, 0.2, 1)
        
    def load_model_assets(self):
        """Load 3D models for aircraft and missiles with fallback to procedural geometry"""
        
        # Try to load GLTF models
        gltf_paths = [
            ("blue_plane", "assets/f16.gltf", (0, 0, 1, 1)),
            ("blue_plane_glb", "assets/f16.glb", (0, 0, 1, 1)),
            ("red_plane", "assets/f16.gltf", (1, 0, 0, 1)),
            ("red_plane_glb", "assets/f16.glb", (1, 0, 0, 1)),
        ]
        
        loaded_plane = False
        for name, path, color in gltf_paths:
            if os.path.exists(path):
                try:
                    model = self.loader.loadModel(path)
                    if model:
                        # Auto-normalize scale
                        bounds = model.getTightBounds()
                        if bounds:
                            min_pt, max_pt = bounds
                            size = max_pt - min_pt
                            current_len = size.y
                            if current_len > 0:
                                # Force jet to be 15 units long (represents ~15 meters)
                                scale_factor = 15.0 / current_len
                                model.setScale(scale_factor)
                        
                        model.setColor(color)
                        self.model_assets[name] = model
                        loaded_plane = True
                        print(f"✅ Loaded {name} from {path}")
                        break  # Use first successful load
                except Exception as e:
                    print(f"⚠️  Failed to load {path}: {e}")
        
        if not loaded_plane:
            print("ℹ️  No GLTF models found, using procedural geometry")
            # Fallback to procedural models
            self.model_assets["blue_plane"] = self.create_procedural_plane((0, 0, 1, 1))
            self.model_assets["red_plane"] = self.create_procedural_plane((1, 0, 0, 1))
        
        # Missile always uses simple geometry
        self.model_assets["missile"] = self.create_procedural_missile()
        
    def create_procedural_plane(self, color):
        """Create a simple plane shape using basic geometry"""
        # Fuselage (elongated box)
        fuselage = self.loader.loadModel("models/box")
        fuselage.setScale(0.5, 2, 0.3)
        fuselage.setColor(color)
        
        # Wings (flat box)
        wings = self.loader.loadModel("models/box")
        wings.setScale(2, 0.5, 0.1)
        wings.setPos(0, 0, 0)
        wings.setColor(color)
        wings.reparentTo(fuselage)
        
        return fuselage
        
    def create_procedural_missile(self):
        """Create a simple missile shape"""
        missile = self.loader.loadModel("models/box")
        missile.setScale(0.2, 0.8, 0.2)
        missile.setColor(1, 1, 0, 1)  # Yellow
        return missile
        
    def update_entities(self, entities, map_limits):
        """Update entity positions and orientations"""
        if self.ref_lat is None:
            # Initialize reference point (center of map)
            self.ref_lat = (map_limits[1] + map_limits[3]) / 2.0
            self.ref_lon = (map_limits[0] + map_limits[2]) / 2.0
            
        active_uids = set()
        
        for uid, ent in entities.items():
            active_uids.add(uid)
            
            # Create node if new
            if uid not in self.nodes:
                if ent.type == "plane":
                    if ent.team == "blue":
                        # Copy the cached model
                        model = self.model_assets.get("blue_plane", self.model_assets.get("blue_plane_glb"))
                        if model:
                            model = model.copyTo(self.render)
                        else:
                            model = self.create_procedural_plane((0, 0, 1, 1))
                            model.reparentTo(self.render)
                    else:
                        model = self.model_assets.get("red_plane", self.model_assets.get("red_plane_glb"))
                        if model:
                            model = model.copyTo(self.render)
                        else:
                            model = self.create_procedural_plane((1, 0, 0, 1))
                            model.reparentTo(self.render)
                else:  # missile
                    model = self.model_assets.get("missile")
                    if model:
                        model = model.copyTo(self.render)
                    else:
                        model = self.create_procedural_missile()
                        model.reparentTo(self.render)
                    
                self.nodes[uid] = model
                
            # Update position
            # Convert lat/lon to X/Z (scale down for viewability)
            z = (ent.lat - self.ref_lat) * 111000.0 / 100.0
            x = (ent.lon - self.ref_lon) * 85000.0 / 100.0
            y = ent.alt / 100.0  # Altitude
            
            self.nodes[uid].setPos(x, z, y)
            
            # Update orientation
            # Panda3D uses HPR (Heading, Pitch, Roll) in degrees
            h = ent.heading  # Already in degrees
            p = math.degrees(ent.pitch)  # Convert from radians
            r = math.degrees(ent.roll)   # Convert from radians
            
            self.nodes[uid].setHpr(h, -p, r)  # Negative pitch for correct orientation
            
            # Track first blue plane for camera
            if ent.team == "blue" and ent.type == "plane" and self.follow_target is None:
                self.follow_target = self.nodes[uid]
                
        # Remove dead entities
        dead_uids = [uid for uid in self.nodes if uid not in active_uids]
        for uid in dead_uids:
            self.nodes[uid].removeNode()
            del self.nodes[uid]
            
        # Update camera to follow target
        if self.follow_target and self.follow_target in self.nodes.values():
            target_pos = self.follow_target.getPos()
            target_hpr = self.follow_target.getHpr()
            
            # Camera position: behind and above the plane
            cam_distance = 60
            cam_height = 30
            
            # Calculate camera position based on plane's heading
            h_rad = math.radians(target_hpr[0])
            cam_x = target_pos[0] - cam_distance * math.sin(h_rad)
            cam_y = target_pos[1] - cam_distance * math.cos(h_rad)
            cam_z = target_pos[2] + cam_height
            
            self.camera.setPos(cam_x, cam_y, cam_z)
            self.camera.lookAt(self.follow_target)
            
    def check_running(self):
        """Check if window is still open"""
        if self.win is None or not self.win.isValid():
            return False
        return self.is_running
        
    def cleanup(self):
        """Clean up resources"""
        self.destroy()
