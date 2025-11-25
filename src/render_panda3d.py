from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import math

class Panda3DRenderer(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
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
        
    def create_plane_model(self, color):
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
        
    def create_missile_model(self):
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
                        model = self.create_plane_model((0, 0, 1, 1))  # Blue
                    else:
                        model = self.create_plane_model((1, 0, 0, 1))  # Red
                else:  # missile
                    model = self.create_missile_model()
                    
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
