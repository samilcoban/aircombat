import unittest
import math
from src.core import AirCombatCore, Entity
from config import Config

class TestSensors(unittest.TestCase):
    def setUp(self):
        self.core = AirCombatCore()
        # Mock Config for deterministic testing
        self.core.cfg.RADAR_RANGE_KM = 100.0
        self.core.cfg.RADAR_FOV_DEG = 60.0
        self.core.cfg.RADAR_NOTCH_SPEED_KNOTS = 10.0

    def create_entity(self, uid, lat, lon, heading, speed, team="blue"):
        e = Entity(
            uid=uid, team=team, type="plane",
            lat=lat, lon=lon, alt=10000.0,
            heading=heading, speed=speed
        )
        self.core.entities[uid] = e
        return e

    def test_radar_range(self):
        # Observer at (0,0)
        obs = self.create_entity(1, 0, 0, 0, 600, "blue")
        
        # Target 1: Inside Range (approx 50km North)
        # 1 deg lat approx 111km. 0.45 deg approx 50km
        tgt1 = self.create_entity(2, 0.45, 0, 180, 600, "red")
        
        # Target 2: Outside Range (approx 150km North)
        tgt2 = self.create_entity(3, 1.5, 0, 180, 600, "red")

        visible1, _ = self.core.get_sensor_state(1, 2)
        visible2, _ = self.core.get_sensor_state(1, 3)

        self.assertTrue(visible1, "Target within range should be visible")
        self.assertFalse(visible2, "Target outside range should not be visible")

    def test_radar_fov(self):
        # Observer facing North (0)
        obs = self.create_entity(1, 0, 0, 0, 600, "blue")

        # Target 1: Inside FOV (Directly North)
        tgt1 = self.create_entity(2, 0.1, 0, 180, 600, "red")

        # Target 2: Outside FOV (East, 90 deg bearing)
        tgt2 = self.create_entity(3, 0, 0.1, 180, 600, "red")

        visible1, _ = self.core.get_sensor_state(1, 2)
        visible2, _ = self.core.get_sensor_state(1, 3)

        self.assertTrue(visible1, "Target in FOV should be visible")
        self.assertFalse(visible2, "Target outside FOV should not be visible")

    def test_doppler_notch(self):
        # Observer facing North
        obs = self.create_entity(1, 0, 0, 0, 600, "blue")

        # Target 1: Hot Aspect (Flying South towards Observer) -> High Closure
        tgt1 = self.create_entity(2, 0.1, 0, 180, 600, "red")

        # Target 2: Notching (Flying East, Perpendicular to Observer) -> Low Radial Speed
        # Bearing to Obs is South (180). Target Heading is East (90). Aspect = 90.
        tgt2 = self.create_entity(3, 0.1, 0, 90, 600, "red")

        visible1, _ = self.core.get_sensor_state(1, 2)
        visible2, _ = self.core.get_sensor_state(1, 3)

        self.assertTrue(visible1, "Hot aspect target should be visible")
        self.assertFalse(visible2, "Notching target should be invisible (Doppler Filter)")

    def test_rwr_locking(self):
        # Observer (Blue) facing North
        obs = self.create_entity(1, 0, 0, 0, 600, "blue")
        
        # Enemy (Red) facing South (Locking Blue)
        enemy = self.create_entity(2, 0.1, 0, 180, 600, "red")

        # Check if Enemy is locking Blue
        # get_sensor_state(Observer=Enemy, Target=Blue)
        visible, locking = self.core.get_sensor_state(2, 1)

        self.assertTrue(visible)
        self.assertTrue(locking)

if __name__ == '__main__':
    unittest.main()
