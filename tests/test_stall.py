import unittest
from src.core import AirCombatCore, Entity
from config import Config

class TestStallPhysics(unittest.TestCase):
    def setUp(self):
        self.core = AirCombatCore()

    def create_entity(self, uid, lat, lon, heading, speed, alt=10000.0):
        e = Entity(
            uid=uid, team="blue", type="plane",
            lat=lat, lon=lon, alt=alt,
            heading=heading, speed=speed
        )
        self.core.entities[uid] = e
        return e

    def test_stall_behavior(self):
        # 1. Normal Flight (Speed > 150)
        # 300 knots, Level flight (Pitch 0)
        e1 = self.create_entity(1, 0, 0, 0, 300.0)
        e1.pitch = 0.0
        
        # 2. Stalled Flight (Speed < 150)
        # 100 knots, Level flight attempt (Pitch 0)
        e2 = self.create_entity(2, 0, 0.1, 0, 100.0)
        e2.pitch = 0.0
        
        # Step physics
        # Mock actions (noop)
        actions = {
            1: [0, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0]
        }
        
        initial_alt1 = e1.alt
        initial_alt2 = e2.alt
        
        self.core.step(actions)
        
        # Check Results
        # E1 should maintain altitude (approx)
        self.assertAlmostEqual(e1.alt, initial_alt1, delta=1.0, msg="Normal flight should maintain altitude")
        
        # E2 should drop significantly
        self.assertLess(e2.alt, initial_alt2 - 10.0, msg="Stalled plane should lose altitude rapidly")
        
        # E2 pitch should drop (nose down)
        self.assertLess(e2.pitch, 0.0, msg="Stalled plane should pitch down")

if __name__ == '__main__':
    unittest.main()
