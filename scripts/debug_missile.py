#!/usr/bin/env python3
import sys
import os
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core_flat import AirCombatCore


def test_missile_flight():
    core = AirCombatCore()

    # 1. Spawn Shooter and Target
    # Shooter at 0, Target at 2000. Both heading East (PI/2).
    # Using Radians!
    h_east = math.pi / 2.0

    sid = core.spawn(0, 0, 5000, h_east, 600, "blue", "plane")
    tid = core.spawn(0, 2000, 5000, h_east, 600, "red", "plane")

    shooter = core.entities[sid]
    target = core.entities[tid]

    # 2. Fire Missile
    # Manually call the internal spawn logic to ensure we control it
    mid = core.spawn(shooter.x, shooter.y, shooter.alt, shooter.heading, shooter.speed, "blue", "missile")
    missile = core.entities[mid]
    missile.target_id = tid
    missile.owner_id = sid
    missile.pitch = shooter.pitch

    print(f"INIT: M_Pos=({missile.x:.0f},{missile.y:.0f}) T_Pos=({target.x:.0f},{target.y:.0f})")

    # 3. Step Physics
    for i in range(50):
        # We must simulate the core loop manually to update missiles
        # core.step clears events, so we check them after

        # Move planes (fly straight)
        # We can just let them drift or apply simple physics
        # Let's apply a dummy action to keep them moving
        actions = {
            sid: np.array([0, 0, 1.0, 0, 0]),  # Full throttle
            tid: np.array([0, 0, 1.0, 0, 0])
        }

        core.step(actions)

        if mid not in core.entities:
            print(f"STEP {i}: MISSILE DIED/GONE!")
            # Check events
            for e in core.events:
                print(f"EVENT: {e}")
            break

        m = core.entities[mid]
        t = core.entities[tid]
        dist = math.hypot(t.x - m.x, t.y - m.y)

        print(f"STEP {i}: M_Y={m.y:.0f} T_Y={t.y:.0f} Dist={dist:.0f} Speed={m.speed:.0f}")


if __name__ == "__main__":
    test_missile_flight()