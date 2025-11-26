#!/usr/bin/env python3
"""
Test script to verify smooth stall physics.
Checks that descent rate transitions smoothly instead of having a discontinuous cliff.
"""

import numpy as np
import sys
sys.path.insert(0, '/media/samil/hdd/pythonfiles/learn_pytorch/rl/aircombat')

from src.core import AirCombatCore

def test_stall_smoothness():
    """Test that stall physics provide smooth gradients"""
    print("=" * 70)
    print("STALL SMOOTHNESS TEST")
    print("=" * 70)
    print("\nThis test verifies that stall physics transition smoothly")
    print("instead of creating discontinuous 'gradient cliffs'.\n")
    
    # Test speeds from well above stall to well below
    test_speeds = [200, 175, 150, 140, 130, 120, 110, 100, 90]
    
    print(f"{'Speed (kts)':>12} | {'Alt Start':>10} | {'Alt End':>10} | {'Descent Rate':>15} | {'Gradient':>10}")
    print("-" * 70)
    
    prev_descent_rate = None
    
    for speed in test_speeds:
        # Create fresh simulation
        core = AirCombatCore()
        
        # Spawn aircraft at test speed
        uid = core.spawn(lat=0.0, lon=0.0, heading=0.0, speed=speed, team="blue", etype="plane")
        ent = core.entities[uid]
        
        # Record initial altitude
        initial_alt = ent.alt
        
        # Run simulation for 1 second with neutral action (maintain speed/altitude attempt)
        # Action: [roll=0, g=0 (1.0G level), throttle=0.6 (80%), fire=0, cm=0]
        action = [0.0, 0.0, 0.6, 0.0, 0.0]
        
        for _ in range(20):  # 20 steps * 0.05s = 1.0 second
            if uid in core.entities:  # Check if not crashed
                core.entities[uid].pitch = 0.0  # Force level flight attitude
                core._update_plane_physics(core.entities[uid], action, execute_discrete_actions=False)
        
        # Calculate descent rate
        if uid in core.entities:
            final_alt = core.entities[uid].alt  # Get altitude value, not entity
            descent_rate = final_alt - initial_alt  # Altitude change over 1 second
            
            # Calculate gradient (change in descent rate between speed steps)
            if prev_descent_rate is not None:
                gradient = descent_rate - prev_descent_rate
                gradient_str = f"{gradient:+.2f} m/s²"
            else:
                gradient_str = "N/A"
            
            print(f"{speed:>12.0f} | {initial_alt:>10.1f} | {final_alt:>10.1f} | {descent_rate:>+15.2f} m/s | {gradient_str:>10}")
            
            prev_descent_rate = descent_rate
        else:
            print(f"{speed:>12.0f} | {'CRASHED':>10} | {'CRASHED':>10} | {'CRASHED':>15} | {'N/A':>10}")
    
    print("\n" + "=" * 70)
    print("EXPECTED BEHAVIOR:")
    print("  - Descent rate should change GRADUALLY as speed decreases")
    print("  - No sudden jumps in 'Gradient' column")
    print("  - Smooth transition from positive (climbing) to negative (descending)")
    print("=" * 70)

def test_throttle_control():
    """Test that agent's throttle action actually affects physics"""
    print("\n" + "=" * 70)
    print("THROTTLE CONTROL TEST")
    print("=" * 70)
    print("\nThis test verifies that the agent's throttle action")
    print("actually controls engine thrust (not hardcoded).\n")
    
    # Test different throttle settings
    throttle_tests = [
        ("Idle", -1.0, 0.0),      # Min throttle
        ("Cruise", 0.0, 0.5),     # 50% throttle
        ("Military", 0.6, 0.8),   # 80% throttle
        ("Max", 1.0, 1.0),        # Full throttle
    ]
    
    print(f"{'Setting':>10} | {'Action[2]':>10} | {'Expected':>10} | {'Final Speed':>12} | {'Accel':>10} | {'Status':>8}")
    print("-" * 70)
    
    for name, action_throttle, expected_throttle in throttle_tests:
        # Create fresh simulation
        core = AirCombatCore()
        uid = core.spawn(lat=0.0, lon=0.0, heading=0.0, speed=300.0, team="blue", etype="plane")
        
        initial_speed = core.entities[uid].speed
        
        # Run with this throttle setting
        action = [0.0, 0.0, action_throttle, 0.0, 0.0]
        
        for _ in range(20):  # 1 second
            if uid in core.entities:
                core._update_plane_physics(core.entities[uid], action, execute_discrete_actions=False)
        
        if uid in core.entities:
            final_speed = core.entities[uid].speed
            speed_change = final_speed - initial_speed
            
            # Check if speed changed in expected direction
            if expected_throttle > 0.5 and speed_change > 0:
                status = "✓ PASS"
            elif expected_throttle < 0.5 and speed_change < 0:
                status = "✓ PASS"
            elif abs(speed_change) < 5:  # Near equilibrium
                status = "~ NEUT"
            else:
                status = "✗ FAIL"
            
            print(f"{name:>10} | {action_throttle:>10.2f} | {expected_throttle:>10.2f} | {final_speed:>12.1f} | {speed_change:>+10.1f} | {status:>8}")
        else:
            print(f"{name:>10} | {action_throttle:>10.2f} | {expected_throttle:>10.2f} | {'CRASHED':>12} | {'N/A':>10} | {'✗ FAIL':>8}")
    
    print("\n" + "=" * 70)
    print("EXPECTED BEHAVIOR:")
    print("  - Higher throttle settings should increase speed")
    print("  - Lower throttle settings should decrease speed")
    print("  - All tests should show PASS or NEUT status")
    print("=" * 70)

if __name__ == "__main__":
    test_stall_smoothness()
    test_throttle_control()
    print("\n✓ Tests complete!")
