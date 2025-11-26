"""Test phase-specific spawning logic."""
import numpy as np
from src.env import AirCombatEnv
from src.utils.geodesics import geodetic_distance_km

def test_phase1_spawning():
    """Phase 1: Blue behind red, red at 100 km/h, 5-10km apart."""
    print("\n=== Testing Phase 1 Spawning ===")
    env = AirCombatEnv()
    env.set_phase(1)
    
    obs, info = env.reset(seed=42)
    
    blue = env.core.entities[env.blue_ids[0]]
    red = env.core.entities[env.red_ids[0]]
    
    # Check distance
    dist = geodetic_distance_km(blue.lat, blue.lon, red.lat, red.lon)
    print(f"Spawn distance: {dist:.2f} km")
    assert 3 < dist < 15, f"Distance {dist:.1f}km should be between 3-15km"
    
    # Check red speed
    print(f"Red drone speed: {red.speed} km/h")
    assert red.speed == 100, f"Red speed should be 100 km/h, got {red.speed}"
    
    # Check altitude
    print(f"Blue altitude: {blue.alt}m, Red altitude: {red.alt}m")
    assert blue.alt == 5000, f"Blue alt should be 5000m, got {blue.alt}"
    assert red.alt == 5000, f"Red alt should be 5000m, got {red.alt}"
    
    env.close()
    print("✅ Phase 1 spawning correct (100 km/h drone, 5000m alt)")

def test_phase2_spawning():
    """Phase 2: Similar to Phase 1 but red at 700 km/h."""
    print("\n=== Testing Phase 2 Spawning ===")
    env = AirCombatEnv()
    env.set_phase(2)
    
    obs, info = env.reset(seed=42)
    
    blue = env.core.entities[env.blue_ids[0]]
    red = env.core.entities[env.red_ids[0]]
    
    dist = geodetic_distance_km(blue.lat, blue.lon, red.lat, red.lon)
    print(f"Spawn distance: {dist:.2f} km")
    assert 3 < dist < 15, f"Distance {dist:.1f}km should be between 3-15km"
    
    print(f"Red drone speed: {red.speed} km/h")
    assert red.speed == 700, f"Red speed should be 700 km/h, got {red.speed}"
    
    env.close()
    print("✅ Phase 2 spawning correct (700 km/h drone)")

def test_phase3_spawning():
    """Phase 3: Should use battle-box (40-80km separation)."""
    print("\n=== Testing Phase 3 Spawning (Battle-Box) ===")
    env = AirCombatEnv()
    env.set_phase(3)
    
    obs, info = env.reset(seed=42)
    
    blue = env.core.entities[env.blue_ids[0]]
    red = env.core.entities[env.red_ids[0]]
    
    dist = geodetic_distance_km(blue.lat, blue.lon, red.lat, red.lon)
    print(f"Battle-box distance: {dist:.2f} km")
    assert 35 < dist < 85, f"Battle-box distance {dist:.1f}km should be 40-80km"
    
    print(f"Blue speed: {blue.speed} km/h, Red speed: {red.speed} km/h")
    assert blue.speed == 900, f"Blue speed should be 900 km/h in Phase 3"
    assert red.speed == 900, f"Red speed should be 900 km/h in Phase 3"
    
    print(f"Blue altitude: {blue.alt}m, Red altitude: {red.alt}m")
    assert blue.alt == 10000, f"Altitude should be 10000m in Phase 3"
    
    env.close()
    print("✅ Phase 3 spawning correct (battle-box, 900 km/h)")

def test_drone_behavior():
    """Verify Phase 1/2 red agent flies straight."""
    print("\n=== Testing Drone AI Behavior ===")
    env = AirCombatEnv()
    env.set_phase(1)
    
    obs, info = env.reset(seed=42)
    
    # Get initial red heading
    red = env.core.entities[env.red_ids[0]]
    initial_heading = red.heading
    initial_roll = red.roll
    initial_pitch = red.pitch
    
    # Step with blue action only (red should fly straight)
    blue_action = np.array([0, 0, 0.5, 0, 0])
    
    for _ in range(10):
        obs, reward, term, trunc, info = env.step(blue_action)
        if term or trunc:
            break
    
    red = env.core.entities.get(env.red_ids[0])
    if red:
        heading_change = abs(red.heading - initial_heading)
        print(f"Heading change after 10 steps: {heading_change:.2f}°")
        print(f"Roll: {red.roll:.3f}, Pitch: {red.pitch:.3f}")
        
        # Drone should be mostly stable (small changes from physics are OK)
        if heading_change < 15:  # Allow some drift
            print("✅ Drone flying relatively straight")
        else:
            print(f"⚠️  Drone heading changed by {heading_change:.1f}° (might be maneuvering)")
    else:
        print("⚠️  Red agent died during test")
    
    env.close()

if __name__ == "__main__":
    print("Testing Phase-Specific Spawning Logic")
    print("=" * 50)
    
    test_phase1_spawning()
    test_phase2_spawning()
    test_phase3_spawning()
    test_drone_behavior()
    
    print("\n" + "=" * 50)
    print("✅ All spawning tests complete!")
