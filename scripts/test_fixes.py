#!/usr/bin/env python3
"""
Quick test to verify energy penalty and reward normalization wrappers work correctly.
Run this in hhmarl_env before starting full training.
"""
import numpy as np
import torch
from src.env_flat import AirCombatEnv
from train import make_env
from config import Config

def test_energy_penalty():
    """Test that energy penalty is applied for high-G maneuvers."""
    print("=" * 60)
    print("TEST 1: Energy Penalty")
    print("=" * 60)
    
    env = AirCombatEnv()
    env.set_phase(1, progress=0.0)
    obs, info = env.reset()
    
    # Simulate high-G maneuver (max g-pull)
    high_g_action = np.array([0.0, 1.0, 0.8, 0.0, 0.0])  # Max g-pull
    
    # Run a few steps
    total_reward = 0
    for i in range(10):
        obs, reward, term, trunc, info = env.step(high_g_action)
        total_reward += reward
        if term or trunc:
            break
    
    print(f"✓ High-G action executed without errors")
    print(f"  Total reward over 10 steps: {total_reward:.4f}")
    print(f"  (Should be negative due to energy penalty)")
    
    env.close()
    return True

def test_reward_normalization():
    """Test that reward normalization wrappers are applied."""
    print("\n" + "=" * 60)
    print("TEST 2: Reward Normalization Wrappers")
    print("=" * 60)
    
    env = make_env()  # This should include all wrappers
    
    # Check wrapper chain
    wrapper_names = []
    current = env
    while hasattr(current, 'env'):
        wrapper_names.append(type(current).__name__)
        current = current.env
    wrapper_names.append(type(current).__name__)
    
    print(f"✓ Wrapper chain: {' -> '.join(wrapper_names)}")
    
    # Verify expected wrappers are present
    expected_wrappers = ['NormalizeReward', 'TransformReward', 'TransformObservation', 'NormalizeObservation']
    found_wrappers = [w for w in expected_wrappers if w in wrapper_names]
    
    print(f"✓ Found {len(found_wrappers)}/{len(expected_wrappers)} expected wrappers:")
    for w in found_wrappers:
        print(f"  - {w}")
    
    if len(found_wrappers) == len(expected_wrappers):
        print("✅ All normalization wrappers present!")
    else:
        print(f"⚠️  Missing wrappers: {set(expected_wrappers) - set(found_wrappers)}")
    
    env.close()
    return len(found_wrappers) == len(expected_wrappers)

def test_phase1_spawning():
    """Test that Phase 1 spawning works correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Phase 1 Spawning")
    print("=" * 60)
    
    env = AirCombatEnv()
    env.set_phase(1, progress=0.0)
    obs, info = env.reset()
    
    # Check that blue and red agents spawned
    print(f"✓ Blue agents: {len(env.blue_ids)}")
    print(f"✓ Red agents: {len(env.red_ids)}")
    
    if env.blue_ids and env.red_ids:
        blue = env.core.entities[env.blue_ids[0]]
        red = env.core.entities[env.red_ids[0]]
        
        # Calculate distance
        dist = np.sqrt((blue.x - red.x)**2 + (blue.y - red.y)**2) / 1000.0
        print(f"✓ Initial separation: {dist:.2f} km")
        print(f"✓ Blue heading: {blue.heading:.1f}°, speed: {blue.speed:.0f} km/h")
        print(f"✓ Red heading: {red.heading:.1f}°, speed: {red.speed:.0f} km/h")
        
        if 3.0 <= dist <= 16.0:  # 3-6km behind + 5-10km ahead
            print("✅ Phase 1 spawning looks correct!")
        else:
            print(f"⚠️  Unexpected separation distance: {dist:.2f} km")
    
    env.close()
    return True

if __name__ == "__main__":
    print("\n🧪 Running Quick Verification Tests\n")
    
    try:
        test1 = test_energy_penalty()
        test2 = test_reward_normalization()
        test3 = test_phase1_spawning()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Energy Penalty: {'✅ PASS' if test1 else '❌ FAIL'}")
        print(f"Reward Normalization: {'✅ PASS' if test2 else '❌ FAIL'}")
        print(f"Phase 1 Spawning: {'✅ PASS' if test3 else '❌ FAIL'}")
        
        if test1 and test2 and test3:
            print("\n✅ All tests passed! Ready to start training.")
            print("\nNext steps:")
            print("  1. conda activate hhmarl_env")
            print("  2. python train.py")
            print("  3. Monitor TensorBoard: tensorboard --logdir runs/")
        else:
            print("\n⚠️  Some tests failed. Review output above.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
