"""Test reward magnitudes for each phase."""
import numpy as np
from src.env import AirCombatEnv

def simulate_phase(phase_id, max_steps=1200):
    """
    Simulate a stable flight episode and accumulate rewards.
    
    Returns total reward to verify it meets phase expectations.
    """
    env = AirCombatEnv()
    env.set_phase(phase_id)
    obs, info = env.reset(seed=42)
    
    total_reward = 0.0
    step_count = 0
    
    for _ in range(max_steps):
        # Simple stable flight action: level wings, gentle pull, medium throttle
        action = np.array([0.0, 0.2, 0.6, 0.0, 0.0])
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if term or trunc:
            break
    
    env.close()
    return total_reward, step_count

def test_phase1_rewards():
    """Phase 1 should give ~12-15 total (6 survival + 6 approach + up to 3 stability)."""
    print("\n=== Testing Phase 1 Rewards ===")
    total, steps = simulate_phase(1, max_steps=1200)
    print(f"Total reward: {total:.2f} over {steps} steps")
    print(f"Average per step: {total/steps:.4f}")
    
    # Phase 1 target: 12-15 total
    if 8 < total < 20:
        print("✅ Phase 1 reward magnitude looks good (~12-15 expected)")
    else:
        print(f"⚠️  Phase 1 reward {total:.2f} outside expected range (8-20)")
    
    return total

def test_phase2_rewards():
    """Phase 2 should give Phase 1 rewards + positioning (~20 total)."""
    print("\n=== Testing Phase 2 Rewards ===")
    total, steps = simulate_phase(2, max_steps=1200)
    print(f"Total reward: {total:.2f} over {steps} steps")
    print(f"Average per step: {total/steps:.4f}")
    
    # Phase 2 target: ~20 total
    if 15 < total < 30:
        print("✅ Phase 2 reward magnitude looks good (~20 expected)")
    else:
        print(f"⚠️  Phase 2 reward {total:.2f} outside expected range (15-30)")
    
    return total

def test_phase3_combat():
    """Phase 3 should have all rewards active."""
    print("\n=== Testing Phase 3 Rewards ===")
    total, steps = simulate_phase(3, max_steps=500)
    print(f"Total reward: {total:.2f} over {steps} steps")
    print("Note: Phase 3 includes kill rewards (+50), so total can vary greatly")
    
    if total > -50:  # Should at least survive a bit
        print("✅ Phase 3 agent survived some steps")
    else:
        print("⚠️  Phase 3 agent crashed immediately")
    
    return total

if __name__ == "__main__":
    print("Testing Phase-Gated Reward Magnitudes")
    print("=" * 50)
    
    r1 = test_phase1_rewards()
    r2 = test_phase2_rewards()
    r3 = test_phase3_combat()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Phase 1: {r1:.2f} (target: 12-15)")
    print(f"  Phase 2: {r2:.2f} (target: ~20)")
    print(f"  Phase 3: {r3:.2f} (variable, includes combat)")
    print("\n✅ Reward magnitude testing complete!")
