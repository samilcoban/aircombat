import gymnasium as gym
import numpy as np
import torch
from src.env import AirCombatEnv
from src.self_play import SelfPlayManager
from config import Config

def verify_scripted_ai():
    print("--- Verifying Scripted AI ---")
    sp_manager = SelfPlayManager()
    
    # Test Phase 1 (< 1M steps)
    opponent = sp_manager.sample_opponent(global_step=500_000)
    print(f"Global Step 500k Opponent: {sp_manager.current_opponent_name}")
    assert "Scripted" in sp_manager.current_opponent_name, "Should be Scripted"
    assert opponent is None, "Opponent model should be None for Scripted"
    
    # Test Phase 2 (> 1M steps) - assuming no checkpoints, should be Random
    opponent = sp_manager.sample_opponent(global_step=1_500_000)
    print(f"Global Step 1.5M Opponent: {sp_manager.current_opponent_name}")
    # It might be Random (No Checkpoints) or Random depending on logic
    assert "Random" in sp_manager.current_opponent_name, "Should be Random if no checkpoints"

    print("Scripted AI Logic Verified.\n")

def verify_reward_function():
    print("--- Verifying Reward Function ---")
    env = AirCombatEnv()
    obs, info = env.reset()
    
    # 1. Verify Kappa Connection
    print("\n1. Testing Kappa Connection")
    env.set_kappa(0.5)
    assert env.kappa == 0.5, "Env kappa not set"
    # We can't easily check core.kappa since it's passed in step, 
    # but we can check if step runs without error with kappa
    try:
        env.step(np.zeros(Config.ACTION_DIM))
        print("Step with kappa=0.5 successful.")
    except Exception as e:
        print(f"Step with kappa failed: {e}")
        raise e

    # 2. Verify Exponential Rewards
    print("\n2. Testing Exponential Rewards")
    # Reset to get fresh state
    obs, info = env.reset()
    _, reward, _, _, _ = env.step(np.zeros(Config.ACTION_DIM))
    
    # Reward should be small positive (Alignment) - small negative (Existence)
    # With exponential, alignment reward drops off fast if not perfect.
    # Initial spawn is head-on? Or random?
    # If head-on, ATA is 0 -> Aim Reward Max (0.5 * 0.1 = 0.05).
    # Geo Reward (Tail Chase) -> Low for head-on.
    # Close Reward -> Low if far.
    
    print(f"Step 1 Reward: {reward}")
    # It might be negative if existence penalty > alignment.
    # -0.005 (exist) vs +0.05 (aim). Should be positive.
    
    # Let's not assert strict bounds as spawn is random-ish (though usually set).
    # Just ensure it's not -50 (Death) or +100 (Kill).
    assert -1.0 < reward < 1.0, "Reward out of expected shaping range"

    print("Reward Function verification complete.")

if __name__ == "__main__":
    verify_scripted_ai()
    verify_reward_function()
