import gymnasium as gym
import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env import AirCombatEnv
from src.self_play import SelfPlayManager
from config import Config

def verify_spawn_speed():
    print("--- Verifying Spawn Speed ---")
    env = AirCombatEnv()
    obs, info = env.reset()
    
    blue_id = env.blue_ids[0]
    speed = env.core.entities[blue_id].speed
    print(f"Spawn Speed: {speed}")
    
    assert abs(speed - 900.0) < 1.0, f"Spawn speed should be 900, got {speed}"
    print("Spawn Speed Verified.\n")

def verify_safety_override():
    print("--- Verifying Safety Override ---")
    env = AirCombatEnv()
    obs, info = env.reset()
    blue_id = env.blue_ids[0]
    
    # Force Dangerous State
    env.core.entities[blue_id].alt = 1500.0 # Below 2000m
    env.core.entities[blue_id].pitch = -0.5 # Diving
    env.core.entities[blue_id].roll = 0.5   # Banking
    
    print(f"Initial State: Alt={env.core.entities[blue_id].alt}, Pitch={env.core.entities[blue_id].pitch}")
    
    # Step with a "bad" action (e.g. continue diving)
    # Action: [Roll, G, Throttle, Fire, CM]
    bad_action = np.array([0.0, -1.0, 0.5, 0.0, 0.0], dtype=np.float32) # Push nose down
    
    obs, reward, term, trunc, info = env.step(bad_action)
    
    # Check if Override Triggered
    # 1. Reward should have penalty (-0.5)
    print(f"Reward: {reward}")
    assert reward <= -0.4, f"Reward should be penalized (approx -0.5), got {reward}"
    
    print("Safety Override Verified (Reward Penalty Confirmed).\n")

def verify_curriculum_pause():
    print("--- Verifying Curriculum Pause ---")
    sp_manager = SelfPlayManager()
    
    # Case 1: Winning -> Kappa Decays
    sp_manager.last_eval_passed = True
    sp_manager.sample_opponent(global_step=500_000)
    kappa_win = sp_manager.kappa
    print(f"Winning (Step 500k) -> Kappa: {kappa_win}")
    assert kappa_win < 0.6, f"Kappa should decay if winning, got {kappa_win}"
    
    # Case 2: Failing -> Kappa Holds/Resets
    sp_manager.last_eval_passed = False
    sp_manager.sample_opponent(global_step=500_000)
    kappa_fail = sp_manager.kappa
    print(f"Failing (Step 500k) -> Kappa: {kappa_fail}")
    assert kappa_fail >= 0.8, f"Kappa should be high (>=0.8) if failing, got {kappa_fail}"
    
    print("Curriculum Pause Verified.\n")

def verify_diagnostics():
    print("--- Verifying Diagnostics ---")
    env = AirCombatEnv()
    env.reset()
    
    # Force Crash
    blue_id = env.blue_ids[0]
    env.core.entities[blue_id].alt = -1000.0 # Crash (Deep enough to not be saved by override)
    
    # Step to trigger crash logic
    _, _, term, _, info = env.step(np.zeros(Config.ACTION_DIM))
    
    # Debug
    if blue_id in env.core.entities:
        print(f"Agent still alive! Alt: {env.core.entities[blue_id].alt}")
    else:
        print("Agent died.")
    
    print(f"Events: {env.core.events}")
    print(f"Termination Reason: {info.get('termination_reason')}")
    assert info.get('termination_reason') == "crash", "Should report crash"
    
    print("Diagnostics Verified.\n")

if __name__ == "__main__":
    verify_spawn_speed()
    verify_safety_override()
    verify_curriculum_pause()
    verify_diagnostics()
