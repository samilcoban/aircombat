import numpy as np
import torch
from src.self_play import SelfPlayManager
from src.env import AirCombatEnv
from src.model import AgentTransformer
from config import Config

def make_env():
    return AirCombatEnv()

def verify_aos():
    print("--- Verifying AOS Framework ---")
    sp_manager = SelfPlayManager()
    
    # 1. Verify Kappa-PPG Sampling
    print("\n1. Testing Kappa-PPG (Phase 1)")
    opp = sp_manager.sample_opponent(global_step=500_000)
    print(f"Opponent: {sp_manager.current_opponent_name}")
    assert "Scripted (Kappa=0.50)" in sp_manager.current_opponent_name, "Kappa should be 0.5 at 500k steps"
    assert opp is None, "Opponent should be None for Scripted"
    
    # 2. Verify Gate Function
    print("\n2. Testing Gate Function")
    # Mock a candidate model
    candidate = AgentTransformer().to(Config.DEVICE)
    
    # Mock the pool to have at least one opponent so evaluation runs
    # We can manually inject a dummy entry if pool is empty
    if not sp_manager.opponent_pool:
        print("Pool empty, injecting dummy for test.")
        sp_manager.opponent_pool.append({
            'path': 'dummy_path.pt', 
            'win_rate': 0.5, 
            'score': 1.0
        })
        # We need to mock _load_weights to not fail on dummy path
        original_load = sp_manager._load_weights
        sp_manager._load_weights = lambda path: None # No-op
    
    # Run evaluation (should run but might fail/pass depending on random weights)
    # We just want to ensure it doesn't crash
    try:
        result = sp_manager.evaluate_candidate(candidate, make_env)
        print(f"Evaluation Result: {result}")
    except Exception as e:
        print(f"Evaluation Failed with error: {e}")
        raise e
    
    # 3. Verify SA-Boltzmann Sampling
    print("\n3. Testing SA-Boltzmann (Phase 2)")
    # Force Phase 2
    sp_manager.sample_opponent(global_step=2_000_000)
    print(f"Opponent: {sp_manager.current_opponent_name}")
    # Should be "Model (...)" or "Random (Pool Empty)" if we injected dummy but load failed?
    # Since we mocked load, it might be "Model (dummy_path.pt)"
    
    print("\nAOS Verification Complete.")

if __name__ == "__main__":
    verify_aos()
