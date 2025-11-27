"""Test evaluation logic selects correct opponent for Phase 1."""
import sys
import os
sys.path.append(os.getcwd())

from src.self_play import SelfPlayManager
from unittest.mock import MagicMock

def test_evaluation_selection():
    print("Testing Evaluation Opponent Selection...")
    
    # Mock dependencies
    sp = SelfPlayManager()
    candidate_model = MagicMock()
    
    # Mock Env
    mock_env = MagicMock()
    import numpy as np
    dummy_obs = np.zeros(15, dtype=np.float32) # Assuming OBS_DIM=15
    mock_env.reset.return_value = (dummy_obs, {}) 
    mock_env.step.return_value = (dummy_obs, 0, True, False, {}) 
    
    def env_maker():
        return mock_env
    
    # Test Phase 1 (Global Step 0)
    print("\n--- Testing Phase 1 (Step 0) ---")
    sp.evaluate_candidate(candidate_model, env_maker, global_step=0)
    
    # Verify set_phase was called with 1
    mock_env.set_phase.assert_called_with(1)
    print("✅ Env.set_phase(1) called correctly")
    
    # Verify opponent type
    if sp.current_opponent_type == "stable_drone":
        print("✅ Opponent type set to 'stable_drone'")
    else:
        print(f"❌ Opponent type is {sp.current_opponent_type}")

    # Test Phase 3 (Step 3M)
    print("\n--- Testing Phase 3 (Step 3,000,000) ---")
    # Add a dummy opponent to pool so it doesn't return early
    sp.opponent_pool.append({'path': 'dummy.pt', 'win_rate': 0.5})
    
    sp.evaluate_candidate(candidate_model, env_maker, global_step=3_000_000)
    
    # Verify set_phase was called with 3
    mock_env.set_phase.assert_called_with(3)
    print("✅ Env.set_phase(3) called correctly")
    
    if sp.current_opponent_type == "model":
        print("✅ Opponent type set to 'model'")
    else:
        print(f"❌ Opponent type is {sp.current_opponent_type}")

if __name__ == "__main__":
    test_evaluation_selection()
