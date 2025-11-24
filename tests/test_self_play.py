import unittest
import numpy as np
import torch
from src.env import AirCombatEnv
from src.self_play import SelfPlayManager
from config import Config

class TestSelfPlay(unittest.TestCase):
    def test_self_play_integration(self):
        # 1. Init Manager
        sp_manager = SelfPlayManager()
        print(f"Opponent: {sp_manager.current_opponent_name}")
        
        # 2. Init Env
        env = AirCombatEnv()
        obs, info = env.reset()
        
        # 3. Step Loop
        for _ in range(10):
            # Blue Action (Random)
            blue_action = np.random.uniform(-1, 1, Config.ACTION_DIM)
            
            # Red Action (From Manager)
            if "red_obs" in info:
                # Manager expects batch, so unsqueeze
                red_obs_batch = np.expand_dims(info["red_obs"], axis=0)
                red_action_batch = sp_manager.get_action(red_obs_batch)
                red_action = red_action_batch[0]
            else:
                red_action = np.zeros(Config.ACTION_DIM)
                
            # Concatenate
            concat_action = np.concatenate([blue_action, red_action])
            
            # Step
            obs, reward, term, trunc, info = env.step(concat_action)
            
            if term or trunc:
                break
                
        print("Self-Play Test Loop Completed Successfully")

if __name__ == '__main__':
    unittest.main()
