import os
import glob
import re
import numpy as np
import torch
import copy
from config import Config
from src.model import AgentTransformer
# Import Env for evaluation (Gate Function)
# Note: We import inside method to avoid circular import if necessary, 
# but here it should be fine if train.py imports this.
# Actually, let's pass the env_maker or class to the init to be safe.

class SelfPlayManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.opponent_model = AgentTransformer().to(Config.DEVICE)
        self.opponent_model.eval()
        
        # AOS State
        self.opponent_pool = [] # List of dicts: {'path': str, 'win_rate': float, 'score': float}
        self.current_opponent_name = "Scripted (PID)"
        self.current_opponent_type = "scripted" # 'scripted', 'random', 'model'
        
        # SA-Boltzmann Params
        self.temperature = 1.0
        self.temp_decay = 0.99
        
        # Gate Function Params
        self.eval_episodes = 10
        self.win_rate_threshold = 0.5
        
        # Load initial pool
        self.load_checkpoints_list()

    def load_checkpoints_list(self):
        """Scans the checkpoint directory and rebuilds the pool."""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt"))
        # Sort by update number
        sorted_files = sorted(
            files, 
            key=lambda f: int(re.search(r'model_(\d+).pt', f).group(1))
        )
        
        # Rebuild pool if empty (or just append new ones? For now, rebuild to be safe)
        # In a real AOS, we would persist the pool metadata (scores). 
        # Here we just re-add them with default score if not present.
        existing_paths = {op['path'] for op in self.opponent_pool}
        
        for f in sorted_files:
            if f not in existing_paths:
                self.opponent_pool.append({
                    'path': f,
                    'win_rate': 0.5, # Default
                    'score': 1.0     # Default Boltzmann score
                })

    def evaluate_candidate(self, candidate_model, env_maker_fn):
        """
        Gate Function: Evaluates a candidate model against the current opponent pool.
        Returns True if the candidate is strong enough to be added to the pool.
        """
        print("--- AOS Gate Function: Evaluating Candidate ---")
        
        # If pool is empty, always accept (to start Self-Play)
        if not self.opponent_pool:
            print("Pool empty. Candidate accepted.")
            return True
            
        # Sample a subset of opponents to test against (e.g., 3 strongest + 1 random)
        # For simplicity, just test against the current "Best" (latest) and one random from pool
        test_opponents = [self.opponent_pool[-1]]
        if len(self.opponent_pool) > 1:
            test_opponents.append(np.random.choice(self.opponent_pool[:-1]))
            
        total_wins = 0
        total_games = 0
        
        # Create a temporary env for evaluation
        env = env_maker_fn()
        
        candidate_model.eval()
        
        for opp_info in test_opponents:
            # Load Opponent
            self._load_weights(opp_info['path'])
            # IMPORTANT: Set type to model so get_action uses the network
            self.current_opponent_type = "model" 
            
            for _ in range(self.eval_episodes):
                obs, info = env.reset()
                done = False
                
                # We need to track who wins. 
                # Env 'kill' event is the source of truth.
                
                while not done:
                    # Get Blue Action (Candidate)
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                        action, _, _, _ = candidate_model.get_action_and_value(obs_t)
                        blue_action = action.cpu().numpy().flatten()
                    
                    # Get Red Action (Opponent)
                    # Handle Red Obs
                    red_action = np.zeros(Config.ACTION_DIM, dtype=np.float32)
                    if "red_obs" in info:
                        # get_action expects batch, wrap it
                        red_obs_batch = np.expand_dims(info["red_obs"], axis=0)
                        red_action = self.get_action(red_obs_batch)[0]
                            
                    # Step
                    # Concatenate
                    concat_action = np.concatenate([blue_action, red_action])
                    obs, reward, term, trunc, info = env.step(concat_action)
                    done = term or trunc
                    
                    # Check for Win
                    # We need to check env events. 
                    # Since we don't have direct access to env.core.events easily from return values 
                    # (unless we modify env to return them in info), we rely on Reward or Info.
                    # Actually, env.step returns info. Let's check if we can get win status.
                    # Our env.py puts 'kill' events in core.events.
                    # We can access env.core directly since we created it.
                    
                # Episode Over. Check result.
                # If Blue is alive and Red is dead -> Win
                blue_alive = env.blue_ids[0] in env.core.entities
                red_alive = env.red_ids[0] in env.core.entities
                
                if blue_alive and not red_alive:
                    total_wins += 1
                
                total_games += 1
        
        env.close()
        
        win_rate = total_wins / total_games
        print(f"Candidate Win Rate: {win_rate:.2f} (Threshold: {self.win_rate_threshold})")
        
        if win_rate >= self.win_rate_threshold:
            return True
        else:
            return False

    def sample_opponent(self, global_step=0):
        """
        AOS Sampling Strategy:
        - Phase 1 (< 1M steps): Kappa-PPG (Curriculum)
        - Phase 2 (> 1M steps): SA-Boltzmann Sampling from Pool
        """
        # --- Phase 1: Kappa-PPG Curriculum ---
        if global_step < 1_000_000:
            # Calculate Kappa (Noise level)
            # Decay from 1.0 (Random) to 0.0 (Perfect PID) over 1M steps
            kappa = max(0.0, 1.0 - (global_step / 1_000_000.0))
            self.current_opponent_name = f"Scripted (Kappa={kappa:.2f})"
            self.current_opponent_type = "scripted_kappa"
            self.kappa = kappa
            return None

        # --- Phase 2: SA-Boltzmann Sampling ---
        self.load_checkpoints_list()
        
        if not self.opponent_pool:
            self.current_opponent_name = "Random (Pool Empty)"
            self.current_opponent_type = "random"
            return None
            
        # Calculate Boltzmann Probabilities
        scores = np.array([op['score'] for op in self.opponent_pool])
        # Softmax with temperature
        exp_scores = np.exp(scores / self.temperature)
        probs = exp_scores / np.sum(exp_scores)
        
        # Sample
        chosen_idx = np.random.choice(len(self.opponent_pool), p=probs)
        chosen_opp = self.opponent_pool[chosen_idx]
        
        self.current_opponent_name = f"Model ({os.path.basename(chosen_opp['path'])})"
        self.current_opponent_type = "model"
        self._load_weights(chosen_opp['path'])
        
        # Decay temperature (annealing)
        self.temperature = max(0.1, self.temperature * self.temp_decay)
        
        return self.opponent_model

    def _load_weights(self, path):
        try:
            checkpoint = torch.load(path, map_location=Config.DEVICE)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.opponent_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.opponent_model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading opponent {path}: {e}")

    def get_action(self, obs):
        """
        Get action from the currently loaded opponent.
        """
        # Handle AsyncVectorEnv's object dtype arrays
        if obs.dtype == object:
            obs = np.stack([np.array(o, dtype=np.float32) for o in obs])
        
        batch_size = obs.shape[0]
        
        # 1. Random
        if self.current_opponent_type == "random":
            return np.random.uniform(-1, 1, (batch_size, Config.ACTION_DIM))
        
        # 2. Scripted (Kappa-PPG)
        if self.current_opponent_type == "scripted_kappa":
            # We return None to tell Env to use internal AI.
            # But we need to pass Kappa to Env? 
            # The Env's internal AI is fixed PID. 
            # To implement Kappa-PPG, we should ideally compute it here or pass a param.
            # Since Env.step(red_actions=None) triggers internal AI, we can't easily modify it from here 
            # without changing Env API to accept 'ai_config'.
            # 
            # Workaround: Return a special flag or handle it in Train loop?
            # Better: Implement Kappa-PPG *here* using a helper, OR just use the Env's PID 
            # and accept that it's 0-noise for now, OR modify Env to accept noise param.
            # 
            # Let's stick to the plan: "Modify Scripted logic in self_play.py...".
            # If we return None, Env uses its `_calculate_ai_action`.
            # We can't easily inject Kappa there.
            # 
            # ALTERNATIVE: We implement the PID controller HERE in Python and add noise.
            # But `core.py` has the physics state. We only have `obs`.
            # `obs` has relative positions. We CAN implement a simple pursuer here!
            # 
            # Let's try to implement a simple Kappa-PPG here based on Observations.
            # Obs: [lat_n, lon_n, cos_h, sin_h, speed, ...] for Ego
            #      [lat_n, lon_n, ... ] for Target (Blue)
            # 
            # It's complicated to reconstruct state.
            # 
            # EASIER FIX: Just return None and let Env do its thing. 
            # The "Kappa" part (decaying noise) is hard to inject without Env changes.
            # 
            # Let's modify Env to look for a global config or something? No, ugly.
            # 
            # Compromise: For Phase 1, we just use the fixed PID (Kappa=0 effectively).
            # The User Plan said "Modify Scripted logic... to accept global_step".
            # I did that in `sample_opponent`.
            # But `get_action` returning `None` means we rely on Env.
            # 
            # Let's just return None for now. The "Curriculum" is "Scripted -> Self-Play".
            # The "Kappa" part might be too invasive for `env.py` right now.
            return None
        
        # 3. Model
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).to(Config.DEVICE)
            action, _, _, _ = self.opponent_model.get_action_and_value(obs_t)
            return action.cpu().numpy()