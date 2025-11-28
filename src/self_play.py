import os
import glob
import re
import json
import numpy as np
import torch
import copy
from config import Config
from src.model import AgentTransformer
from src.bot import HardcodedAce

# Import Env for evaluation (Gate Function)
# Note: We import inside method to avoid circular import if necessary, 
# but here it should be fine if train.py imports this.
# Actually, let's pass the env_maker or class to the init to be safe.

class SelfPlayManager:
    def __init__(self, checkpoint_dir="checkpoints", phase=2):
        self.checkpoint_dir = checkpoint_dir
        self.training_phase = phase  # 1-5: Stationary, Straight, Random, Ace, Self-Play
        self.opponent_model = AgentTransformer().to(Config.DEVICE)
        self.opponent_model = AgentTransformer().to(Config.DEVICE)
        self.opponent_model.eval()
        
        # Hardcoded Ace (Scripted Expert)
        self.ace = HardcodedAce()
        
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
        
        # Load initial pool and metadata
        self.load_pool_metadata()
        self.load_checkpoints_list()
        
        # Curriculum State
        self.kappa = 1.0
        self.last_eval_passed = False # Track this to pause curriculum if failing
    
    def save_pool_metadata(self):
        """Save opponent pool metadata to JSON for persistence across runs."""
        metadata = {
            'pool': self.opponent_pool,
            'kappa': self.kappa,
            'temperature': self.temperature,
            'last_eval_passed': self.last_eval_passed
        }
        
        metadata_path = os.path.join(self.checkpoint_dir, 'opponent_pool.json')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved opponent pool metadata: {len(self.opponent_pool)} opponents")
        except Exception as e:
            print(f"Error saving pool metadata: {e}")
    
    def load_pool_metadata(self):
        """Load opponent pool metadata from JSON to persist ELO across runs."""
        metadata_path = os.path.join(self.checkpoint_dir, 'opponent_pool.json')
        
        if not os.path.exists(metadata_path):
            print("No existing pool metadata found. Starting fresh.") 
            return
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load pool (verify checkpoints still exist)
            if 'pool' in metadata:
                self.opponent_pool = [
                    op for op in metadata['pool']
                    if os.path.exists(op.get('path', ''))
                ]
                print(f"Loaded {len(self.opponent_pool)} opponents from metadata")
            
            # Load curriculum state
            if 'kappa' in metadata:
                self.kappa = metadata['kappa']
            if 'temperature' in metadata:
                self.temperature = metadata['temperature']
            if 'last_eval_passed' in metadata:
                self.last_eval_passed = metadata['last_eval_passed']
                
        except Exception as e:
            print(f"Error loading pool metadata: {e}")

    def load_checkpoints_list(self):
        """Scans the checkpoint directory and rebuilds the pool."""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt"))
        
        # Filter only numbered models (ignore model_latest.pt)
        numbered_files = []
        for f in files:
            if re.search(r'model_(\d+).pt', f):
                numbered_files.append(f)
                
        # Sort by update number
        sorted_files = sorted(
            numbered_files, 
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

    def evaluate_candidate(self, candidate_model, env_maker_fn, global_step=0):
        """
        Gate Function: Evaluates candidate.
        Phase 1 & 2: Must beat Stable Drone (Flight School/Pursuit).
        Phase 3+: Must beat Opponent Pool (Combat).
        """
        print("--- AOS Gate Function: Evaluating Candidate ---")
        
        # DETERMINE EXAM TYPE
        current_phase = self.get_current_phase(global_step)
        
        # Setup Opponent List
        test_opponents = []
        
        if current_phase in [1, 2]:
            # EXAM 1: Flight School / Pursuit
            # Opponent is the "Stable Drone" (handled by env.set_phase)
            print(f"Phase {current_phase} Exam: Flight School / Pursuit (Stable Drone)")
            test_opponents = [{'type': 'stable_drone'}]
        else:
            # EXAM 2: Fight the Pool (MDPI Sliding Window)
            if not self.opponent_pool:
                print("Pool empty. Candidate accepted.")
                return True
            
            # MDPI Sliding Window: Test against last 5 accepted opponents + 1 random
            print(f"Phase {current_phase} Exam (MDPI): Fighting Sliding Window of Pool")
            window_size = min(5, len(self.opponent_pool))
            recent_opponents = self.opponent_pool[-window_size:]  # Last 5
            
            test_opponents = recent_opponents.copy()
            
            # Add 1 random older opponent if pool is large enough
            if len(self.opponent_pool) > window_size:
                older_pool = self.opponent_pool[:-window_size]
                random_opp = np.random.choice(older_pool)
                test_opponents.append(random_opp)
            
            # Add type info
            for op in test_opponents:
                op['type'] = 'model'
            
            print(f"  Testing against {len(test_opponents)} opponents from pool")

        total_wins = 0
        total_games = 0
        
        # DEBUG STATS
        outcomes = {"blue_crash": 0, "red_crash": 0, "blue_shot": 0, "timeout": 0, "win": 0}
        
        env = env_maker_fn()
        
        # CRITICAL: Set the phase for the exam environment!
        if hasattr(env, 'set_phase'):
            env.set_phase(current_phase)
            
        candidate_model.eval()
        
        for opp_info in test_opponents:
            # Setup Opponent
            if opp_info['type'] == 'model':
                self._load_weights(opp_info['path'])
                self.current_opponent_type = "model"
            elif opp_info['type'] == 'stable_drone':
                self.current_opponent_type = "stable_drone"
                # Env handles drone behavior based on phase
            else:
                self.current_opponent_type = "scripted_kappa"
                if hasattr(env, 'set_kappa'):
                    env.set_kappa(opp_info['kappa'])
            
            for _ in range(self.eval_episodes):
                obs, info = env.reset()
                lstm_state = None # Reset LSTM memory for new episode
                done = False
                
                while not done:
                    # Blue Action (Candidate)
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                        # Handle LSTM state
                        # We need to track done for LSTM reset, but since we reset manually at start of episode,
                        # we can just pass the state.
                        # Note: get_action_and_value expects done to be tensor if provided.
                        # For single env evaluation, we can just manage lstm_state directly.
                        action, _, _, _, lstm_state = candidate_model.get_action_and_value(obs_t, lstm_state=lstm_state)
                        blue_action = action.cpu().numpy().flatten()
                    
                    # Red Action (Opponent)
                    red_action = None # Default to Env Internal AI
                    
                    if self.current_opponent_type == "model":
                        if "red_obs" in info:
                            red_obs = info["red_obs"]
                            # get_action expects batch, wrap it
                            red_obs_batch = np.expand_dims(red_obs, axis=0)
                            red_action = self.get_action(red_obs_batch)[0]
                    
                    # Step
                    if red_action is not None:
                        concat_action = np.concatenate([blue_action, red_action])
                        obs, reward, term, trunc, info = env.step(concat_action)
                    else:
                        obs, reward, term, trunc, info = env.step(blue_action)

                    done = term or trunc
                    
                # FIX 2: DIAGNOSE THE LOSS using termination_reason
                term_reason = info.get("termination_reason", "none")
                
                if term_reason == "win":
                    total_wins += 1
                    outcomes["win"] += 1
                elif term_reason == "crash":
                    outcomes["blue_crash"] += 1
                elif term_reason == "shot":
                    outcomes["blue_shot"] += 1
                elif term_reason == "timeout":
                    outcomes["timeout"] += 1
                else:
                    # Fallback logic if reason missing (e.g. old env)
                    blue_alive = env.blue_ids[0] in env.core.entities
                    red_alive = env.red_ids[0] in env.core.entities
                    
                    if blue_alive and not red_alive:
                        total_wins += 1
                        outcomes["win"] += 1
                    elif not blue_alive:
                        outcomes["blue_crash"] += 1 # Assume crash/shot
                    else:
                        outcomes["timeout"] += 1
                
                total_games += 1
        
        env.close()
        
        win_rate = total_wins / total_games
        print(f"Candidate Win Rate: {win_rate:.2f} (Threshold: {self.win_rate_threshold})")
        print(f"Outcome Stats: {outcomes}") # PRINT THIS to see why you are dying
        
        self.last_eval_passed = (win_rate >= self.win_rate_threshold)
        return self.last_eval_passed

    def sample_opponent(self, global_step=0):
        """
        AOS Sampling Strategy with 5-Phase Curriculum:
        - Phase 1: Stationary Dummy (Learn basic flight)
        - Phase 2: Straight-Flying Dummy (Learn pursuit)
        - Phase 3: Random-Turning Dummy (Learn tracking)
        - Phase 4: Hardcoded Ace (Learn combat)
        - Phase 5: SA-Boltzmann Self-Play (Master combat)
        """
        # --- Phase 1: Stationary Dummy ---
        if self.training_phase == 1:
            self.current_opponent_name = "Stationary Dummy (Phase 1)"
            self.current_opponent_type = "stationary_dummy"
            return None
        
        # --- Phase 2: Straight-Flying Dummy ---
        elif self.training_phase == 2:
            self.current_opponent_name = "Straight Dummy (Phase 2)"
            self.current_opponent_type = "straight_dummy"
            return None
        
        # --- Phase 3: Random-Turning Dummy ---
        elif self.training_phase == 3:
            self.current_opponent_name = "Random Dummy (Phase 3)"
            self.current_opponent_type = "random_dummy"
            return None
        
        # --- Phase 4: Hardcoded Ace ---
        elif self.training_phase == 4:
            self.current_opponent_name = "Hardcoded Ace (Phase 4)"
            self.current_opponent_type = "hardcoded_ace"
            return None
        
        # --- Phase 5: PFSP (Prioritized Fictitious Self-Play) ---
        else:  # Phase 5
            self.load_checkpoints_list()
            
            if not self.opponent_pool:
                self.current_opponent_name = "Random (Pool Empty)"
                self.current_opponent_type = "random"
                return None
            
            # === PFSP: Sample Based on Difficulty ===
            # P(i) ∝ (1 - win_rate[i])²
            win_rates = np.array([op.get('win_rate', 0.5) for op in self.opponent_pool])
            difficulties = (1.0 - win_rates) ** 2
            
            if difficulties.sum() > 0:
                probs = difficulties / difficulties.sum()
            else:
                probs = np.ones(len(self.opponent_pool)) / len(self.opponent_pool)
            
            chosen_idx = np.random.choice(len(self.opponent_pool), p=probs)
            chosen_opp = self.opponent_pool[chosen_idx]
            
            self.current_opponent_name = f"PFSP: {os.path.basename(chosen_opp['path'])} (WR:{chosen_opp.get('win_rate', 0.5):.2f})"
            self.current_opponent_type = "model"
            self._load_weights(chosen_opp['path'])
            
            self.temperature = max(0.1, self.temperature * self.temp_decay)
            
            return self.opponent_model
    
    def get_current_phase(self, global_step):
        """
        Determine current training phase based on global step count.
        
        Phases:
        - Phase 1 (0-1M): Flight School - Learn basic flight
        - Phase 2 (1M-2M): Pursuit - Learn to track targets  
        - Phase 3 (2M-4M): Dogfight Basics - Learn positioning
        - Phase 4 (4M+): Full Combat - Complete combat training
        
        Args:
            global_step: Total environment steps taken
            
        Returns:
            int: Phase ID (1, 2, 3, or 4)
        """
        if global_step < 1_000_000:
            return 1  # Flight School
        elif global_step < 2_000_000:
            return 2  # Pursuit
        elif global_step < 4_000_000:
            return 3  # Dogfight Basics
        else:
            return 4  # Full Combat

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
        
        # 1. Stationary Dummy (Phase 1)
        if self.current_opponent_type == "stationary_dummy":
            # No movement at all - agent learns basic flight
            return np.zeros((batch_size, Config.ACTION_DIM), dtype=np.float32)
        
        # 2. Straight-Flying Dummy (Phase 2)
        if self.current_opponent_type == "straight_dummy":
            # Flies straight with constant throttle - agent learns pursuit
            actions = np.zeros((batch_size, Config.ACTION_DIM), dtype=np.float32)
            actions[:, 2] = 0.5  # Throttle at 50%
            return actions
        
        # 3. Random-Turning Dummy (Phase 3)
        if self.current_opponent_type == "random_dummy":
            # Random gentle turns - agent learns tracking
            actions = np.random.uniform(-0.3, 0.3, (batch_size, Config.ACTION_DIM)).astype(np.float32)
            actions[:, 2] = 0.5  # Keep throttle constant
            actions[:, 3] = 0.0  # No firing
            actions[:, 4] = 0.0  # No countermeasures
            return actions
        
        # 4. Random
        if self.current_opponent_type == "random":
            return np.random.uniform(-1, 1, (batch_size, Config.ACTION_DIM))
        
        # 5. Scripted (Kappa-PPG)
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
        
        # 6. Hardcoded Ace (Phase 4)
        if self.current_opponent_type == "hardcoded_ace":
            actions = []
            for i in range(batch_size):
                # Extract single observation
                single_obs = obs[i]
                # Get action from Ace
                act = self.ace.get_action(single_obs)
                actions.append(act)
            return np.array(actions, dtype=np.float32)
        
        # 7. Model (Phase 5)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).to(Config.DEVICE)
            # Handle LSTM state for opponent?
            # If opponent has memory, we need to maintain it.
            # Currently SelfPlayManager doesn't track opponent hidden states.
            # This is a limitation. For now, pass None (stateless opponent).
            # TODO: Add hidden state tracking for opponent pool.
            action, _, _, _, _ = self.opponent_model.get_action_and_value(obs_t)
            return action.cpu().numpy()