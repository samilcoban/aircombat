# ================================================
# FILE: src/self_play.py
# ================================================
import os
import glob
import re
import json
import numpy as np
import torch
from config import Config
from src.model import HybridActorCritic  # <--- UPDATED IMPORT
from src.bot import HardcodedAce


class SelfPlayManager:
    """
    Manages the self-play curriculum and opponent pool.
    
    Self-Play Strategy:
    - Maintains a pool of past policy checkpoints ("opponents").
    - Prioritized Fictitious Self-Play (PFSP): Samples opponents based on their win rate (difficulty).
    - Gating Mechanism: New checkpoints must defeat a mix of past opponents to be added to the pool.
    """
    def __init__(self, checkpoint_dir="checkpoints", phase=2):
        self.checkpoint_dir = checkpoint_dir
        self.training_phase = phase

        # Update to new model class
        self.opponent_model = HybridActorCritic().to(Config.DEVICE)
        self.opponent_model.eval()

        self.ace = HardcodedAce()

        self.opponent_pool = []
        self.current_opponent_name = "Scripted"
        self.current_opponent_type = "scripted"

        self.eval_episodes = 10
        self.win_rate_threshold = 0.5
        self.kappa = 1.0
        self.last_eval_passed = False

        self.load_pool_metadata()
        self.load_checkpoints_list()

    def save_pool_metadata(self):
        """Persists the opponent pool metadata (win rates, paths) to disk."""
        metadata = {'pool': self.opponent_pool, 'kappa': self.kappa}
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        try:
            with open(os.path.join(self.checkpoint_dir, 'opponent_pool.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving pool metadata: {e}")

    def load_pool_metadata(self):
        path = os.path.join(self.checkpoint_dir, 'opponent_pool.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.opponent_pool = data.get('pool', [])
                    self.kappa = data.get('kappa', 1.0)
            except Exception as e:
                print(f"Error loading pool metadata: {e}")

    def load_checkpoints_list(self):
        """Scans the checkpoint directory for new models to add to the pool."""
        if not os.path.exists(self.checkpoint_dir): return
        files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt"))
        existing_paths = {op['path'] for op in self.opponent_pool}
        added_new = False
        for f in files:
            if "latest" not in f and re.search(r'model_(\d+).pt', f) and f not in existing_paths:
                self.opponent_pool.append({'path': f, 'win_rate': 0.5, 'score': 1.0})
                added_new = True
        if added_new:
            self.save_pool_metadata()

    def evaluate_candidate(self, candidate_model, env_maker_fn, phase_id):
        """
        Gating Function: Determines if the current candidate model is good enough to be added to the pool.
        
        The candidate must play against a set of test opponents.
        - Phase 1-2: Test against stable drone.
        - Phase 3+: Test against recent pool models (to prevent regression) and random older models.
        """
        print(f"\n--- AOS Gate Function: Evaluating Candidate (Phase {phase_id}) ---")
        test_opponents = []

        if phase_id in [1, 2]:
            test_opponents = [{'type': 'stable_drone'}]
        else:
            if not self.opponent_pool:
                print("  Pool empty. Candidate accepted by default.")
                candidate_model.train()
                return True

            # Test against the 5 most recent opponents (ensure progress)
            window = self.opponent_pool[-5:]
            test_opponents = window.copy()
            # Add one random older opponent (prevent cyclic forgetting)
            if len(self.opponent_pool) > 5:
                test_opponents.append(np.random.choice(self.opponent_pool[:-5]))

            for op in test_opponents: op['type'] = 'model'

        total_wins = 0
        total_games = 0
        outcomes = {"win": 0, "loss": 0, "draw": 0}

        env = env_maker_fn()
        env.unwrapped.set_phase(phase_id)
        candidate_model.eval()

        # Initialize GRU states
        # Shape: (1, 1, Hidden) - 1 agent
        blue_gru = torch.zeros(1, 1, Config.D_MODEL).to(Config.DEVICE)

        try:
            for opp_info in test_opponents:
                if opp_info['type'] == 'model':
                    self._load_weights(opp_info['path'])
                    self.current_opponent_type = "model"
                else:
                    self.current_opponent_type = "stable_drone"

                for _ in range(self.eval_episodes):
                    obs, info = env.reset()

                    # Reset GRU for new episode
                    blue_gru = torch.zeros(1, 1, Config.D_MODEL).to(Config.DEVICE)

                    done = False
                    while not done:
                        # 1. Blue Action
                        with torch.no_grad():
                            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                            # Passing graph_data=None because Actor doesn't use it
                            action, _, _, _, blue_gru = candidate_model.get_action_and_value(
                                obs_t, graph_data=None, gru_state=blue_gru
                            )
                            blue_action = action.cpu().numpy().flatten()

                        # 2. Red Action
                        red_action = None
                        if self.current_opponent_type == "model" and "red_obs" in info:
                            red_obs_batch = np.expand_dims(info["red_obs"], axis=0)
                            # get_action handles its own internal GRU state reset for simplicity here
                            red_action = self.get_action(red_obs_batch)[0]

                        # 3. Step
                        if red_action is not None:
                            obs, _, term, trunc, info = env.step(blue_action, red_actions=red_action)
                        else:
                            obs, _, term, trunc, info = env.step(blue_action)
                        done = term or trunc

                    reason = info.get("termination_reason", "none")
                    if reason == "win":
                        total_wins += 1;
                        outcomes["win"] += 1
                    elif reason in ["crash", "shot", "floor_violation"]:
                        outcomes["loss"] += 1
                    else:
                        outcomes["draw"] += 1
                    total_games += 1
        finally:
            env.close()
            candidate_model.train()

        win_rate = total_wins / total_games if total_games > 0 else 0
        print(f"  Result: Win Rate {win_rate:.2f} ({outcomes})")
        self.last_eval_passed = (win_rate >= self.win_rate_threshold)
        return self.last_eval_passed

    def sample_opponent(self, global_step=0):
        """
        Selects an opponent for the next training episode.
        
        Selection Logic:
        - 20%: Play against the latest self-copy (True Self-Play).
        - 10%: Play against Hardcoded Ace (Exploiter to prevent overfitting to weak opponents).
        - 10%: Play against Random/Drone (Sanity check).
        - 60%: Play against History Pool (PFSP).
        """
        self.load_checkpoints_list()
        self.save_pool_metadata()
        rand = np.random.rand()
        latest_path = os.path.join(self.checkpoint_dir, "model_latest.pt")

        if rand < 0.20 and os.path.exists(latest_path):
            self.current_opponent_name = "True Self-Play (Latest)"
            self.current_opponent_type = "model"
            self._load_weights(latest_path)
            return
        if rand < 0.30:
            self.current_opponent_name = "Hardcoded Ace (Exploiter)"
            self.current_opponent_type = "ace"
            return
        if rand < 0.40:
            self.current_opponent_name = "Random/Drone (Weak)"
            self.current_opponent_type = "random"
            return
        if not self.opponent_pool:
            self.current_opponent_name = "Random (Pool Empty)"
            self.current_opponent_type = "random"
            return

        # Prioritized Sampling based on difficulty (Win Rate close to 0.5 is hardest?)
        # Actually, here difficulty is defined as (1 - win_rate)^2.
        # This means we prioritize opponents that beat us (low win rate for us).
        win_rates = np.array([op.get('win_rate', 0.5) for op in self.opponent_pool])
        difficulties = (1.0 - win_rates) ** 2
        total_difficulty = difficulties.sum()
        if total_difficulty < 1e-9:
            probs = np.ones(len(difficulties)) / len(difficulties)
        else:
            probs = difficulties / total_difficulty
        probs = probs / probs.sum()

        chosen_opp = np.random.choice(self.opponent_pool, p=probs)
        self.current_opponent_name = f"PFSP: {os.path.basename(chosen_opp['path'])}"
        self.current_opponent_type = "model"
        self._load_weights(chosen_opp['path'])

    def _load_weights(self, path):
        try:
            ckpt = torch.load(path, map_location=Config.DEVICE)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            # Strip compile prefixes
            clean_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.opponent_model.load_state_dict(clean_dict)
        except Exception as e:
            print(f"Error loading opponent {path}: {e}")

    def get_action(self, obs):
        """
        Get action for opponent.
        Note: For simplicity in self-play manager, we don't maintain persistent GRU state
        across steps for the *opponent* (it resets every call).
        To improve this, one would need to pass opponent GRU states in and out.
        """
        if isinstance(obs, np.ndarray) and obs.dtype == np.object_:
            obs = np.stack(obs).astype(np.float32)

        batch_size = obs.shape[0]
        n_enemies = obs.shape[1]
        flat_obs = obs.reshape(-1, obs.shape[-1])
        total_agents = flat_obs.shape[0]

        if self.current_opponent_type == "stable_drone":
            actions = np.zeros((batch_size, n_enemies, Config.ACTION_DIM), dtype=np.float32)
            actions[:, :, 2] = 0.8  # Throttle High
            return actions
        if self.current_opponent_type == "random":
            return np.random.uniform(-1, 1, (batch_size, n_enemies, Config.ACTION_DIM)).astype(np.float32)
        if self.current_opponent_type == "ace":
            actions = []
            for i in range(total_agents):
                actions.append(self.ace.get_action(flat_obs[i]))
            return np.array(actions).reshape(batch_size, n_enemies, Config.ACTION_DIM)

        with torch.no_grad():
            t_obs = torch.tensor(flat_obs, dtype=torch.float32).to(Config.DEVICE)
            # Init fresh GRU state for opponent inference
            gru_state = torch.zeros(1, total_agents, Config.D_MODEL).to(Config.DEVICE)

            act, _, _, _, _ = self.opponent_model.get_action_and_value(
                t_obs, graph_data=None, gru_state=gru_state
            )
            return act.cpu().numpy().reshape(batch_size, n_enemies, Config.ACTION_DIM)