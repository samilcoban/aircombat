import os
import glob
import re
import json
import numpy as np
import torch
from config import Config
from src.model import AgentTransformer
from src.bot import HardcodedAce


class SelfPlayManager:
    def __init__(self, checkpoint_dir="checkpoints", phase=2):
        self.checkpoint_dir = checkpoint_dir
        self.training_phase = phase
        self.opponent_model = AgentTransformer().to(Config.DEVICE)
        self.opponent_model.eval()
        self.ace = HardcodedAce()

        # Pool State
        self.opponent_pool = []
        self.current_opponent_name = "Scripted"
        self.current_opponent_type = "scripted"

        # Evaluation Params
        self.eval_episodes = 10
        self.win_rate_threshold = 0.5

        self.load_pool_metadata()
        self.load_checkpoints_list()
        self.kappa = 1.0
        self.last_eval_passed = False

    def save_pool_metadata(self):
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
        if not os.path.exists(self.checkpoint_dir): return
        files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt"))
        existing_paths = {op['path'] for op in self.opponent_pool}
        for f in files:
            if re.search(r'model_(\d+).pt', f) and f not in existing_paths:
                self.opponent_pool.append({'path': f, 'win_rate': 0.5, 'score': 1.0})

    def evaluate_candidate(self, candidate_model, env_maker_fn, phase_id):
        print(f"\n--- AOS Gate Function: Evaluating Candidate (Phase {phase_id}) ---")
        test_opponents = []

        if phase_id in [1, 2]:
            test_opponents = [{'type': 'stable_drone'}]
        else:
            if not self.opponent_pool:
                print("  Pool empty. Candidate accepted by default.")
                # FIX: Restore train mode
                candidate_model.train()
                return True
            window = self.opponent_pool[-5:]
            test_opponents = window.copy()
            if len(self.opponent_pool) > 5:
                test_opponents.append(np.random.choice(self.opponent_pool[:-5]))
            for op in test_opponents: op['type'] = 'model'

        total_wins = 0
        total_games = 0
        outcomes = {"win": 0, "loss": 0, "draw": 0}

        env = env_maker_fn()
        env.unwrapped.set_phase(phase_id)

        # Set Eval Mode
        candidate_model.eval()

        try:
            for opp_info in test_opponents:
                if opp_info['type'] == 'model':
                    self._load_weights(opp_info['path'])
                    self.current_opponent_type = "model"
                else:
                    self.current_opponent_type = "stable_drone"

                for _ in range(self.eval_episodes):
                    obs, info = env.reset()
                    lstm_state = None
                    done = False
                    while not done:
                        with torch.no_grad():
                            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                            action, _, _, _, lstm_state = candidate_model.get_action_and_value(obs_t,
                                                                                               lstm_state=lstm_state)
                            blue_action = action.cpu().numpy().flatten()

                        red_action = None
                        if self.current_opponent_type == "model" and "red_obs" in info:
                            red_obs_batch = np.expand_dims(info["red_obs"], axis=0)
                            red_action = self.get_action(red_obs_batch)[0]

                        if red_action is not None:
                            obs, _, term, trunc, info = env.step(np.concatenate([blue_action, red_action]))
                        else:
                            obs, _, term, trunc, info = env.step(blue_action)
                        done = term or trunc

                    reason = info.get("termination_reason", "none")
                    if reason == "win":
                        total_wins += 1; outcomes["win"] += 1
                    elif reason in ["crash", "shot", "floor_violation"]:
                        outcomes["loss"] += 1
                    else:
                        outcomes["draw"] += 1
                    total_games += 1
        finally:
            env.close()
            # FIX: RESTORE TRAINING MODE CRITICAL!
            candidate_model.train()

        win_rate = total_wins / total_games if total_games > 0 else 0
        print(f"  Result: Win Rate {win_rate:.2f} ({outcomes})")
        self.last_eval_passed = (win_rate >= self.win_rate_threshold)
        return self.last_eval_passed

    def sample_opponent(self, global_step=0):
        self.load_checkpoints_list()

        if not self.opponent_pool:
            self.current_opponent_name = "Random (Pool Empty)"
            self.current_opponent_type = "random"
            return

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
            clean_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.opponent_model.load_state_dict(clean_dict)
        except Exception as e:
            print(f"Error loading opponent {path}: {e}")

    def get_action(self, obs):
        if isinstance(obs, np.ndarray) and obs.dtype == np.object_:
            obs = np.stack(obs).astype(np.float32)

        batch_size = obs.shape[0]
        if self.current_opponent_type == "stable_drone":
            return np.zeros((batch_size, Config.ACTION_DIM), dtype=np.float32)
        if self.current_opponent_type == "random":
            return np.random.uniform(-1, 1, (batch_size, Config.ACTION_DIM))

        with torch.no_grad():
            t_obs = torch.tensor(obs, dtype=torch.float32).to(Config.DEVICE)
            act, _, _, _, _ = self.opponent_model.get_action_and_value(t_obs)
            return act.cpu().numpy()