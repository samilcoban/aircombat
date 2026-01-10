# ================================================
# FILE: src/self_play.py
# ================================================
import os
import glob
import re
import json
import numpy as np
import torch
import shutil
from config import Config
from src.model import HybridActorCritic
from src.bot import HardcodedAce


class SelfPlayManager:
    """
    Manages the self-play curriculum and opponent pool.

    Features:
    - Prioritized Fictitious Self-Play (PFSP): Samples opponents based on difficulty.
    - Persistent Memory: Maintains GRU states for opponents across steps.
    - Gating: New models must defeat a gauntlet of past versions to enter the pool.
    - Self-Healing: Automatically cleans up metadata if checkpoint files are deleted.
    """

    def __init__(self, checkpoint_dir="checkpoints", phase=1):
        self.checkpoint_dir = checkpoint_dir

        # Opponent Model (The "Red" Team)
        self.opponent_model = HybridActorCritic().to(Config.DEVICE)
        self.opponent_model.eval()

        # Scripted Baseline
        self.ace = HardcodedAce()

        # State
        self.opponent_pool = []
        self.current_opponent_name = "Scripted"
        self.current_opponent_type = "scripted"  # 'model', 'ace', 'random', 'stable_drone'

        # Persistent Memory for Opponents (Batch, 1, D_MODEL)
        self.opponent_gru_states = None

        # Gating Params
        self.eval_episodes = 10
        self.win_rate_threshold = 0.50
        self.kappa = 1.0  # Difficulty scalar (1.0 = hard/normal, 0.0 = easy)

        self.load_pool_metadata()
        self.load_checkpoints_list()

    def save_pool_metadata(self):
        """Persists opponent pool stats to JSON."""
        metadata = {'pool': self.opponent_pool, 'kappa': self.kappa}
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        try:
            with open(os.path.join(self.checkpoint_dir, 'opponent_pool.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving pool metadata: {e}")

    def load_pool_metadata(self):
        """Loads opponent pool stats and cleans up missing files."""
        path = os.path.join(self.checkpoint_dir, 'opponent_pool.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    raw_pool = data.get('pool', [])
                    self.kappa = data.get('kappa', 1.0)

                    # === FIX: VALIDATE EXISTENCE ===
                    # Only keep opponents whose files actually exist on disk
                    self.opponent_pool = []
                    for opp in raw_pool:
                        if os.path.exists(opp['path']):
                            self.opponent_pool.append(opp)
                        else:
                            print(f"🧹 Cleaning up ghost entry from pool: {opp['path']}")
                    # ===============================

            except Exception as e:
                print(f"⚠️ Error loading pool metadata: {e}")

    def load_checkpoints_list(self):
        """Scans disk for new checkpoints to potentially add to the pool."""
        if not os.path.exists(self.checkpoint_dir): return

        files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt"))
        existing_paths = {op['path'] for op in self.opponent_pool}
        added_new = False

        for f in files:
            # Skip latest/best aliases, look for numbered checkpoints
            if "latest" not in f and "best" not in f:
                match = re.search(r'model_(\d+).pt', f)
                if match and f not in existing_paths:
                    # Found a new file not in metadata.
                    self.opponent_pool.append({
                        'path': f,
                        'win_rate': 0.5,  # Assume neutral until played
                        'step': int(match.group(1))
                    })
                    added_new = True

        if added_new:
            # Sort by step
            self.opponent_pool.sort(key=lambda x: x.get('step', 0))
            self.save_pool_metadata()

    def sample_opponent(self, global_step=0, phase=1):
        """
        Selects an opponent for the next training iteration based on Curriculum Phase.
        """
        # 1. Reset Internal State (New opponent = New brain)
        self.opponent_gru_states = None
        self.load_checkpoints_list()

        # PHASE 1: TARGET PRACTICE
        # Force a stable drone so the agent learns kinematics without being attacked.
        if phase == 1:
            self.current_opponent_type = "stable_drone"
            self.current_opponent_name = "Stable Drone (School)"
            return

        # PHASE 2: DOGFIGHT INSTRUCTOR
        # Mostly use the Hardcoded Ace (Expert) to teach angles.
        # Occasionally use Random to check robustness.
        if phase == 2:
            if np.random.rand() < 0.8:
                self.current_opponent_type = "ace"
                self.current_opponent_name = "Hardcoded Ace (Instructor)"
            else:
                self.current_opponent_type = "random"
                self.current_opponent_name = "Random (Warmup)"
            return

        # PHASE 3+: FULL PFSP
        rand = np.random.rand()
        latest_path = os.path.join(self.checkpoint_dir, "model_latest.pt")

        # Distribution Strategy:
        # 20% - True Self-Play (Latest Model)
        # 10% - Hardcoded Ace (Baseline / Reality Check)
        # 10% - Random / Drone (Sanity Check / Easy wins)
        # 60% - PFSP (Historical Pool)

        if rand < 0.20 and os.path.exists(latest_path):
            self.current_opponent_type = "model"
            self.current_opponent_name = "True Self-Play (Latest)"
            self._load_weights(latest_path)
            return

        if rand < 0.30:
            self.current_opponent_type = "ace"
            self.current_opponent_name = "Hardcoded Ace"
            return

        if rand < 0.40 or not self.opponent_pool:
            self.current_opponent_type = "stable_drone"
            self.current_opponent_name = "Stable Drone (Fallback)"
            return

        # PFSP: Prioritized Fictitious Self-Play
        # Probability of picking opponent i is proportional to (1 - win_rate_vs_i)^2
        # We want to play against opponents that beat us (low win rate for us).
        win_rates = np.array([op.get('win_rate', 0.5) for op in self.opponent_pool])

        # Difficulty: If we win 100% (1.0), prob -> 0. If we win 0%, prob -> 1.
        difficulty_score = (1.0 - win_rates) ** 2

        # Normalize
        if difficulty_score.sum() < 1e-6:
            probs = np.ones(len(difficulty_score)) / len(difficulty_score)
        else:
            probs = difficulty_score / difficulty_score.sum()

        chosen_idx = np.random.choice(len(self.opponent_pool), p=probs)
        chosen_opp = self.opponent_pool[chosen_idx]

        # === FIX: CHECK EXISTENCE BEFORE LOADING ===
        if not os.path.exists(chosen_opp['path']):
            print(f"⚠️ Opponent file missing: {chosen_opp['path']}. Cleaning and Resampling.")
            # Remove invalid entry from pool
            self.opponent_pool.pop(chosen_idx)
            self.save_pool_metadata()
            # Recursive retry to pick a valid one
            self.sample_opponent(global_step, phase)
            return
        # ===========================================

        self.current_opponent_type = "model"
        self.current_opponent_name = f"PFSP: {os.path.basename(chosen_opp['path'])}"
        self._load_weights(chosen_opp['path'])

    def _load_weights(self, path):
        try:
            ckpt = torch.load(path, map_location=Config.DEVICE)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

            # Clean DDP/Compile prefixes
            clean_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

            self.opponent_model.load_state_dict(clean_dict, strict=False)
        except Exception as e:
            print(f"⚠️ Error loading opponent {path}: {e}")
            self.current_opponent_type = "random"  # Fallback

    def get_action(self, obs, dones=None):
        """
        Generates actions for the Red Team.

        Args:
            obs: Observation Array (Num_Envs, Num_Red_Agents, Obs_Dim)
            dones: Boolean Array (Num_Envs,) indicating if the env just reset.

        Returns:
            actions: (Num_Envs, Num_Red_Agents, Action_Dim)
        """
        # Ensure obs is numpy
        if isinstance(obs, list): obs = np.array(obs)

        num_envs = obs.shape[0]
        num_agents_per_env = obs.shape[1]

        # Flatten for processing: (Total_Agents, Obs_Dim)
        flat_obs = obs.reshape(-1, obs.shape[-1])
        total_agents = flat_obs.shape[0]

        # --- A. Simple Opponents ---
        if self.current_opponent_type == "stable_drone":
            # Fly straight and level-ish
            actions = np.zeros((num_envs, num_agents_per_env, Config.ACTION_DIM), dtype=np.float32)
            actions[:, :, 2] = 0.8  # High throttle
            return actions

        if self.current_opponent_type == "random":
            return np.random.uniform(-1, 1, (num_envs, num_agents_per_env, Config.ACTION_DIM)).astype(np.float32)

        if self.current_opponent_type == "ace":
            actions = []
            for i in range(total_agents):
                actions.append(self.ace.get_action(flat_obs[i]))
            return np.array(actions).reshape(num_envs, num_agents_per_env, -1)

        # --- B. Neural Network Opponent ---
        with torch.no_grad():
            t_obs = torch.tensor(flat_obs, dtype=torch.float32).to(Config.DEVICE)

            # 1. Initialize Memory if needed
            if self.opponent_gru_states is None or self.opponent_gru_states.shape[1] != total_agents:
                self.opponent_gru_states = torch.zeros(1, total_agents, Config.D_MODEL).to(Config.DEVICE)

            # 2. Handle Resets (Dones)
            # If an environment reset, we must clear the GRU state for its agents.
            t_dones = None
            if dones is not None:
                # Dones usually come in as (Num_Envs,)
                # We need to broadcast this to (Num_Envs * Num_Agents,)
                if len(dones.shape) == 1 and dones.shape[0] == num_envs:
                    dones_expanded = np.repeat(dones, num_agents_per_env)
                    t_dones = torch.tensor(dones_expanded, dtype=torch.float32).to(Config.DEVICE)
                elif dones.shape[0] == total_agents:
                    t_dones = torch.tensor(dones, dtype=torch.float32).to(Config.DEVICE)

            # 3. Inference
            # Note: graph_data=None because the Actor doesn't use the GNN (only Critic does)
            action, _, _, _, next_gru = self.opponent_model.get_action_and_value(
                t_obs,
                graph_data=None,
                action=None,
                gru_state=self.opponent_gru_states,
                done=t_dones
            )

            # 4. Update Memory
            self.opponent_gru_states = next_gru.detach()

            return action.cpu().numpy().reshape(num_envs, num_agents_per_env, -1)

    def evaluate_candidate(self, candidate_model, env_maker_fn, phase_id):
        """
        The Gatekeeper.
        Runs a tournament. If candidate > 50% win rate, it joins the pool.
        """
        print(f"\n🛡️ Gatekeeper: Evaluating Candidate (Phase {phase_id})...")

        # 1. Select Test Opponents
        test_ops = []
        if phase_id <= 2:
            test_ops = [{'type': 'stable_drone'}]
        else:
            if not self.opponent_pool:
                print("  -> Pool empty. Candidate auto-accepted.")
                return True
            # Test against last 3 added (progress) + 1 Random historical (robustness)
            test_ops = self.opponent_pool[-3:]
            if len(self.opponent_pool) > 3:
                test_ops.append(np.random.choice(self.opponent_pool[:-3]))
            for op in test_ops: op['type'] = 'model'

        # 2. Run Games
        total_wins = 0
        total_games = 0

        # Create a dedicated evaluation env (single process to keep it simple/debuggable)
        env = env_maker_fn()
        env.unwrapped.set_phase(phase_id)

        candidate_model.eval()

        # Candidate Memory
        cand_gru = torch.zeros(1, Config.N_AGENTS, Config.D_MODEL).to(Config.DEVICE)

        for opp_cfg in test_ops:
            # Setup Opponent
            if opp_cfg['type'] == 'model':
                # Check existence before loading in test loop too
                if not os.path.exists(opp_cfg['path']):
                    continue
                self._load_weights(opp_cfg['path'])
                self.current_opponent_type = 'model'
            else:
                self.current_opponent_type = 'stable_drone'

            # Reset Opponent Memory
            self.opponent_gru_states = None

            for _ in range(self.eval_episodes):
                obs, info = env.reset()
                cand_gru.zero_()  # Reset candidate memory

                done = False
                while not done:
                    # Candidate Action
                    with torch.no_grad():
                        t_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                        # Unsqueeze batch dim -> (1, N_Agents, Obs)
                        # Flatten for model
                        flat_obs = t_obs.view(-1, Config.OBS_DIM)

                        act, _, _, _, next_cand_gru = candidate_model.get_action_and_value(
                            flat_obs, graph_data=None, gru_state=cand_gru
                        )
                        cand_gru = next_cand_gru
                        blue_act = act.cpu().numpy().flatten()  # (N_Agents * Act_Dim)

                    # Opponent Action
                    red_act = None
                    if "red_obs" in info:
                        # info['red_obs'] shape: (N_Agents, Obs_Dim) -> Add batch dim
                        red_obs_batch = np.expand_dims(info["red_obs"], axis=0)
                        # get_action handles internal GRU state
                        red_act = self.get_action(red_obs_batch, dones=None)[0]

                    # Step
                    obs, _, term, trunc, info = env.step(blue_act, red_actions=red_act)
                    done = term or trunc

                # Tally Result
                reason = info.get('termination_reason', 'none')
                if reason == 'win': total_wins += 1
                total_games += 1

        env.close()

        # Handle case where all opponents were missing/skipped
        if total_games == 0:
            return True

        win_rate = total_wins / total_games
        passed = win_rate >= self.win_rate_threshold

        print(f"  -> Result: {total_wins}/{total_games} Wins ({win_rate:.2%})")
        print(f"  -> Candidate {'ACCEPTED ✅' if passed else 'REJECTED ❌'}")

        return passed