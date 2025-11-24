import os
import glob
import re
import numpy as np
import torch
from config import Config
from src.model import AgentTransformer

class SelfPlayManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.opponent_model = AgentTransformer().to(Config.DEVICE)
        self.opponent_model.eval()
        self.available_checkpoints = []
        self.current_opponent_name = "Random"
        self.load_checkpoints_list()

    def load_checkpoints_list(self):
        """Scans the checkpoint directory for valid model files."""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt"))
        # Sort by update number
        self.available_checkpoints = sorted(
            files, 
            key=lambda f: int(re.search(r'model_(\d+).pt', f).group(1))
        )

    def sample_opponent(self):
        """
        Loads an opponent model.
        Strategy:
        - 20% Chance: Random Agent (No model loaded, returns None)
        - 60% Chance: Latest Checkpoint (Strongest available)
        - 20% Chance: Random Past Checkpoint (Robustness)
        """
        self.load_checkpoints_list()
        
        if not self.available_checkpoints:
            self.current_opponent_name = "Random (No Checkpoints)"
            return None

        rand = np.random.rand()
        
        if rand < 0.2:
            self.current_opponent_name = "Random"
            return None
        elif rand < 0.8:
            # Load Latest
            ckpt_path = self.available_checkpoints[-1]
            self.current_opponent_name = f"Latest ({os.path.basename(ckpt_path)})"
            self._load_weights(ckpt_path)
            return self.opponent_model
        else:
            # Load Random Past
            ckpt_path = np.random.choice(self.available_checkpoints)
            self.current_opponent_name = f"Past ({os.path.basename(ckpt_path)})"
            self._load_weights(ckpt_path)
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
        Get action from the currently loaded opponent model.
        obs: Numpy array of observation (Batch, Obs_Dim)
        """
        # Handle AsyncVectorEnv's object dtype arrays
        if obs.dtype == object:
            # Convert nested arrays to proper float array
            obs = np.stack([np.array(o, dtype=np.float32) for o in obs])
        
        batch_size = obs.shape[0]
        
        if self.current_opponent_name.startswith("Random"):
            # Return random action for the whole batch
            return np.random.uniform(-1, 1, (batch_size, Config.ACTION_DIM))
        
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).to(Config.DEVICE)
            action, _, _, _ = self.opponent_model.get_action_and_value(obs_t)
            return action.cpu().numpy()