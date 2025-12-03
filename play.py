# ================================================
# FILE: play.py
# ================================================
import argparse
import torch
import numpy as np
import time
import os
import glob
import re
import csv
from src.env import AirCombatEnv
from src.model import HybridActorCritic
from src.self_play import SelfPlayManager
from config import Config
from src.render_panda3d import Panda3DRenderer


class BattleRecorder:
    """
    Records replay data to a CSV file.
    Compact format for post-analysis.
    """

    def __init__(self, output_dir="logs"):
        # Intuition: Create logs directory if it doesn't exist.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Intuition: Generate a unique filename based on the current timestamp.
        timestamp = int(time.time())
        self.filename = os.path.join(output_dir, f"replay_log_{timestamp}.csv")
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)

        # Headers: Time, AgentID, Team, Pos(x,y,z), Att(h,p,r), Speed, G, Actions(5), Reward
        # Intuition: Define the columns for the CSV log file.
        self.headers = [
            "step", "time", "id", "team",
            "x", "y", "alt", "hdg", "pitch", "roll", "spd", "g",
            "act_roll", "act_g", "act_thr", "act_fire", "act_cm",
            "reward"
        ]
        self.writer.writerow(self.headers)
        print(f"📄 Recording flight data to: {self.filename}")

    def log(self, step, sim_time, core, actions_dict, rewards_dict):
        # Log All Agents (Blue and Red)
        # Intuition: Iterate through all entities in the simulation core.
        for uid, ent in core.entities.items():
            if ent.type != "plane": continue

            # Intuition: Retrieve actions and rewards for the current entity, defaulting to zeros if not found.
            act = actions_dict.get(uid, np.zeros(5))
            rew = rewards_dict.get(uid, 0.0)

            # Intuition: Format the data row for the CSV file.
            row = [
                step, f"{sim_time:.2f}", uid, ent.team,
                f"{ent.x:.1f}", f"{ent.y:.1f}", f"{ent.alt:.1f}",
                f"{ent.heading:.1f}", f"{ent.pitch:.2f}", f"{ent.roll:.2f}",
                f"{ent.speed:.1f}", f"{ent.g_load:.2f}",
                f"{act[0]:.2f}", f"{act[1]:.2f}", f"{act[2]:.2f}", f"{act[3]:.1f}", f"{act[4]:.1f}",
                f"{rew:.3f}"
            ]
            self.writer.writerow(row)

    def close(self):
        # Intuition: Close the file handle when logging is finished.
        if self.file:
            self.file.close()


def get_latest_checkpoint():
    # Intuition: Check if checkpoints directory exists.
    if not os.path.exists("checkpoints"):
        return None
    # Intuition: Find all model checkpoint files.
    files = glob.glob("checkpoints/model_*.pt")
    if not files:
        # Intuition: Fallback to 'model_latest.pt' if no numbered checkpoints are found.
        if os.path.exists("checkpoints/model_latest.pt"):
            return "checkpoints/model_latest.pt"
        return None

    # Try to find highest numbered checkpoint
    # Intuition: Parse filenames to extract checkpoint numbers and find the maximum.
    numbered = []
    for f in files:
        match = re.search(r'model_(\d+).pt', f)
        if match:
            numbered.append((int(match.group(1)), f))

    if numbered:
        # Intuition: Return the file path with the highest checkpoint number.
        return max(numbered, key=lambda x: x[0])[1]

    # Fallback to latest.pt
    # Intuition: If parsing fails, check for 'model_latest.pt' again.
    if os.path.exists("checkpoints/model_latest.pt"):
        return "checkpoints/model_latest.pt"

    return None


def play(checkpoint_path=None, output_path="replay.mp4"):
    # Default Checkpoint Handling
    # Intuition: Automatically find the latest checkpoint if none is provided.
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No checkpoints found in 'checkpoints/' directory.")
            return

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load Model
    # Intuition: Initialize the model architecture.
    model = HybridActorCritic().to(Config.DEVICE)
    try:
        # Intuition: Load the saved state dictionary into the model.
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint,
                                                                  dict) and "model_state_dict" in checkpoint else checkpoint

        # Strip compile prefixes if present
        # Intuition: Remove '_orig_mod.' prefix added by torch.compile() if present.
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Intuition: Set the model to evaluation mode (disables dropout, etc.).
    model.eval()

    # Init Self-Play (for opponent)
    # Intuition: Initialize the self-play manager to control the opponent agents.
    sp_manager = SelfPlayManager()
    sp_manager.sample_opponent()
    print(f"Opponent: {sp_manager.current_opponent_name}")

    # Init Env & Recorder
    # Intuition: Initialize the air combat environment.
    env = AirCombatEnv()
    # Force a specific scenario if desired, e.g. Phase 3 for combat
    # Intuition: Set the curriculum phase to 3 (full combat).
    env.set_phase(3)

    # Intuition: Reset the environment to get the initial observation.
    obs, info = env.reset()
    recorder = BattleRecorder()

    # Setup Renderer
    # Intuition: Initialize the Panda3D renderer for visualization.
    renderer = Panda3DRenderer()

    # Init GRU State
    # Correctly size for the number of agents in the observation batch
    # Intuition: Initialize the hidden state for the GRU (Recurrent Neural Network).
    # Math: Shape is (1, N_Agents, Hidden_Size).
    n_agents = obs.shape[0]
    gru_state = torch.zeros(1, n_agents, Config.D_MODEL).to(Config.DEVICE)

    done = False
    step = 0

    print("Running simulation...")
    try:
        # Intuition: Disable gradient calculation for inference to save memory and computation.
        with torch.no_grad():
            while not done:
                # 1. Blue Action
                # Intuition: Convert observation to tensor and move to device.
                obs_t = torch.tensor(obs, dtype=torch.float32).to(Config.DEVICE)

                # Pass gru_state and receive the updated state
                # Intuition: Forward pass through the model to get actions and new hidden state.
                action_t, _, _, _, gru_state = model.get_action_and_value(
                    obs_t, graph_data=None, gru_state=gru_state
                )
                blue_action = action_t.cpu().numpy()

                # 2. Red Action (Using Self-Play Manager)
                red_action = None
                if "red_obs" in info:
                    red_obs = info["red_obs"]
                    # Add Batch Dimension (1, N_Agents, Dim)
                    # Math: Expand dims to match model input expectation.
                    red_obs_batch = np.expand_dims(red_obs, axis=0)
                    # Get Action (Returns (1, N_Agents, 5))
                    # Intuition: Get opponent actions from the self-play policy.
                    red_action_batch = sp_manager.get_action(red_obs_batch)
                    # Remove Batch Dimension -> (N_Agents, 5)
                    red_action = red_action_batch[0]

                # 3. Step
                # Intuition: Execute actions in the environment and get the next state.
                if red_action is not None:
                    obs, rewards, term, trunc, info = env.step(blue_action, red_actions=red_action)
                else:
                    obs, rewards, term, trunc, info = env.step(blue_action)

                done = term or trunc
                step += 1

                # 4. Log Data
                actions_map = {}
                rewards_map = {}

                # Map Blue Actions
                # Intuition: Store blue agent actions and rewards for logging.
                for i, uid in enumerate(env.blue_ids):
                    if i < len(blue_action): actions_map[uid] = blue_action[i]
                    if i < len(rewards): rewards_map[uid] = rewards[i]

                # Map Red Actions
                # Intuition: Store red agent actions for logging.
                if red_action is not None:
                    for i, uid in enumerate(env.red_ids):
                        if i < len(red_action): actions_map[uid] = red_action[i]

                # Intuition: Write the current step data to the CSV log.
                recorder.log(step, env.core.time, env.core, actions_map, rewards_map)

                # 5. Render
                # Intuition: Update the 3D visualization with the new entity states.
                renderer.update_entities(env.core.entities, Config.MAP_LIMITS)
                renderer.taskMgr.step()

                # Intuition: Check if the renderer window was closed by the user.
                if not renderer.check_running():
                    print("Window closed by user")
                    break

                if done:
                    print(f"Episode finished in {step} steps. Winner: {info.get('termination_reason', 'unknown')}")
                    # Loop reset
                    # Intuition: Reset the environment for the next episode.
                    obs, info = env.reset()

                    # Reset GRU state for new episode
                    # Intuition: Reset hidden state to zeros for the new episode.
                    n_agents = obs.shape[0]
                    gru_state = torch.zeros(1, n_agents, Config.D_MODEL).to(Config.DEVICE)

                    done = False
                    step = 0

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Intuition: Clean up resources.
        recorder.close()
        renderer.cleanup()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="replay.mp4", help="Output video path")
    args = parser.parse_args()

    play(args.checkpoint, args.output)