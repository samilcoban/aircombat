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
    def __init__(self, output_dir="logs"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = int(time.time())
        self.filename = os.path.join(output_dir, f"replay_log_{timestamp}.csv")
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)

        self.headers = [
            "step", "time", "id", "team",
            "x", "y", "alt", "hdg", "pitch", "roll", "spd", "g",
            "act_roll", "act_g", "act_thr", "act_fire", "act_cm",
            "reward"
        ]
        self.writer.writerow(self.headers)
        print(f"📄 Recording flight data to: {self.filename}")

    def log(self, step, sim_time, core, actions_dict, rewards_dict):
        for uid, ent in core.entities.items():
            if ent.type != "plane": continue

            act = actions_dict.get(uid, np.zeros(5))
            rew = rewards_dict.get(uid, 0.0)

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
        if self.file:
            self.file.close()


def get_latest_checkpoint():
    if not os.path.exists("checkpoints"):
        return None
    files = glob.glob("checkpoints/model_*.pt")
    if not files:
        if os.path.exists("checkpoints/model_latest.pt"):
            return "checkpoints/model_latest.pt"
        return None

    numbered = []
    for f in files:
        match = re.search(r'model_(\d+).pt', f)
        if match:
            numbered.append((int(match.group(1)), f))

    if numbered:
        return max(numbered, key=lambda x: x[0])[1]

    if os.path.exists("checkpoints/model_latest.pt"):
        return "checkpoints/model_latest.pt"
    return None


def play(checkpoint_path=None, phase=1, opponent_type="drone"):
    # Default Checkpoint
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No checkpoints found in 'checkpoints/' directory.")
            return

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load Model
    model = HybridActorCritic().to(Config.DEVICE)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint,
                                                                  dict) and "model_state_dict" in checkpoint else checkpoint
        # Strip compile prefixes
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Init Self-Play & Override Opponent
    sp_manager = SelfPlayManager()

    # FORCE OPPONENT TYPE
    sp_manager.current_opponent_type = opponent_type
    if opponent_type == "drone":
        sp_manager.current_opponent_name = "Stable Drone (Target Practice)"
    elif opponent_type == "ace":
        sp_manager.current_opponent_name = "Hardcoded Ace (Expert)"
    else:
        sp_manager.current_opponent_name = "Random (Suicidal)"

    print(f"Opponent: {sp_manager.current_opponent_name}")

    # Init Env
    env = AirCombatEnv()
    print(f"Initializing Environment in Phase {phase}...")
    env.set_phase(phase)

    obs, info = env.reset()
    recorder = BattleRecorder()
    renderer = Panda3DRenderer()

    # Init GRU
    n_agents = obs.shape[0]
    gru_state = torch.zeros(1, n_agents, Config.D_MODEL).to(Config.DEVICE)

    done = False
    step = 0

    print("Running simulation...")
    try:
        with torch.no_grad():
            while not done:
                # 1. Blue Action
                obs_t = torch.tensor(obs, dtype=torch.float32).to(Config.DEVICE)
                if obs_t.dim() == 1: obs_t = obs_t.unsqueeze(0)

                action_t, _, _, _, gru_state = model.get_action_and_value(
                    obs_t, graph_data=None, gru_state=gru_state
                )
                blue_action = action_t.cpu().numpy()

                # 2. Red Action (Self-Play Manager Handles Types)
                red_action = None
                if "red_obs" in info:
                    # Create batch for sp_manager
                    red_obs = info["red_obs"]
                    red_obs_batch = np.expand_dims(red_obs, axis=0)

                    # Get Action based on forced type
                    red_action_batch = sp_manager.get_action(red_obs_batch)
                    red_action = red_action_batch[0]

                # 3. Step
                obs, rewards, term, trunc, info = env.step(blue_action, red_actions=red_action)
                done = term or trunc
                step += 1

                # 4. Log Data
                actions_map = {}
                rewards_map = {}

                # Blue map
                for i, uid in enumerate(env.blue_ids):
                    if i < len(blue_action): actions_map[uid] = blue_action[i]
                    if i < len(rewards): rewards_map[uid] = rewards[i]

                # Red map
                if red_action is not None:
                    for i, uid in enumerate(env.red_ids):
                        if i < len(red_action): actions_map[uid] = red_action[i]

                recorder.log(step, env.core.time, env.core, actions_map, rewards_map)

                # 5. Render
                renderer.update_entities(env.core.entities, Config.MAP_LIMITS)
                renderer.taskMgr.step()

                if not renderer.check_running():
                    print("Window closed by user")
                    break

                # Slow down visualization slightly
                # time.sleep(0.01)

                if done:
                    print(f"Episode finished in {step} steps. Winner: {info.get('termination_reason', 'unknown')}")
                    obs, info = env.reset()

                    n_agents = obs.shape[0]
                    gru_state = torch.zeros(1, n_agents, Config.D_MODEL).to(Config.DEVICE)
                    done = False
                    step = 0

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        recorder.close()
        renderer.cleanup()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--phase", type=int, default=1, help="Simulation Phase (1=School, 3=Combat)")
    parser.add_argument("--opponent", type=str, default="drone", choices=["drone", "random", "ace"],
                        help="Opponent Type")
    args = parser.parse_args()

    play(args.checkpoint, phase=args.phase, opponent_type=args.opponent)