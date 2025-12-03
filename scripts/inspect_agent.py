#!/usr/bin/env python3
"""
Inspect agent behavior by logging observations, actions, and rewards to a file.
Updated for Relative/Egocentric Observation Space (29 features).
"""

import argparse
import torch
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env import AirCombatEnv
from src.model import HybridActorCritic
from src.self_play import SelfPlayManager
from config import Config


def inspect_agent(checkpoint_path, output_file="agent_inspection.txt", max_steps=500):
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load Model
    model = HybridActorCritic().to(Config.DEVICE)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint,
                                                                  dict) and "model_state_dict" in checkpoint else checkpoint
        clean_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_dict)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.eval()

    sp_manager = SelfPlayManager()
    sp_manager.sample_opponent()

    env = AirCombatEnv()
    obs, info = env.reset()

    # Init GRU
    n_agents = obs.shape[0]
    gru_state = torch.zeros(1, n_agents, Config.D_MODEL).to(Config.DEVICE)

    with open(output_file, 'w') as f:
        f.write(f"INSPECTION LOG - Relative Observation Space\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")

        done = False
        step = 0
        total_reward = 0.0

        try:
            with torch.no_grad():
                while not done and step < max_steps:
                    obs_t = torch.tensor(obs, dtype=torch.float32).to(Config.DEVICE)

                    action, _, _, _, gru_state = model.get_action_and_value(
                        obs_t, graph_data=None, gru_state=gru_state
                    )
                    blue_action = action.cpu().numpy()

                    # Log State (Ego + Threat)
                    f.write(f"STEP {step}\n{'-' * 60}\n")

                    # 1. Parse Ego (Slot 0)
                    # Indices: 7=Speed, 8=Alt, 16=Ammo
                    ego = obs[0, :Config.FEAT_DIM]
                    f.write(f"EGO STATE:\n")
                    f.write(f"  Speed: {ego[7] * 1000:.0f} kts, Alt: {ego[8] * 15000:.0f} m\n")
                    f.write(f"  Fuel: {ego[15]:.2f}, Ammo: {ego[16] * 4:.0f}\n")

                    # 2. Parse Threats (Slots 1+)
                    # Find closest valid entity
                    closest_threat = None
                    min_range = 1.0

                    for i in range(1, Config.MAX_ENTITIES):
                        ent = obs[0, i * Config.FEAT_DIM: (i + 1) * Config.FEAT_DIM]
                        if ent[17] == 0: continue  # Empty/Dead

                        rng = ent[0]
                        if rng < min_range:
                            min_range = rng
                            closest_threat = ent

                    if closest_threat is not None:
                        # Decode Relative Metrics
                        rng_km = closest_threat[0] * 60.0
                        az_deg = np.degrees(np.arctan2(closest_threat[2], closest_threat[1]))
                        el_sin = closest_threat[3]
                        closure = closest_threat[6] * 2000.0
                        is_missile = closest_threat[18] > 0.5

                        type_str = "MISSILE" if is_missile else "PLANE"

                        f.write(f"\nCLOSEST THREAT ({type_str}):\n")
                        f.write(f"  Range: {rng_km:.1f} km\n")
                        f.write(f"  Azimuth: {az_deg:.1f} deg (Neg=Left, Pos=Right)\n")
                        f.write(f"  Elevation: {el_sin:.2f} (Sin)\n")
                        f.write(f"  Closure: {closure:.0f} kts\n")
                        f.write(f"  RWR: {closest_threat[20]:.0f}, MAWS: {closest_threat[21]:.0f}\n")
                    else:
                        f.write("\nNO THREATS VISIBLE\n")

                    # Log Action
                    act = blue_action[0]
                    f.write(f"\nACTION:\n")
                    f.write(f"  Roll: {act[0]:.2f}, G: {act[1]:.2f}, Thr: {act[2]:.2f}, Fire: {act[3]:.2f}\n")

                    # Step
                    red_action = None
                    if "red_obs" in info:
                        red_action = sp_manager.get_action(np.expand_dims(info["red_obs"], 0))[0]

                    if red_action is not None:
                        obs, rewards, term, trunc, info = env.step(blue_action, red_actions=red_action)
                    else:
                        obs, rewards, term, trunc, info = env.step(blue_action)

                    reward = rewards[0]
                    total_reward += reward
                    done = term[0] or trunc[0]

                    f.write(f"  Reward: {reward:.4f} (Total: {total_reward:.4f})\n\n")
                    step += 1

        except KeyboardInterrupt:
            print("Interrupted.")

        f.write(f"EPISODE END: {info.get('termination_reason', 'unknown')}\n")
        env.close()
        print(f"Log saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="agent_inspection.txt")
    args = parser.parse_args()
    inspect_agent(args.checkpoint, args.output)