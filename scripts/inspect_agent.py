#!/usr/bin/env python3
"""
INSPECT AGENT SCRIPT (PHASE 5 VERIFICATION)
-------------------------------------------
Runs a full episode and logs DECODED observations to verify
that the math changes resulted in intelligible data.
"""

import argparse
import torch
import numpy as np
import time
import sys
import os
import math

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

    # Init Self-Play (Use Drone for consistent testing)
    sp_manager = SelfPlayManager()
    sp_manager.current_opponent_type = "stable_drone"
    print("Opponent: Stable Drone")

    env = AirCombatEnv()
    obs, info = env.reset()

    # Init GRU
    n_agents = obs.shape[0]
    gru_state = torch.zeros(1, n_agents, Config.D_MODEL).to(Config.DEVICE)

    print(f"Logging to {output_file}...")

    with open(output_file, 'w') as f:
        f.write(f"INSPECTION LOG - Phase 5 Refactor\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Note: Angles decoded from Cos/Sin features.\n\n")

        done = False
        step = 0
        total_reward = 0.0

        try:
            with torch.no_grad():
                while not done and step < max_steps:
                    obs_t = torch.tensor(obs, dtype=torch.float32).to(Config.DEVICE)

                    # Get Action
                    action, _, _, _, gru_state = model.get_action_and_value(
                        obs_t, graph_data=None, gru_state=gru_state
                    )
                    blue_action = action.cpu().numpy()

                    # Log Step
                    f.write(f"STEP {step} | Reward: {total_reward:.4f}\n{'-' * 60}\n")

                    # --- DECODE OBSERVATION (Blue 0) ---
                    agent_obs = obs[0]

                    # 1. Ego State
                    # Indices: 1=Alt, 2=Speed, 4=CosH, 5=SinH, 6=Pitch, 7=Roll
                    alt_m = agent_obs[1] * 15000.0
                    spd_kts = agent_obs[2] * 1000.0
                    # Decode Heading
                    hdg_rad = math.atan2(agent_obs[5], agent_obs[4])
                    hdg_deg = math.degrees(hdg_rad) % 360
                    # Decode Attitude
                    pitch_deg = math.degrees(agent_obs[6] * 1.57)
                    roll_deg = math.degrees(agent_obs[7] * 3.14)

                    f.write(
                        f"EGO: Alt {alt_m:.0f}m | Spd {spd_kts:.0f}kts | Hdg {hdg_deg:.1f}° | Pitch {pitch_deg:.1f}° | Roll {roll_deg:.1f}°\n")

                    # 2. Threat Analysis (Tracks)
                    # Iterate tracks (14 features each)
                    tracks_flat = agent_obs[Config.FEAT_DIM_EGO:]
                    num_tracks = len(tracks_flat) // Config.FEAT_DIM_EDGE

                    for i in range(num_tracks):
                        t_vec = tracks_flat[i * Config.FEAT_DIM_EDGE: (i + 1) * Config.FEAT_DIM_EDGE]

                        # Range (Index 0)
                        rng_norm = t_vec[0]
                        if rng_norm < 1e-5: continue  # Padding

                        rng_km = rng_norm * 60.0

                        # Decode Angles (Indices 1=CosAz, 2=SinAz, 3=SinEl)
                        az_rad = math.atan2(t_vec[2], t_vec[1])
                        az_deg = math.degrees(az_rad)

                        # Elevation (Sin only, assume -90 to 90)
                        el_deg = math.degrees(math.asin(np.clip(t_vec[3], -1, 1)))

                        # Closure (Index 4)
                        close_kts = t_vec[4] * 2000.0

                        # ID
                        kind = "MISSILE" if t_vec[6] > 0.5 else "PLANE"
                        team = "FRIEND" if t_vec[7] > 0.5 else "ENEMY"

                        f.write(
                            f"   TRK {i} [{kind}-{team}]: Range {rng_km:.1f}km | Az {az_deg:.1f}° | El {el_deg:.1f}° | Close {close_kts:.0f}kts\n")

                    # 3. Action
                    act = blue_action[0]
                    # Roll/G/Thr/Fire/CM
                    f.write(
                        f"ACT: Roll {act[0]:.2f} | G {act[1]:.2f} | Thr {act[2]:.2f} | Fire {act[3]:.1f} | CM {act[4]:.1f}\n\n")

                    # --- STEP ENV ---
                    red_action = None
                    if "red_obs" in info:
                        r_obs = np.expand_dims(info["red_obs"], 0)
                        # Pass None for dones during inspection
                        red_action = sp_manager.get_action(r_obs, dones=None)[0]

                    if red_action is not None:
                        obs, rewards, term, trunc, info = env.step(blue_action, red_actions=red_action)
                    else:
                        obs, rewards, term, trunc, info = env.step(blue_action)

                    total_reward += rewards[0]
                    done = term[0] or trunc[0]
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
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print("Checkpoint not found (Creating dummy for test)...")
        # For verifying the script logic itself without a real training run
        from src.model import HybridActorCritic

        m = HybridActorCritic()
        torch.save(m.state_dict(), args.checkpoint)

    inspect_agent(args.checkpoint, args.output, args.steps)