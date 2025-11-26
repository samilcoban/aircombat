#!/usr/bin/env python3
"""
Inspect agent behavior by logging observations, actions, and rewards to a file.
This helps debug issues like spinning, crashing, or poor flight control.
"""

import argparse
import torch
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.self_play import SelfPlayManager
from config import Config

def inspect_agent(checkpoint_path, output_file="agent_inspection.txt", max_steps=500):
    """
    Run agent and log detailed state information for debugging.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_file: Output text file for logs
        max_steps: Maximum steps to run before stopping
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load Model
    model = AgentTransformer().to(Config.DEVICE)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Handle torch.compile() wrapped models
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Detected compiled model checkpoint, stripping _orig_mod prefix...")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.eval()
    
    # Init Self-Play (for opponent)
    sp_manager = SelfPlayManager()
    sp_manager.sample_opponent()
    print(f"Opponent: {sp_manager.current_opponent_name}")
    
    # Init Env
    env = AirCombatEnv()
    obs, info = env.reset()
    
    # Open log file
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AGENT INSPECTION LOG\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Opponent: {sp_manager.current_opponent_name}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write observation space description (Corrected based on src/env.py)
        f.write("OBSERVATION SPACE (Corrected):\n")
        f.write("  [0-1]:   Position (lat, lon) [normalized]\n")
        f.write("  [2-3]:   Heading (cos, sin)\n")
        f.write("  [4]:     Speed [normalized]\n")
        f.write("  [5]:     Team (+1 blue, -1 red)\n")
        f.write("  [6]:     Type (1 missile, 0 plane)\n")
        f.write("  [7]:     Is Ego\n")
        f.write("  [8-9]:   Roll (cos, sin)\n")
        f.write("  [10-11]: Pitch (cos, sin)\n")
        f.write("  [12]:    RWR Signal\n")
        f.write("  [13]:    MAWS Signal\n")
        f.write("  [14]:    Altitude [normalized]\n")
        f.write("  [15]:    Fuel\n")
        f.write("  [16]:    Ammo\n")
        f.write("  [17]:    ATA (Antenna Train Angle)\n")
        f.write("  [18]:    AA (Aspect Angle)\n")
        f.write("  [19]:    Closure Rate\n\n")
        
        f.write("ACTION SPACE (4 dimensions):\n")
        f.write("  [0]: Roll Rate [-1, 1] (negative=left, positive=right)\n")
        f.write("  [1]: Pitch Rate [-1, 1] (negative=down, positive=up)\n")
        f.write("  [2]: Throttle [0, 1] (0=idle, 1=max)\n")
        f.write("  [3]: Fire [0, 1] (>0.5 = fire missile)\n\n")
        f.write("=" * 80 + "\n\n")
        
        done = False
        step = 0
        total_reward = 0.0
        
        print(f"Running inspection (max {max_steps} steps)...")
        print(f"Logging to: {output_file}")
        
        try:
            with torch.no_grad():
                while not done and step < max_steps:
                    # Get action from model
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                    action, _, _, _ = model.get_action_and_value(obs_t)
                    blue_action = action.cpu().numpy().flatten()
                    
                    # Log current state
                    f.write(f"STEP {step}\n")
                    f.write("-" * 80 + "\n")
                    
                    # Parse and log key observations (Corrected indices)
                    # Note: obs is flattened. Ego is usually the first entity (first 20+ features)
                    # We assume index 0-19 corresponds to Ego state
                    
                    f.write(f"Own State:\n")
                    f.write(f"  Position (norm): ({obs[0]:.3f}, {obs[1]:.3f})\n")
                    f.write(f"  Heading (cos, sin): ({obs[2]:.3f}, {obs[3]:.3f})\n")
                    f.write(f"  Speed (norm): {obs[4]:.3f} (~{obs[4]*1000:.1f} km/h)\n")
                    f.write(f"  Altitude (norm): {obs[14]:.3f} (~{obs[14]*10000:.1f} m)\n")
                    f.write(f"  Roll (cos, sin): ({obs[8]:.3f}, {obs[9]:.3f})\n")
                    f.write(f"  Pitch (cos, sin): ({obs[10]:.3f}, {obs[11]:.3f})\n")
                    f.write(f"  Fuel: {obs[15]:.3f}\n")
                    f.write(f"  Ammo: {obs[16]:.3f}\n")
                    
                    f.write(f"\nSensors:\n")
                    f.write(f"  RWR (Locked): {obs[12]:.1f}\n")
                    f.write(f"  MAWS (Missile): {obs[13]:.1f}\n")
                    
                    f.write(f"\nGeometry:\n")
                    f.write(f"  ATA: {obs[17]:.3f}\n")
                    f.write(f"  AA: {obs[18]:.3f}\n")
                    f.write(f"  Closure: {obs[19]:.3f}\n")
                    
                    f.write(f"\nAction Taken:\n")
                    f.write(f"  Roll Rate: {blue_action[0]:+.3f} (left < 0 < right)\n")
                    f.write(f"  Pitch Rate: {blue_action[1]:+.3f} (down < 0 < up)\n")
                    f.write(f"  Throttle: {blue_action[2]:.3f} (0=idle, 1=max)\n")
                    f.write(f"  Fire: {blue_action[3]:.3f} ({'FIRE!' if blue_action[3] > 0.0 else 'hold'})\n")
                    
                    # Get red action if available
                    red_action_batch = None
                    if "red_obs" in info:
                        red_obs_batch = np.expand_dims(info["red_obs"], axis=0)
                        red_action_batch = sp_manager.get_action(red_obs_batch)
                    
                    # Step environment
                    if red_action_batch is not None:
                        red_action = red_action_batch[0]
                        concat_action = np.concatenate([blue_action, red_action])
                        obs, reward, term, trunc, info = env.step(concat_action)
                    else:
                        obs, reward, term, trunc, info = env.step(blue_action)
                    
                    total_reward += reward
                    done = term or trunc
                    
                    f.write(f"\nResult:\n")
                    f.write(f"  Reward: {reward:+.4f}\n")
                    f.write(f"  Cumulative Reward: {total_reward:+.4f}\n")
                    f.write(f"  Terminated: {term}\n")
                    f.write(f"  Truncated: {trunc}\n")
                    
                    if done:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"EPISODE ENDED at step {step}\n")
                        f.write(f"Total Reward: {total_reward:+.4f}\n")
                        f.write(f"Reason: {'Terminated' if term else 'Truncated'}\n")
                        f.write(f"{'='*80}\n")
                    
                    f.write("\n")
                    step += 1
                    
                    # Print progress every 50 steps
                    if step % 50 == 0:
                        print(f"  Step {step}/{max_steps}, Reward: {total_reward:+.2f}")
                        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            f.write(f"\n{'='*80}\n")
            f.write(f"INTERRUPTED at step {step}\n")
            f.write(f"{'='*80}\n")
        
        env.close()
        
    print(f"\n✅ Inspection complete!")
    print(f"   Steps: {step}")
    print(f"   Total Reward: {total_reward:+.4f}")
    print(f"   Log saved to: {output_file}")
    print(f"\nTo view the log:")
    print(f"   less {output_file}")
    print(f"   # or")
    print(f"   cat {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect agent behavior with detailed logging")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="agent_inspection.txt", help="Output log file")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps to run")
    args = parser.parse_args()
    
    inspect_agent(args.checkpoint, args.output, args.max_steps)
