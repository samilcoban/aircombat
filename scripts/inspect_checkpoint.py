#!/usr/bin/env python3
"""
Checkpoint Inspector: Detailed analysis tool for saved agent checkpoints.

This script runs a test episode with a checkpoint and logs detailed metrics to a text file:
- Observations (raw and key features)
- Actions (all 5 dimensions)
- Rewards (step-by-step breakdown)
- Physics state (G-load, speed, altitude, etc.)
- Episode outcome and statistics

Usage:
    python scripts/inspect_checkpoint.py checkpoints/model_100.pt
    python scripts/inspect_checkpoint.py checkpoints/model_latest.pt --episodes 5
"""

import argparse
import numpy as np
import torch
import os
from datetime import datetime
from collections import defaultdict

from src.env_flat import AirCombatEnv
from src.model import AgentTransformer
from config import Config


def inspect_checkpoint(checkpoint_path, num_episodes=3, output_dir="logs/inspections"):
    """
    Run test episodes with a checkpoint and log detailed metrics.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_episodes: Number of episodes to run
        output_dir: Directory to save inspection logs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract checkpoint name for log file
    checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{checkpoint_name}_{timestamp}.txt")
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = AgentTransformer().to(Config.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        update_num = checkpoint.get("update", "unknown")
    else:
        model.load_state_dict(checkpoint)
        update_num = "unknown"
    
    model.eval()
    
    # Open log file
    with open(log_path, 'w') as log:
        log.write("=" * 80 + "\n")
        log.write(f"CHECKPOINT INSPECTION REPORT\n")
        log.write("=" * 80 + "\n")
        log.write(f"Checkpoint: {checkpoint_path}\n")
        log.write(f"Update: {update_num}\n")
        log.write(f"Timestamp: {timestamp}\n")
        log.write(f"Episodes: {num_episodes}\n")
        log.write("=" * 80 + "\n\n")
        
        # Episode statistics aggregation
        all_episode_stats = []
        
        for ep in range(num_episodes):
            log.write(f"\n{'=' * 80}\n")
            log.write(f"EPISODE {ep + 1}/{num_episodes}\n")
            log.write(f"{'=' * 80}\n\n")
            
            # Create environment
            env = AirCombatEnv()
            env.set_phase(1, progress=0.0)  # Start in Phase 1
            
            obs, info = env.reset()
            done = False
            step_count = 0
            
            # Episode tracking
            episode_reward = 0.0
            episode_rewards_breakdown = defaultdict(float)
            actions_history = []
            g_loads = []
            speeds = []
            altitudes = []
            
            log.write(f"Initial State:\n")
            if env.blue_ids and env.red_ids:
                blue = env.core.entities[env.blue_ids[0]]
                red = env.core.entities[env.red_ids[0]]
                dist = np.sqrt((blue.x - red.x)**2 + (blue.y - red.y)**2) / 1000.0
                log.write(f"  Blue: pos=({blue.x:.0f}, {blue.y:.0f}), alt={blue.alt:.0f}m, "
                         f"speed={blue.speed:.0f}km/h, heading={blue.heading:.1f}°\n")
                log.write(f"  Red:  pos=({red.x:.0f}, {red.y:.0f}), alt={red.alt:.0f}m, "
                         f"speed={red.speed:.0f}km/h, heading={red.heading:.1f}°\n")
                log.write(f"  Separation: {dist:.2f} km\n\n")
            
            # Run episode
            with torch.no_grad():
                while not done and step_count < 1200:
                    # Get action from model
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                    action, logprob, entropy, value = model.get_action_and_value(obs_t)
                    action_np = action.cpu().numpy().flatten()
                    
                    # Step environment
                    obs, reward, term, trunc, info = env.step(action_np)
                    done = term or trunc
                    
                    # Track metrics
                    episode_reward += reward
                    actions_history.append(action_np)
                    
                    if env.blue_ids and env.blue_ids[0] in env.core.entities:
                        blue = env.core.entities[env.blue_ids[0]]
                        g_loads.append(blue.g_load)
                        speeds.append(blue.speed)
                        altitudes.append(blue.alt)
                    
                    # Log detailed step info every 50 steps
                    if step_count % 50 == 0:
                        log.write(f"--- Step {step_count} ---\n")
                        log.write(f"  Action: roll={action_np[0]:+.3f}, g_pull={action_np[1]:+.3f}, "
                                 f"throttle={action_np[2]:+.3f}, fire={action_np[3]:+.3f}, cm={action_np[4]:+.3f}\n")
                        log.write(f"  Reward: {reward:+.4f} (cumulative: {episode_reward:+.4f})\n")
                        log.write(f"  Value: {value.item():.4f}\n")
                        
                        if env.blue_ids and env.blue_ids[0] in env.core.entities:
                            blue = env.core.entities[env.blue_ids[0]]
                            log.write(f"  Physics: G={blue.g_load:.2f}, speed={blue.speed:.0f}km/h, "
                                     f"alt={blue.alt:.0f}m, fuel={blue.fuel:.2f}\n")
                            
                            # Find nearest enemy
                            if env.red_ids:
                                min_dist = float('inf')
                                for red_id in env.red_ids:
                                    if red_id in env.core.entities:
                                        red = env.core.entities[red_id]
                                        dist = np.sqrt((blue.x - red.x)**2 + (blue.y - red.y)**2) / 1000.0
                                        if dist < min_dist:
                                            min_dist = dist
                                log.write(f"  Nearest enemy: {min_dist:.2f} km\n")
                        log.write("\n")
                    
                    step_count += 1
            
            # Episode summary
            log.write(f"\n{'=' * 80}\n")
            log.write(f"EPISODE {ep + 1} SUMMARY\n")
            log.write(f"{'=' * 80}\n")
            log.write(f"Outcome: {info.get('termination_reason', 'unknown')}\n")
            log.write(f"Steps: {step_count}\n")
            log.write(f"Total Reward: {episode_reward:+.4f}\n\n")
            
            # Action statistics
            if actions_history:
                actions_array = np.array(actions_history)
                log.write(f"Action Statistics:\n")
                log.write(f"  Roll:     mean={actions_array[:, 0].mean():+.3f}, std={actions_array[:, 0].std():.3f}, "
                         f"min={actions_array[:, 0].min():+.3f}, max={actions_array[:, 0].max():+.3f}\n")
                log.write(f"  G-Pull:   mean={actions_array[:, 1].mean():+.3f}, std={actions_array[:, 1].std():.3f}, "
                         f"min={actions_array[:, 1].min():+.3f}, max={actions_array[:, 1].max():+.3f}\n")
                log.write(f"  Throttle: mean={actions_array[:, 2].mean():+.3f}, std={actions_array[:, 2].std():.3f}, "
                         f"min={actions_array[:, 2].min():+.3f}, max={actions_array[:, 2].max():+.3f}\n")
                log.write(f"  Fire:     mean={actions_array[:, 3].mean():+.3f}, std={actions_array[:, 3].std():.3f}, "
                         f"max={actions_array[:, 3].max():+.3f}\n")
                log.write(f"  CM:       mean={actions_array[:, 4].mean():+.3f}, std={actions_array[:, 4].std():.3f}\n\n")
            
            # Physics statistics
            if g_loads:
                log.write(f"Physics Statistics:\n")
                log.write(f"  G-Load:   mean={np.mean(g_loads):.2f}, max={np.max(g_loads):.2f}, "
                         f"min={np.min(g_loads):.2f}\n")
                log.write(f"  Speed:    mean={np.mean(speeds):.0f} km/h, max={np.max(speeds):.0f}, "
                         f"min={np.min(speeds):.0f}\n")
                log.write(f"  Altitude: mean={np.mean(altitudes):.0f} m, max={np.max(altitudes):.0f}, "
                         f"min={np.min(altitudes):.0f}\n\n")
                
                # G-load distribution
                g_bins = [0, 1.5, 3.0, 6.0, 9.0, 15.0]
                g_hist, _ = np.histogram(g_loads, bins=g_bins)
                log.write(f"  G-Load Distribution:\n")
                for i in range(len(g_bins) - 1):
                    pct = 100.0 * g_hist[i] / len(g_loads)
                    log.write(f"    {g_bins[i]:.1f}-{g_bins[i+1]:.1f}G: {pct:5.1f}% ({g_hist[i]} steps)\n")
            
            # Store episode stats
            all_episode_stats.append({
                'reward': episode_reward,
                'steps': step_count,
                'outcome': info.get('termination_reason', 'unknown'),
                'mean_g': np.mean(g_loads) if g_loads else 0,
                'max_g': np.max(g_loads) if g_loads else 0,
                'mean_speed': np.mean(speeds) if speeds else 0,
            })
            
            env.close()
        
        # Overall summary
        log.write(f"\n\n{'=' * 80}\n")
        log.write(f"OVERALL SUMMARY ({num_episodes} episodes)\n")
        log.write(f"{'=' * 80}\n")
        
        avg_reward = np.mean([s['reward'] for s in all_episode_stats])
        avg_steps = np.mean([s['steps'] for s in all_episode_stats])
        avg_g = np.mean([s['mean_g'] for s in all_episode_stats])
        max_g = np.max([s['max_g'] for s in all_episode_stats])
        
        log.write(f"Average Reward: {avg_reward:+.4f}\n")
        log.write(f"Average Steps: {avg_steps:.0f}\n")
        log.write(f"Average G-Load: {avg_g:.2f}\n")
        log.write(f"Max G-Load: {max_g:.2f}\n\n")
        
        # Outcome distribution
        outcomes = [s['outcome'] for s in all_episode_stats]
        outcome_counts = {}
        for outcome in outcomes:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        log.write(f"Outcome Distribution:\n")
        for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / num_episodes
            log.write(f"  {outcome}: {count}/{num_episodes} ({pct:.1f}%)\n")
        
        log.write(f"\n{'=' * 80}\n")
        log.write(f"END OF REPORT\n")
        log.write(f"{'=' * 80}\n")
    
    print(f"\n✅ Inspection complete!")
    print(f"📄 Report saved to: {log_path}")
    print(f"\nQuick Summary:")
    print(f"  Average Reward: {avg_reward:+.4f}")
    print(f"  Average Steps: {avg_steps:.0f}")
    print(f"  Average G-Load: {avg_g:.2f} (max: {max_g:.2f})")
    print(f"  Outcomes: {outcome_counts}")
    
    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect agent checkpoint with detailed logging")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
    parser.add_argument("--output-dir", type=str, default="logs/inspections", 
                       help="Directory to save inspection logs")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    inspect_checkpoint(args.checkpoint, args.episodes, args.output_dir)
