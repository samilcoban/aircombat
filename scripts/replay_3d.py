#!/usr/bin/env python3
"""
3D Replay Script for Aerial Combat Episodes
Generates interactive 3D HTML visualizations from saved episodes.
"""

import numpy as np
import torch
import argparse
import copy
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.render_3d import Render3D
from config import Config


def replay_episode(model_path=None, output_html="replay_3d.html", max_steps=500):
    """
    Replay an episode and generate 3D visualization.
    
    Args:
        model_path: Path to model checkpoint (None for random policy)
        output_html: Output HTML filename
        max_steps: Maximum steps to run
    """
    print("Initializing environment...")
    env = AirCombatEnv()
    renderer = Render3D(env.map_limits)
    
    # Load model if provided
    model = None
    if model_path:
        print(f"Loading model from {model_path}...")
        model = AgentTransformer().to(Config.DEVICE)
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("Model loaded successfully!")
    else:
        print("No model provided, using random policy...")
    
    # Run episode
    print("Running episode...")
    obs, _ = env.reset()
    renderer.reset()
    
    done = False
    step = 0
    
    # Store frames for animation
    frames = []
    
    while not done and step < max_steps:
        # Update trajectories
        renderer.update_trajectories(env.core.entities, env.core.time)
        
        # Get action
        if model is not None:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                action, _, _, _ = model.get_action_and_value(obs_t)
                action = action.cpu().numpy().flatten()
        else:
            action = env.action_space.sample()
        
        # Step environment
        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        step += 1
        
        # Store frame data every N steps for animation
        if step % 5 == 0:
            frames.append({
                'entities': copy.deepcopy(env.core.entities),
                'time': env.core.time,
                'step': step
            })
    
    print(f"Episode finished after {step} steps")
    print(f"Final entities alive: {len(env.core.entities)}")
    
    # Create final visualization
    print("Generating 3D visualization...")
    fig = renderer.create_figure(
        env.core.entities,
        title=f"Aerial Combat 3D Replay (Step {step}, Time {env.core.time:.1f}s)"
    )
    
    # Save to HTML
    renderer.save_html(fig, output_html)
    print(f"✅ 3D visualization saved to: {output_html}")
    print(f"   Open this file in your browser to view the interactive 3D scene!")
    
    # Generate animation frames if we have enough
    if len(frames) > 10:
        print("\nGenerating animated visualization...")
        renderer.create_animation(frames, "replay_3d_animated.html")
    
    env.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay aerial combat episode in 3D")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model checkpoint (default: random policy)")
    parser.add_argument("--output", type=str, default="replay_3d.html",
                       help="Output HTML filename")
    parser.add_argument("--steps", type=int, default=500,
                       help="Maximum steps to run")
    
    args = parser.parse_args()
    
    replay_episode(args.model, args.output, args.steps)
