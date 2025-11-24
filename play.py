import argparse
import torch
import numpy as np
import imageio
import os
from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.self_play import SelfPlayManager
from config import Config

def play(checkpoint_path, output_path="replay.mp4"):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load Model
    model = AgentTransformer().to(Config.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Init Self-Play (for opponent)
    sp_manager = SelfPlayManager()
    sp_manager.sample_opponent()
    print(f"Opponent: {sp_manager.current_opponent_name}")
    
    # Init Env
    env = AirCombatEnv()
    obs, info = env.reset()
    
    frames = []
    done = False
    step = 0
    
    print("Rendering episode...")
    with torch.no_grad():
        while not done:
            # Render
            frame = env.render()
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            if frame.shape[2] == 4: frame = frame[:, :, :3]
            frames.append(frame)
            
            # Actions
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            action, _, _, _ = model.get_action_and_value(obs_t)
            blue_action = action.cpu().numpy().flatten()
            
            # Red Action
            if "red_obs" in info:
                red_obs_batch = np.expand_dims(info["red_obs"], axis=0)
                red_action_batch = sp_manager.get_action(red_obs_batch)
                red_action = red_action_batch[0]
            else:
                red_action = np.zeros(Config.ACTION_DIM)
            
            # Step
            concat_action = np.concatenate([blue_action, red_action])
            obs, reward, term, trunc, info = env.step(concat_action)
            done = term or trunc
            step += 1
            
            if step % 100 == 0: print(f"Step {step}...")
            
    env.close()
    
    print(f"Saving replay to {output_path}...")
    imageio.mimsave(output_path, frames, fps=30)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="replay.mp4", help="Output video path")
    args = parser.parse_args()
    
    play(args.checkpoint, args.output)
