import argparse
import torch
import numpy as np
import time
from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.self_play import SelfPlayManager
from config import Config
from src.render_panda3d import Panda3DRenderer

def play(checkpoint_path, output_path="replay.mp4"):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load Model
    model = AgentTransformer().to(Config.DEVICE)
    # Handle loading both full dict checkpoints and raw state dicts
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Handle torch.compile() wrapped models (keys have "_orig_mod." prefix)
        # Strip the prefix if present to load into uncompiled model
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Detected compiled model checkpoint, stripping _orig_mod prefix...")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Init Self-Play (for opponent)
    sp_manager = SelfPlayManager()
    sp_manager.sample_opponent()
    print(f"Opponent: {sp_manager.current_opponent_name}")
    
    # Init Env
    env = AirCombatEnv()
    obs, info = env.reset()
    
    # === PANDA3D SETUP ===
    renderer = Panda3DRenderer()
    # =====================
    
    done = False
    step = 0
    
    print("Running simulation...")
    try:
        with torch.no_grad():
            while not done:
                # Blue Action (Model)
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                action, _, _, _ = model.get_action_and_value(obs_t)
                blue_action = action.cpu().numpy().flatten()
                
                # Red Action (Self-Play Manager)
                red_action_batch = None
                if "red_obs" in info:
                    red_obs_batch = np.expand_dims(info["red_obs"], axis=0)
                    red_action_batch = sp_manager.get_action(red_obs_batch)
                
                # === THE FIX IS HERE ===
                # If red_action_batch is None, it means we are fighting a Scripted Bot.
                # We pass ONLY the blue_action to env.step(), and the Env handles the AI.
                if red_action_batch is not None:
                    # Fighting a Model Opponent
                    red_action = red_action_batch[0]
                    concat_action = np.concatenate([blue_action, red_action])
                    obs, reward, term, trunc, info = env.step(concat_action)
                else:
                    # Fighting Scripted/Random Bot (Env internal AI)
                    obs, reward, term, trunc, info = env.step(blue_action)
                # ========================
                
                done = term or trunc
                step += 1
                
                # === UPDATE PANDA3D RENDERER ===
                renderer.update_entities(env.core.entities, Config.MAP_LIMITS)
                
                # Process Panda3D events and render
                renderer.taskMgr.step()
                
                # Check if window is still open
                if not renderer.check_running():
                    print("Window closed by user")
                    break
                # ================================ 

                if done:
                    print(f"Episode done in {step} steps. Resetting...")
                    obs, info = env.reset()
                    done = False
                    step = 0
                    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        renderer.cleanup()
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="replay.mp4", help="Output video path (unused in 3D visualization mode)")
    args = parser.parse_args()
    
    play(args.checkpoint, args.output)