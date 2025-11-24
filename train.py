import gymnasium as gym
import numpy as np
import torch
import os
import time
import imageio
import glob
import re
from tqdm import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter

from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.ppo import PPOAgent
from src.self_play import SelfPlayManager
from config import Config


def make_env():
    return AirCombatEnv()


def save_validation_gif(model, step):
    """Runs a validation episode and saves a GIF + 3D visualization."""
    print("Generating Validation GIF and 3D Visualization...")
    from src.render_3d import Render3D
    
    env = AirCombatEnv()
    renderer_3d = Render3D(env.map_limits)
    renderer_3d.reset()
    
    obs, _ = env.reset()
    frames = []
    frames_3d = []
    step_count = 0

    done = False
    model.eval()
    with torch.no_grad():
        while not done:
            # Update 3D trajectories (for static trails if needed, but animation handles it)
            # actually create_animation calls create_figure which uses self.trajectories
            # so we still need to update trajectories!
            renderer_3d.update_trajectories(env.core.entities, env.core.time)
            
            # Capture 3D frame data (Deep Copy is CRITICAL)
            if step_count % 5 == 0:
                frames_3d.append({
                    'entities': copy.deepcopy(env.core.entities),
                    'time': env.core.time,
                    'step': step_count
                })

            # 2D frame for GIF
            frame = env.render()
            # Ensure standard uint8 format for imageio
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            if frame.shape[2] == 4:  # RGBA -> RGB
                frame = frame[:, :, :3]

            frames.append(frame)

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            action, _, _, _ = model.get_action_and_value(obs_t)
            obs, _, term, trunc, _ = env.step(action.cpu().numpy().flatten())
            done = term or trunc
            step_count += 1

    # Save 2D GIF
    gif_path = f"checkpoints/val_{step}.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved 2D: {gif_path}")
    
    # Save 3D Animation
    html_path = f"checkpoints/val_{step}_3d.html"
    renderer_3d.create_animation(frames_3d, html_path)
    print(f"Saved 3D Animation: {html_path}")
    
    env.close()
    model.train()


def load_latest_checkpoint(model, optimizer):
    """
    Searches 'checkpoints/' for the latest model_X.pt.
    Loads weights and returns the next update number.
    """
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        return 1

    files = glob.glob("checkpoints/model_*.pt")
    if not files:
        print("--- No checkpoints found. Starting fresh. ---")
        return 1

    # Extract update number using Regex to find max
    # Matches 'model_100.pt' -> 100
    latest_file = max(files, key=lambda f: int(re.search(r'model_(\d+).pt', f).group(1)))
    update_num = int(re.search(r'model_(\d+).pt', latest_file).group(1))

    print(f"--- Resuming from: {latest_file} (Update {update_num}) ---")

    checkpoint = torch.load(latest_file, map_location=Config.DEVICE)

    # Handle new dictionary format vs old direct state_dict format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        # Fallback for checkpoints saved before this update
        print("Warning: Loading legacy checkpoint (Model weights only). Optimizer reset.")
        model.load_state_dict(checkpoint)

    return update_num + 1


def train():
    run_name = f"AirCombat_Istanbul_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # 1. Vector Env
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(Config.NUM_ENVS)])

    # 2. Model & Agent
    model = AgentTransformer().to(Config.DEVICE)
    agent = PPOAgent(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE.type == 'cuda'))
    
    # 2.5 Self-Play Manager
    sp_manager = SelfPlayManager()
    # Initial sample (likely Scripted if step < 1M)
    sp_manager.sample_opponent(0)
    print(f"Self-Play Manager Initialized. Opponent: {sp_manager.current_opponent_name}")

    # 3. Auto-Resume Logic
    start_update = load_latest_checkpoint(model, agent.optimizer)

    # 4. Buffers
    next_obs, next_info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(Config.DEVICE)
    next_done = torch.zeros(Config.NUM_ENVS).to(Config.DEVICE)

    global_step = (start_update - 1) * Config.BATCH_SIZE
    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE

    print(f"Starting training loop from Update {start_update} to {num_updates}...")

    for update in tqdm(range(start_update, num_updates + 1)):
        storage_obs = []
        storage_actions = []
        storage_logprobs = []
        storage_rewards = []
        storage_dones = []
        storage_values = []

        # --- Data Collection ---
        # Calculate steps per env to reach BATCH_SIZE
        steps_per_env = Config.BATCH_SIZE // Config.NUM_ENVS

        for step in range(steps_per_env):
            global_step += Config.NUM_ENVS

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs)
                values = value.flatten()

            # --- Broadcast Kappa (Curriculum) ---
            current_kappa = sp_manager.kappa if hasattr(sp_manager, 'kappa') else 0.0
            # Use call_async to set kappa in all envs
            envs.call("set_kappa", current_kappa)

            # --- Self-Play: Get Red Actions ---
            # Extract Red Obs from info
            # AsyncVectorEnv stacks info: {'red_obs': array([...])}
            red_actions = None
            if "red_obs" in next_info:
                red_obs = next_info["red_obs"]
                red_actions = sp_manager.get_action(red_obs)
            
            # If Red Actions are None (Scripted AI), pass ONLY Blue Actions
            if red_actions is None:
                real_next_obs, reward, term, trunc, next_info = envs.step(action.cpu().numpy())
            else:
                # Concatenate Actions
                blue_actions_np = action.cpu().numpy()
                # Ensure shapes match
                if red_actions.shape[0] != blue_actions_np.shape[0]:
                     red_actions = np.zeros_like(blue_actions_np)

                concat_actions = np.concatenate([blue_actions_np, red_actions], axis=1)
                real_next_obs, reward, term, trunc, next_info = envs.step(concat_actions)
            
            done = np.logical_or(term, trunc)

            storage_obs.append(next_obs)
            storage_actions.append(action)
            storage_logprobs.append(logprob)
            storage_rewards.append(torch.tensor(reward).to(Config.DEVICE))
            storage_dones.append(next_done)
            storage_values.append(values)

            next_obs = torch.Tensor(real_next_obs).to(Config.DEVICE)
            next_done = torch.Tensor(done).to(Config.DEVICE)

        # --- GAE Calculation ---
        with torch.no_grad():
            next_value = model.get_value(next_obs).reshape(1, -1)

        storage_rewards = torch.stack(storage_rewards)
        storage_dones = torch.stack(storage_dones)
        storage_values = torch.stack(storage_values)

        advantages = torch.zeros_like(storage_rewards).to(Config.DEVICE)
        lastgaelam = 0
        num_steps = storage_rewards.shape[0]

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - storage_dones[t + 1]
                nextvalues = storage_values[t + 1]

            delta = storage_rewards[t] + Config.GAMMA * nextvalues * nextnonterminal - storage_values[t]
            advantages[t] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * nextnonterminal * lastgaelam

        returns = advantages + storage_values

        # Flatten Tensors
        b_obs = torch.stack(storage_obs).reshape(-1, Config.OBS_DIM)
        b_actions = torch.stack(storage_actions).reshape(-1, Config.ACTION_DIM)
        b_logprobs = torch.stack(storage_logprobs).reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # --- Update ---
        # We pass GPU tensors directly to avoid CPU copy overhead
        # Mixed Precision Update
        with torch.cuda.amp.autocast(enabled=(Config.DEVICE.type == 'cuda')):
            loss = agent.update(
                b_obs,
                b_actions,
                b_logprobs,
                b_returns,
                b_advantages,
                scaler=scaler
            )
        
        # Scaler Step (if using AMP, otherwise standard)
        # Note: PPOAgent.update usually does optimizer.step(). 
        # We need to modify PPOAgent to accept scaler or handle it here.
        # Since PPOAgent is in src/ppo.py, let's check it first.
        # Assuming standard PPO implementation, we might need to refactor `agent.update`.
        # For now, let's assume we need to modify src/ppo.py as well.
        # But wait, if `agent.update` encapsulates the backward pass, we can't just wrap it in autocast 
        # if the backward pass is inside.
        # Let's check src/ppo.py first.


        # --- Logging ---
        writer.add_scalar("charts/loss", loss, global_step)
        writer.add_scalar("charts/mean_step_reward", storage_rewards.mean().item(), global_step)
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
        
        # Action statistics (to monitor firing behavior)
        actions_np = b_actions.cpu().numpy()
        writer.add_scalar("actions/fire_mean", actions_np[:, 3].mean(), global_step)
        writer.add_scalar("actions/fire_std", actions_np[:, 3].std(), global_step)
        writer.add_scalar("actions/throttle_mean", actions_np[:, 2].mean(), global_step)
        writer.add_scalar("actions/g_pull_mean", actions_np[:, 1].mean(), global_step)

        # --- Save & Validate ---
        # --- Save & Validate ---
        if update % Config.SAVE_INTERVAL == 0:
            # AOS Gate Function: Evaluate before saving
            if sp_manager.evaluate_candidate(model, make_env):
                # Save robust checkpoint with optimizer state
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'update': update
                }
                torch.save(checkpoint_data, f"checkpoints/model_{update}.pt")
                print(f"--- Checkpoint saved: model_{update}.pt ---")

                # Render GIF (Only for accepted models)
                save_validation_gif(model, update)
                
                # Update Self-Play Opponent Pool
                sp_manager.load_checkpoints_list()
            else:
                print(f"--- Candidate failed evaluation. Checkpoint discarded. ---")
            
            # Sample new opponent for next phase (regardless of acceptance)
            sp_manager.sample_opponent(global_step)
            print(f"--- New Opponent: {sp_manager.current_opponent_name} ---")

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()