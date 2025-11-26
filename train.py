# Import standard libraries for environment management, computation, and utilities
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

# Import custom modules for air combat RL training
from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.ppo import PPOAgent
from src.self_play import SelfPlayManager
from config import Config


def make_env():
    """
    Environment factory function for vectorized environments.
    
    Returns:
        AirCombatEnv: Fresh environment instance
    """
    return AirCombatEnv()


def save_validation_gif(model, step):
    """
    Generate validation episode visualization as 2D GIF and 3D HTML animation.
    
    Runs a single episode with the current policy (greedy, no exploration)
    and saves the trajectory for visual inspection. Useful for:
    - Monitoring agent behavior evolution
    - Debugging tactics and failure modes
    - Creating demonstration videos
    
    Args:
        model: Trained AgentTransformer network
        step: Current training step (for filename)
    """
    print("Generating Validation GIF and 3D Visualization...")
    from src.render_3d import Render3D
    
    # Create fresh environment and 3D renderer
    env = AirCombatEnv()
    renderer_3d = Render3D(env.map_limits)
    renderer_3d.reset()
    
    # Initialize episode
    obs, _ = env.reset()
    frames = []      # 2D frames for GIF
    frames_3d = []   # 3D frame data for animation
    step_count = 0

    done = False
    model.eval()  # Set to evaluation mode (no dropout, etc.)
    with torch.no_grad():  # Disable gradient computation for inference
        while not done:
            # Update 3D trajectory tracking for animated visualization
            # (Trajectories are stored internally and rendered in create_animation())
            renderer_3d.update_trajectories(env.core.entities, env.core.time)
            
            # Capture 3D frame snapshot (every 5 steps to reduce file size)
            # CRITICAL: Deep copy entity state to prevent mutation
            if step_count % 5 == 0:
                frames_3d.append({
                    'entities': copy.deepcopy(env.core.entities),
                    'time': env.core.time,
                    'step': step_count
                })

            # Capture 2D frame for GIF
            frame = env.render()
            # Ensure standard uint8 format for imageio compatibility
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            if frame.shape[2] == 4:  # RGBA → RGB (remove alpha channel)
                frame = frame[:, :, :3]

            frames.append(frame)

            # Get greedy action from policy (deterministic, no sampling)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            action, _, _, _ = model.get_action_and_value(obs_t)
            
            # Step environment with policy action
            obs, _, term, trunc, _ = env.step(action.cpu().numpy().flatten())
            done = term or trunc
            step_count += 1

    # Save 2D GIF (top-down tactical view)
    gif_path = f"checkpoints/val_{step}.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved 2D: {gif_path}")
    
    # Save 3D Animation (interactive HTML with Plotly)
    html_path = f"checkpoints/val_{step}_3d.html"
    renderer_3d.create_animation(frames_3d, html_path)
    print(f"Saved 3D Animation: {html_path}")
    
    env.close()
    model.train()  # Restore to training mode


def load_latest_checkpoint(model, optimizer):
    """
    Auto-resume training from latest checkpoint.
    
    Searches for the most recent numbered checkpoint (model_X.pt) and
    restores model weights, optimizer state, and training progress.
    Ignores "model_latest.pt" which is used as a safety backup.
    
    Args:
        model: AgentTransformer to load weights into
        optimizer: Adam optimizer to restore state
        
    Returns:
        int: Next update number to continue from (checkpoint_num + 1)
    """
    # Create checkpoints directory if missing
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        return 1  # Start fresh

    # Find all checkpoint files
    files = glob.glob("checkpoints/model_*.pt")
    
    # Filter to numbered checkpoints only (ignore model_latest.pt)
    # This prevents loading the "safety net" checkpoint during normal resumption
    numbered_files = []
    for f in files:
        if re.search(r'model_(\d+).pt', f):  # Matches model_100.pt, not model_latest.pt
            numbered_files.append(f)
            
    if not numbered_files:
        print("--- No numbered checkpoints found. Starting fresh. ---")
        return 1

    # Find checkpoint with highest update number
    # Extract number using regex: "model_100.pt" → 100
    latest_file = max(numbered_files, key=lambda f: int(re.search(r'model_(\d+).pt', f).group(1)))
    update_num = int(re.search(r'model_(\d+).pt', latest_file).group(1))

    print(f"--- Resuming from: {latest_file} (Update {update_num}) ---")

    # Load checkpoint (map to current device)
    checkpoint = torch.load(latest_file, map_location=Config.DEVICE)

    # Handle checkpoint format (new dict vs old direct state_dict)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New format: includes model + optimizer + metadata
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        # Old format: just model weights (legacy compatibility)
        print("Warning: Loading legacy checkpoint (Model weights only). Optimizer reset.")
        model.load_state_dict(checkpoint)

    return update_num + 1  # Return next update to continue from


def train():
    """
    Main PPO training loop with self-play and curriculum learning.
    
    Training Pipeline:
    1. Initialize vectorized environments (parallel rollout collection  )
    2. Create model, PPO agent, and self-play manager
    3. Auto-resume from latest checkpoint if exists
    4. Training loop:
       a. Collect rollout data from vectorized envs with self-play opponents
       b. Compute advantages using Generalized Advantage Estimation (GAE)
       c. Update policy with PPO clipped objective
       d. Evaluate and save checkpoints via AOS (Accept/Reject gate)
       e. Sample new opponent for next iteration (curriculum learning)
    
    Key Features:
    - Vectorized environments for parallel data collection  
    - Self-play with curriculum learning (kappa parameter)
    - Mixed precision training (FP16 on GPU)
    - Automatic checkpointing with AOS quality gate
    - TensorBoard logging for monitoring
    """
    # === SETUP ===
    # Create unique run name for TensorBoard
    run_name = f"AirCombat_Istanbul_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # 1. Vectorized Environments
    # AsyncVectorEnv runs environments in parallel processes for faster rollout collection
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(Config.NUM_ENVS)])

    # 2. Model & PPO Agent
    model = AgentTransformer().to(Config.DEVICE)
    
    # 2.1 Compile Model for Speed (PyTorch 2.0+)
    # This provides 2-3x speedup on forward/backward passes
    # Critical for reducing 94s/iteration to <30s
    is_compiled = False
    try:
        model = torch.compile(model)
        is_compiled = True
        print("✅ Model compiled with torch.compile() for speedup")
    except Exception as e:
        print(f"⚠️  torch.compile() failed: {e}. Continuing without compilation.")
    
    agent = PPOAgent(model)
    # Mixed precision scaler for GPU acceleration (fp16/fp32 mix)
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE.type == 'cuda'))
    
    # 2.5 Self-Play Manager
    # Manages opponent pool and curriculum difficulty (kappa)
    sp_manager = SelfPlayManager()
    # Initial opponent sample (likely scripted AI for early training)
    sp_manager.sample_opponent(0)
    print(f"Self-Play Manager Initialized. Opponent: {sp_manager.current_opponent_name}")

    # 3. Auto-Resume Logic
    # Load latest checkpoint and continue training  
    start_update = load_latest_checkpoint(model, agent.optimizer)

    # 4. Rollout Buffers
    # Initialize first observation from all parallel environments
    next_obs, next_info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(Config.DEVICE)
    next_done = torch.zeros(Config.NUM_ENVS).to(Config.DEVICE)

    # Calculate training progress
    global_step = (start_update - 1) * Config.BATCH_SIZE
    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE

    print(f"Starting training loop from Update {start_update} to {num_updates}...")

    # === MAIN TRAINING LOOP ===
    for update in tqdm(range(start_update, num_updates + 1)):
        # Rollout storage (will be filled during data collection)
        storage_obs = []
        storage_actions = []
        storage_logprobs = []
        storage_rewards = []
        storage_dones = []
        storage_dones = []
        storage_values = []
        storage_global_states = []  # CTDE: Store global states for critic training

        # Broadcast Curriculum Parameters to all environments
        # Kappa controls AI opponent difficulty (1.0 = easy, 0.0 = expert)
        current_kappa = sp_manager.kappa if hasattr(sp_manager, 'kappa') else 0.0
        
        # Determine current phase and adjust kappa for Phase 3 (gentle maneuvering)
        current_phase = sp_manager.get_current_phase(global_step)
        if current_phase == 3:
            # Phase 3: Force gentle opponent (prevent aggressive maneuvers)
            current_kappa = max(current_kappa, 0.8)
        
        # Broadcast to all parallel environments
        envs.call("set_kappa", current_kappa)
        envs.call("set_phase", current_phase)

        # --- Data Collection ---
        # Calculate steps per env to reach BATCH_SIZE
        steps_per_env = Config.BATCH_SIZE // Config.NUM_ENVS

        for step in range(steps_per_env):
            global_step += Config.NUM_ENVS

            with torch.no_grad():
                # CTDE: Extract global state if available (from previous step's info)
                # For first step, we might not have it, so fallback to obs (or zeros)
                # Actually, env.reset() returns info now? Yes.
                # But we need to handle the first step case.
                # Let's assume next_info has it if we updated env.reset()
                
                # Get global state from info, or fallback to local obs (masked)
                global_state = next_info.get("global_state", next_obs.cpu().numpy())
                
                # Robustness: Handle AsyncVectorEnv object array issues
                if isinstance(global_state, np.ndarray) and global_state.dtype == np.object_:
                    try:
                        # Try to force conversion (e.g. if it's a list of arrays)
                        global_state = np.array(global_state.tolist(), dtype=np.float32)
                    except Exception as e:
                        print(f"Warning: Global state object conversion failed: {e}. Fallback to local obs.")
                        global_state = next_obs.cpu().numpy()

                # Ensure tensor on device
                if isinstance(global_state, np.ndarray):
                    global_state_t = torch.tensor(global_state, dtype=torch.float32).to(Config.DEVICE)
                else:
                    global_state_t = global_state.to(Config.DEVICE)
                
                # Pass both local obs (actor) and global state (critic)
                action, logprob, _, value = model.get_action_and_value(next_obs, global_state=global_state_t)
                values = value.flatten()

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
            storage_dones.append(next_done)
            storage_values.append(values)
            storage_global_states.append(global_state_t)  # Store for PPO update

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
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_global_states = torch.stack(storage_global_states).reshape(-1, Config.OBS_DIM)

        # --- Update ---
        # We pass GPU tensors directly to avoid CPU copy overhead
        # Mixed Precision Update
        with torch.amp.autocast('cuda', enabled=(Config.DEVICE.type == 'cuda')):
            loss = agent.update(
                b_obs,
                b_actions,
                b_logprobs,
                b_returns,
                b_advantages,
                global_states=b_global_states,
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
            
            # ALWAYS SAVE LATEST (Safety Net)
            # Extract state dict from compiled model (if compiled, use _orig_mod)
            if is_compiled and hasattr(model, '_orig_mod'):
                model_state = model._orig_mod.state_dict()
            else:
                model_state = model.state_dict()
            
            checkpoint_data = {
                'model_state_dict': model_state,
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'update': update
            }
            # Overwrite "latest.pt" every time
            torch.save(checkpoint_data, f"checkpoints/model_latest.pt") 

            # === AOS GATE FUNCTION (Evaluation Exam) ===
            print(f"\n===== EVALUATION at Update {global_step // Config.BATCH_SIZE} =====" )
            if sp_manager.evaluate_candidate(model, make_env):
                # ACCEPTED!
                # Assuming checkpoint_dir and save_checkpoint are defined elsewhere or need to be added.
                # For this faithful edit, I'll use the existing torch.save pattern and define checkpoint_dir.
                checkpoint_dir = "checkpoints" # Assuming this is the directory
                save_path = os.path.join(checkpoint_dir, f"model_{update}.pt") # Using 'update' as in original code
                torch.save(checkpoint_data, save_path) # Using existing checkpoint_data
                print(f"✅ Candidate ACCEPTED. Saved: {save_path}")
                
                # Add to opponent pool
                sp_manager.opponent_pool.append({
                    'path': save_path,
                    'win_rate': 0.5,
                    'score': 1.0
                })
                
                # PERSIST POOL METADATA (NEW!)
                sp_manager.save_pool_metadata()

                # Render GIF (Only for accepted models) - Re-adding this from original code
                save_validation_gif(model, update)
                
                # Update Self-Play Opponent Pool - Re-adding this from original code
                sp_manager.load_checkpoints_list()
            else:
                # REJECTED
                print(f"❌ Candidate REJECTED. Continuing training...") # Corrected closing quote
                print(f"Progress saved to 'model_latest.pt' only. ---") # This line was part of the original else block
            
            # Sample new opponent for next phase (regardless of acceptance)
            sp_manager.sample_opponent(global_step)
            print(f"--- New Opponent: {sp_manager.current_opponent_name} ---")

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()