# train.py
import gymnasium as gym
import numpy as np
import torch
import os
import time
import glob
import re
import argparse
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.ppo import PPOAgent
from src.self_play import SelfPlayManager
from src.utils.logger import FlightRecorder
from config import Config


# === MENTOR NOTE: Helper for PPO Diagnostics ===
def compute_explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that y_pred explains about y_true.
    Interpretation:
        ev > 0.9: Great predictor
        ev < 0.0: The critic is worse than predicting the mean (Random guessing)
    """
    var_y = torch.var(y_true)
    if var_y == 0:
        return np.nan
    return 1 - torch.var(y_true - y_pred) / var_y

class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def set_kappa(self, k):
        self.env.unwrapped.set_kappa(k)
        
    def set_phase(self, phase_id, progress=0.0):
        self.env.unwrapped.set_phase(phase_id, progress)

def make_env():
    env = AirCombatEnv()
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # MENTOR NOTE: NormalizeReward is vital for PPO, but can mask scale issues.
    # If rewards look tiny in logs, remember they are normalized here.
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
    env = CurriculumWrapper(env)
    return env

def save_validation_gif(model, step):
    """Generates a 2D visual replay of the agent's behavior using matplotlib."""
    print("Generating Validation GIF...")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation, PillowWriter
    from src.utils.logger import FlightRecorder
    
    # Validation uses a FRESH environment (no normalization wrappers for rendering clarity)
    env = make_env() 
    
    obs, _ = env.reset()
    frames_data = []
    step_count = 0

    done = False
    model.eval()
    
    # Flight recorder for validation
    if not hasattr(save_validation_gif, "recorder"):
        save_validation_gif.recorder = FlightRecorder(log_dir="logs/validation")
    
    try:
        with torch.no_grad():
            while not done and step_count < 1000:
                # Capture frame data
                if step_count % 5 == 0:
                    frames_data.append({
                        'entities': {uid: ent for uid, ent in env.unwrapped.core.entities.items()},
                        'time': env.unwrapped.core.time,
                        'step': step_count,
                        'phase': env.unwrapped.phase
                    })

                # Inference
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                action, _, _, _, _ = model.get_action_and_value(obs_t)
                
                obs, _, term, trunc, _ = env.step(action.cpu().numpy().flatten())
                done = term or trunc
                step_count += 1
                
                # Flight Recorder Logging
                blue_ent = env.unwrapped.core.entities.get(env.unwrapped.blue_ids[0]) if env.unwrapped.blue_ids else None
                red_ent = env.unwrapped.core.entities.get(env.unwrapped.red_ids[0]) if env.unwrapped.red_ids else None
                
                is_locked = False
                if blue_ent and red_ent:
                    _, is_locked = env.unwrapped.core.get_sensor_state(blue_ent.uid, red_ent.uid)
                
                missile_active = any(e.type == "missile" for e in env.unwrapped.core.entities.values())
                
                save_validation_gif.recorder.log_step(
                    episode=step,
                    step=step_count,
                    time_sec=env.unwrapped.core.time,
                    blue_ent=blue_ent,
                    red_ent=red_ent,
                    action=action.cpu().numpy().flatten(),
                    reward=0.0,
                    is_locked=is_locked,
                    missile_active=missile_active
                )
    except Exception as e:
        print(f"Error during GIF generation: {e}")
    finally:
        save_validation_gif.recorder.save_episode(step)
        env.close()

    # Create 2D matplotlib animation
    if frames_data:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(i):
            ax.clear()
            frame = frames_data[i]
            entities = frame['entities']
            
            # Get render limits
            limits = env.unwrapped.render_limits
            ax.set_xlim(limits.min_x / 1000, limits.max_x / 1000)  # Convert to km
            ax.set_ylim(limits.min_y / 1000, limits.max_y / 1000)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('East (km)')
            ax.set_ylabel('North (km)')
            ax.set_title(f"Step {frame['step']}, Time {frame['time']:.1f}s, Phase {frame['phase']}")
            
            # Draw entities
            for uid, ent in entities.items():
                x_km = ent.y / 1000  # East
                y_km = ent.x / 1000  # North
                
                if ent.type == "plane":
                    color = 'blue' if ent.team == "blue" else 'red'
                    marker = 'o'
                    size = 100
                    
                    # Draw plane
                    ax.scatter(x_km, y_km, c=color, marker=marker, s=size, edgecolors='white', linewidths=2, zorder=3)
                    
                    # Draw heading vector
                    heading_rad = math.radians(ent.heading)
                    dx = math.sin(heading_rad) * 2  # 2km vector
                    dy = math.cos(heading_rad) * 2
                    ax.arrow(x_km, y_km, dx, dy, head_width=0.5, head_length=0.3, fc=color, ec=color, alpha=0.7, zorder=2)
                    
                    # Label
                    ax.text(x_km, y_km + 1, f"{ent.team[0].upper()}{uid}", ha='center', fontsize=8, color=color, weight='bold')
                    
                elif ent.type == "missile":
                    ax.scatter(x_km, y_km, c='yellow', marker='^', s=50, edgecolors='orange', linewidths=1, zorder=3)
            
            return ax,
        
        anim = FuncAnimation(fig, animate, frames=len(frames_data), interval=100, blit=False)
        
        gif_path = f"checkpoints/val_{step}.gif"
        writer = PillowWriter(fps=10)
        anim.save(gif_path, writer=writer)
        plt.close(fig)
        print(f"Saved 2D GIF: {gif_path}")
    
    model.train()


def load_latest_checkpoint(model, optimizer):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        return 1

    files = glob.glob("checkpoints/model_*.pt")
    numbered_files = [f for f in files if re.search(r'model_(\d+).pt', f)]
    
    latest_file = None
    update_num = 0
    
    # First priority: Load from numbered checkpoints (evaluation-passed models)
    if numbered_files:
        latest_file = max(numbered_files, key=lambda f: int(re.search(r'model_(\d+).pt', f).group(1)))
        update_num = int(re.search(r'model_(\d+).pt', latest_file).group(1))
        print(f"--- Resuming from numbered checkpoint: {latest_file} (Update {update_num}) ---")
    
    # Second priority: Fall back to model_latest.pt if no numbered checkpoints exist
    elif os.path.exists("checkpoints/model_latest.pt"):
        latest_file = "checkpoints/model_latest.pt"
        print(f"--- Resuming from latest checkpoint: {latest_file} ---")
    
    # No checkpoints found at all
    else:
        print("--- No checkpoints found. Starting fresh. ---")
        return 1

    # Load the checkpoint
    checkpoint = torch.load(latest_file, map_location=Config.DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        
        # Detect if current model is compiled
        is_compiled_model = hasattr(model, '_orig_mod')
        
        # Detect if checkpoint is from compiled model
        has_orig_mod_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())
        
        # Adjust state dict keys based on model type
        if is_compiled_model and not has_orig_mod_prefix:
            # Loading uncompiled checkpoint into compiled model - add prefix
            new_state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
        elif not is_compiled_model and has_orig_mod_prefix:
            # Loading compiled checkpoint into uncompiled model - remove prefix
            new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        else:
            # No conversion needed
            new_state_dict = state_dict
        
        model.load_state_dict(new_state_dict)
        
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Extract update number from checkpoint metadata if available
        if "update" in checkpoint:
            update_num = checkpoint["update"]
            print(f"--- Continuing from update {update_num} ---")
    else:
        model.load_state_dict(checkpoint)

    return update_num + 1

def train(phase=2):
    run_name = f"AirCombat_Marmara_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"Logging to runs/{run_name}")

    # 1. Init Envs
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(Config.NUM_ENVS)])

    # 2. Model Setup
    model = AgentTransformer().to(Config.DEVICE)

    is_compiled = False
    try:
        model = torch.compile(model)
        is_compiled = True
        print("✅ Model compiled with torch.compile()")
    except Exception as e:
        print(f"⚠️  torch.compile() failed: {e}. Running in eager mode.")

    agent = PPOAgent(model)
    # Use torch.amp for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(Config.DEVICE.type == 'cuda'))
    
    sp_manager = SelfPlayManager(phase=phase)
    sp_manager.sample_opponent(0)
    sp_manager.sample_opponent(0)
    print(f"Initial Opponent: {sp_manager.current_opponent_name}")

    # Flight Recorder
    flight_recorder = FlightRecorder()

    start_update = load_latest_checkpoint(model, agent.optimizer)
    
    # 3. Rollout Storage Setup
    next_obs, next_info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(Config.DEVICE)
    next_done = torch.zeros(Config.NUM_ENVS).to(Config.DEVICE)
    
    # LSTM State Init
    next_lstm_state = (
        torch.zeros(1, Config.NUM_ENVS, Config.D_MODEL).to(Config.DEVICE),
        torch.zeros(1, Config.NUM_ENVS, Config.D_MODEL).to(Config.DEVICE)
    )

    global_step = (start_update - 1) * Config.BATCH_SIZE
    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE

    print(f"Starting training: {num_updates} updates, Batch Size {Config.BATCH_SIZE}")

    for update in tqdm(range(start_update, num_updates + 1)):
        # Storage
        storage = {
            'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': [],
            'global_states': [], 'lstm_h': [], 'lstm_c': []
        }

        # --- Curriculum Logic ---
        current_kappa = sp_manager.kappa if hasattr(sp_manager, 'kappa') else 0.0
        current_phase = sp_manager.get_current_phase(global_step)
        
        # MENTOR NOTE: Smoothing phase progress avoids abrupt difficulty jumps
        phase_progress = 0.0
        if current_phase == 1: phase_progress = global_step / 1_000_000.0
        elif current_phase == 2: phase_progress = (global_step - 1_000_000.0) / 1_000_000.0
        elif current_phase == 3: phase_progress = (global_step - 2_000_000.0) / 2_000_000.0
        
        phase_progress = min(max(phase_progress, 0.0), 1.0)
        
        envs.call("set_kappa", current_kappa)
        envs.call("set_phase", current_phase, progress=phase_progress)

        # Tracking metrics
        ep_rewards = []
        ep_lengths = []
        term_reasons = []

        # --- Collection Loop ---
        steps_per_env = Config.BATCH_SIZE // Config.NUM_ENVS
        
        for step in range(steps_per_env):
            global_step += Config.NUM_ENVS

            with torch.no_grad():
                # Extract global state for CTDE (Critic)
                # Handle cases where info might be missing on first step or reset
                global_state_np = next_info.get("global_state", next_obs.cpu().numpy())
                
                # Handling generic object arrays from VectorEnv
                if isinstance(global_state_np, np.ndarray) and global_state_np.dtype == np.object_:
                    global_state_np = np.array(global_state_np.tolist(), dtype=np.float32)
                
                global_state_t = torch.tensor(global_state_np, dtype=torch.float32).to(Config.DEVICE)

                # Get Action
                action, logprob, _, value, next_lstm_state = model.get_action_and_value(
                    next_obs, 
                    global_state=global_state_t,
                    lstm_state=next_lstm_state,
                    done=next_done
                )

            # Self-Play: Combine Blue Action with Red Action
            red_actions = None
            if "red_obs" in next_info:
                red_obs = next_info["red_obs"]
                red_actions = sp_manager.get_action(red_obs)
            
            # Step Environment
            if red_actions is None:
                real_next_obs, reward, term, trunc, next_info = envs.step(action.cpu().numpy())
            else:
                blue_act = action.cpu().numpy()
                # Ensure dimensions match for concatenation
                if red_actions.shape[0] != blue_act.shape[0]:
                    red_actions = np.zeros_like(blue_act)
                concat_actions = np.concatenate([blue_act, red_actions], axis=1)
                real_next_obs, reward, term, trunc, next_info = envs.step(concat_actions)

            done = np.logical_or(term, trunc)

            # --- Flight Recorder Logging ---
            # Log data for the first environment only to avoid massive CSVs
            if envs.num_envs > 0:
                # Get raw entities from the first env's core
                # Note: AsyncVectorEnv makes accessing internal attributes tricky.
                pass

            # Store Episode Stats
            for idx, d in enumerate(done):
                if d and "final_info" in next_info:
                    info_item = next_info["final_info"][idx]
                    if info_item and "episode" in info_item:
                        ep_rewards.append(info_item["episode"]["r"])
                        ep_lengths.append(info_item["episode"]["l"])
                    if info_item and "termination_reason" in info_item:
                        term_reasons.append(info_item["termination_reason"])

            # Store Rollout Data
            storage['obs'].append(next_obs)
            storage['actions'].append(action)
            storage['logprobs'].append(logprob)
            storage['rewards'].append(torch.tensor(reward).to(Config.DEVICE))
            storage['dones'].append(next_done)
            storage['values'].append(value.flatten())
            storage['global_states'].append(global_state_t)
            storage['lstm_h'].append(next_lstm_state[0].squeeze(0))
            storage['lstm_c'].append(next_lstm_state[1].squeeze(0))

            next_obs = torch.Tensor(real_next_obs).to(Config.DEVICE)
            next_done = torch.Tensor(done).to(Config.DEVICE)

        # --- GAE (Generalized Advantage Estimation) ---
        with torch.no_grad():
            next_value = model.get_value(
                next_obs,
                global_state=global_state_t, # Use latest global state
                lstm_state=next_lstm_state,
                done=next_done
            ).reshape(-1)
            
            # Convert lists to tensors
            storage_rewards = torch.stack(storage['rewards'])
            storage_dones = torch.stack(storage['dones'])
            storage_values = torch.stack(storage['values'])

            advantages = torch.zeros_like(storage_rewards).to(Config.DEVICE)
            lastgaelam = 0
            
            # Reverse iteration for GAE
            for t in reversed(range(steps_per_env)):
                if t == steps_per_env - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - storage_dones[t + 1]
                    nextvalues = storage_values[t + 1]
                
                delta = storage_rewards[t] + Config.GAMMA * nextvalues * nextnonterminal - storage_values[t]
                advantages[t] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * nextnonterminal * lastgaelam
            
            returns = advantages + storage_values

        # --- Flattening for PPO Update ---
        def flatten_seq(x_list):
            x = torch.stack(x_list)
            # (Steps, Envs, ...) -> (Envs, Steps, ...)
            x = x.transpose(0, 1)
            return x.reshape(-1, *x.shape[2:])

        b_obs = flatten_seq(storage['obs'])
        b_actions = flatten_seq(storage['actions'])
        b_logprobs = flatten_seq(storage['logprobs'])
        b_returns = flatten_seq([r for r in returns])
        b_advantages = flatten_seq([a for a in advantages])
        b_dones = flatten_seq([d for d in storage_dones])
        b_global_states = flatten_seq(storage['global_states'])
        
        b_lstm_h = flatten_seq(storage['lstm_h'])
        b_lstm_c = flatten_seq(storage['lstm_c'])
        b_lstm_states = (b_lstm_h, b_lstm_c)
        
        # Log Explained Variance (Diagnostic)
        b_values = flatten_seq([v for v in storage_values])
        explained_var = compute_explained_variance(b_values, b_returns)
        writer.add_scalar("charts/explained_variance", explained_var.item(), global_step)

        # --- PPO Update ---
        with torch.amp.autocast('cuda', enabled=(Config.DEVICE.type == 'cuda')):
            loss = agent.update(
                b_obs, b_actions, b_logprobs, b_returns, b_advantages,
                global_states=b_global_states,
                lstm_states=b_lstm_states,
                dones=b_dones,
                old_values=b_values,  # For Value Function Clipping (CleanMARL/MAPPO)
                scaler=scaler
            )

        # --- Logging ---
        writer.add_scalar("charts/loss", loss, global_step)
        writer.add_scalar("charts/mean_step_reward_norm", storage_rewards.mean().item(), global_step)
        writer.add_scalar("curriculum/phase", current_phase, global_step)
        writer.add_scalar("curriculum/kappa", current_kappa, global_step)
        if "physics_alt" in next_info:
            # VectorEnv returns a batch of values (one per environment)
            # We take the mean to get the average fleet status
            writer.add_scalar("physics/true_altitude_m", np.mean(next_info["physics_alt"]), global_step)
            writer.add_scalar("physics/true_speed_kts", np.mean(next_info["physics_speed"]), global_step)
            writer.add_scalar("physics/fuel_remaining", np.mean(next_info["physics_fuel"]), global_step)
        # ==============================

        if ep_rewards:
            writer.add_scalar("episode/raw_return_mean", np.mean(ep_rewards), global_step)
            writer.add_scalar("episode/length_mean", np.mean(ep_lengths), global_step)
            
            # Log termination reasons
            from collections import Counter
            counts = Counter(term_reasons)
            for reason, count in counts.items():
                writer.add_scalar(f"termination/{reason}", count, global_step)

        # Monitor Action Behavior (Detect Seizure Pilot)
        actions_np = b_actions.cpu().numpy()
        writer.add_scalar("actions/roll_std", actions_np[:, 0].std(), global_step)
        writer.add_scalar("actions/throttle_mean", actions_np[:, 2].mean(), global_step)

        # --- Tactical Telemetry ---
        # Denormalize observations to get physical values
        # Obs mapping: 
        # 4: Speed (normalized by /1000)
        # 14: Altitude (normalized by /10000)
        # 17: ATA (normalized by /180)
        
        # We need to reshape b_obs back to (Envs * Steps, Obs_Dim)
        # b_obs is (Envs * Steps, Obs_Dim)
        
        # Extract features (assuming single agent for now or taking mean across all)
        # We only care about the ego agent's features which are at the start of the obs vector
        # The obs vector is flattened [Ego, Entity1, Entity2...]
        # Ego features are at indices 0-21 (FEAT_DIM=22)
        
        # Speed: Index 4
        speed_mean = b_obs[:, 4].mean().item() * 1000.0
        writer.add_scalar("physics/speed_mean", speed_mean, global_step)
        
        # Altitude: Index 14
        alt_mean = b_obs[:, 14].mean().item() * 10000.0
        writer.add_scalar("physics/altitude_mean", alt_mean, global_step)
        
        # ATA: Index 17 (This is ATA to the nearest enemy if we look at the first entity after ego?)
        # Wait, _vectorize puts Ego first. Then others.
        # Ego vector doesn't have ATA to itself (it's 0).
        # We need ATA from Ego to Enemy. This is in the Enemy's feature block relative to Ego.
        # Enemy feature block starts at FEAT_DIM (22).
        # Enemy ATA is at index 22 + 17 = 39.
        # Let's check if there is an enemy.
        if b_obs.shape[1] >= 44: # At least 2 entities (22 * 2)
            ata_mean = b_obs[:, 22 + 17].abs().mean().item() * 180.0
            writer.add_scalar("tactics/mean_ata_deg", ata_mean, global_step)

        # Episode Length (Survival)
        if ep_lengths:
             writer.add_scalar("performance/episode_length", np.mean(ep_lengths), global_step)

        # --- Checkpointing & Evaluation ---
        if update % Config.SAVE_INTERVAL == 0:
            # 1. Save Latest (Safety Net)
            if is_compiled and hasattr(model, '_orig_mod'):
                model_state = model._orig_mod.state_dict()
            else:
                model_state = model.state_dict()
            
            checkpoint_data = {
                'model_state_dict': model_state,
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'update': update
            }
            torch.save(checkpoint_data, f"checkpoints/model_latest.pt")
            
            # 2. Gate Function (AOS)
            # Evaluate the *current* model. If it passes, it gets saved as model_X.pt and added to pool.
            print(f"\n===== EVALUATION at Update {update} =====")
            accepted = sp_manager.evaluate_candidate(model, make_env, global_step=global_step)
            
            if accepted:
                save_path = f"checkpoints/model_{update}.pt"
                torch.save(checkpoint_data, save_path)
                print(f"✅ PASSED. Saved to {save_path}")
                
                sp_manager.opponent_pool.append({
                    'path': save_path,
                    'win_rate': 0.5,
                    'score': 1.0
                })
                sp_manager.save_pool_metadata()
            else:
                print("❌ FAILED. Continuing training...")
            
            # Generate GIF every checkpoint interval (regardless of acceptance)
            save_validation_gif(model, update)

            # 3. New Opponent Sampling
            sp_manager.load_checkpoints_list()
            sp_manager.sample_opponent(global_step)
            
            model.train() # Ensure we return to train mode

    envs.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Combat RL Training")
    parser.add_argument('--phase', type=int, default=2, choices=[1, 2, 3, 4, 5],
                        help='Training phase: 1=Stationary, 2=Straight, 3=Random, 4=Ace, 5=Self-Play')
    args = parser.parse_args()
    
    print(f"=== Starting Training: Phase {args.phase} ===")
    train(phase=args.phase)