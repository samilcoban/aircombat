# ================================================
# FILE: train.py
# ================================================
import gymnasium as gym
import numpy as np
import torch
import os
import time
import glob
import re
import argparse
import math
import imageio
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.ppo import PPOAgent
from src.self_play import SelfPlayManager
from src.utils.logger import FlightRecorder
from src.utils.scenario_plotter import ScenarioPlotter, Airplane, Missile, StatusMessage, ColorRGBA
from config import Config


class CurriculumManager:
    def __init__(self, sp_manager):
        self.sp_manager = sp_manager
        self.phase = 1
        self.survival_buffer = []
        self.win_buffer = []
        self.buffer_size = 50

    def update(self, infos, global_step):
        survived = [1.0 if i.get("termination_reason") not in ["crash", "floor_violation"] else 0.0 for i in infos if
                    "termination_reason" in i]
        won = [1.0 if i.get("termination_reason") == "win" else 0.0 for i in infos if "termination_reason" in i]
        if survived: self.survival_buffer.append(np.mean(survived))
        if won: self.win_buffer.append(np.mean(won))
        if len(self.survival_buffer) > self.buffer_size: self.survival_buffer.pop(0)
        if len(self.win_buffer) > self.buffer_size: self.win_buffer.pop(0)

        avg_surv = np.mean(self.survival_buffer) if self.survival_buffer else 0.0
        avg_win = np.mean(self.win_buffer) if self.win_buffer else 0.0

        if self.phase == 1 and avg_surv > 0.90 and global_step > 200_000:
            print(f"\n🚀 Phase 1 -> 2 (Survival: {avg_surv:.2f})")
            self.phase = 2;
            self.win_buffer = []
        elif self.phase == 2 and avg_win > 0.30 and global_step > 500_000:
            print(f"\n🚀 Phase 2 -> 3 (Win Rate: {avg_win:.2f})")
            self.phase = 3
        elif self.phase == 3 and avg_win > 0.60 and global_step > 1_000_000:
            print(f"\n🚀 Phase 3 -> 4 (Win Rate: {avg_win:.2f})")
            self.phase = 4
        return self.phase, avg_surv, avg_win


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env): super().__init__(env)

    def set_phase(self, p): self.env.unwrapped.set_phase(p)

    def set_kappa(self, k): self.env.unwrapped.set_kappa(k)


def make_env():
    env = AirCombatEnv()
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
    env = CurriculumWrapper(env)
    return env


def load_latest_checkpoint(model, optimizer):
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    files = glob.glob("checkpoints/model_*.pt")
    numbered = [f for f in files if re.search(r'model_(\d+).pt', f)]
    if numbered:
        latest = max(numbered, key=lambda f: int(re.search(r'model_(\d+).pt', f).group(1)))
        update = int(re.search(r'model_(\d+).pt', latest).group(1))
    elif os.path.exists("checkpoints/model_latest.pt"):
        latest = "checkpoints/model_latest.pt";
        update = 0
    else:
        return 1
    print(f"Loading {latest}...")
    ckpt = torch.load(latest, map_location=Config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    if 'optimizer_state_dict' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return update + 1


# === PLOTTER VISUALIZATION ===
def save_validation_gif(model, step):
    print("Rendering Replay...")
    env = make_env()
    # Safe extraction of limits before loop
    render_limits = env.unwrapped.render_limits
    plotter = ScenarioPlotter(render_limits)

    obs, _ = env.reset()
    frames = []
    lstm_state = None
    done = False
    model.eval()
    tmp_dir = f"temp_frames_{step}"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with torch.no_grad():
            for i in range(1000):
                if done: break
                if i % 3 == 0:
                    objects = []
                    core = env.unwrapped.core
                    for uid in env.unwrapped.blue_ids:
                        if uid in core.entities:
                            e = core.entities[uid]
                            objects.append(Airplane(e.x, e.y, e.heading, edge_color=ColorRGBA(0, 0, 1, 1),
                                                    fill_color=ColorRGBA(0, 0, 0.5, 0.5), info_text=f"B{uid}"))
                    for uid in env.unwrapped.red_ids:
                        if uid in core.entities:
                            e = core.entities[uid]
                            objects.append(Airplane(e.x, e.y, e.heading, edge_color=ColorRGBA(1, 0, 0, 1),
                                                    fill_color=ColorRGBA(0.5, 0, 0, 0.5), info_text=f"R{uid}"))
                    for e in core.entities.values():
                        if e.type == "missile":
                            objects.append(Missile(e.x, e.y, e.heading, edge_color=ColorRGBA(1, 1, 0, 1),
                                                   fill_color=ColorRGBA(1, 1, 0, 1)))
                    objects.append(StatusMessage(f"Step: {i} | Time: {core.time:.1f}s | Phase {env.unwrapped.phase}"))
                    fname = f"{tmp_dir}/{i:04d}.png"
                    plotter.to_png(fname, objects)
                    frames.append(fname)
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                action_t, _, _, _, lstm_state = model.get_action_and_value(obs_t, lstm_state=lstm_state)
                action = action_t.cpu().numpy().flatten()
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
        if frames:
            gif_path = f"checkpoints/val_{step}.gif"
            images = [imageio.imread(f) for f in frames]
            imageio.mimsave(gif_path, images, fps=20)
            print(f"Saved {gif_path}")
    except Exception as e:
        print(f"Render Error: {e}")
    finally:
        env.close()
        for f in glob.glob(f"{tmp_dir}/*.png"): os.remove(f)
        try:
            os.rmdir(tmp_dir)
        except:
            pass
        model.train()


def train(start_phase=1):
    run_name = f"AirCombat_Curriculum_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"Log: {run_name}")

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(Config.NUM_ENVS)])
    model = AgentTransformer().to(Config.DEVICE)
    agent = PPOAgent(model)
    scaler = torch.amp.GradScaler('cuda', enabled=(Config.DEVICE.type == 'cuda'))
    sp_manager = SelfPlayManager(phase=start_phase)
    curr_manager = CurriculumManager(sp_manager)
    curr_manager.phase = start_phase

    start_update = load_latest_checkpoint(model, agent.optimizer)
    next_obs, next_info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(Config.DEVICE)
    next_done = torch.zeros(Config.NUM_ENVS).to(Config.DEVICE)
    h0 = torch.zeros(1, Config.NUM_ENVS, Config.D_MODEL).to(Config.DEVICE)
    c0 = torch.zeros(1, Config.NUM_ENVS, Config.D_MODEL).to(Config.DEVICE)
    next_lstm = (h0, c0)

    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE
    print(f"Training: {num_updates} updates")

    for update in tqdm(range(start_update, num_updates + 1)):
        step_idx = update * Config.BATCH_SIZE
        storage = {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': [],
                   'global_states': [], 'lstm_h': [], 'lstm_c': []}

        if "final_info" in next_info:
            infos = [i for i in next_info["final_info"] if i is not None]
            current_phase, surv, win = curr_manager.update(infos, step_idx)
        else:
            current_phase, surv, win = curr_manager.phase, 0.0, 0.0

        envs.call("set_phase", current_phase)
        writer.add_scalar("curriculum/phase", current_phase, step_idx)
        writer.add_scalar("curriculum/survival", surv, step_idx)
        writer.add_scalar("curriculum/win", win, step_idx)

        for step in range(Config.BATCH_SIZE // Config.NUM_ENVS):
            with torch.no_grad():
                gs = next_info.get("global_state", next_obs.cpu().numpy())
                if isinstance(gs, np.ndarray) and gs.dtype == np.object_: gs = np.stack(gs).astype(np.float32)
                gs_t = torch.tensor(gs, dtype=torch.float32).to(Config.DEVICE)

                act, logp, _, val, new_lstm = model.get_action_and_value(next_obs, global_state=gs_t,
                                                                         lstm_state=next_lstm, done=next_done)

            red_act = None
            if "red_obs" in next_info: red_act = sp_manager.get_action(next_info["red_obs"])

            if red_act is None:
                real_obs, rew, term, trunc, next_info = envs.step(act.cpu().numpy())
            else:
                blue = act.cpu().numpy()
                if red_act.shape[0] != blue.shape[0]: red_act = np.zeros_like(blue)
                real_obs, rew, term, trunc, next_info = envs.step(np.concatenate([blue, red_act], axis=1))

            done = np.logical_or(term, trunc)
            storage['obs'].append(next_obs);
            storage['actions'].append(act);
            storage['logprobs'].append(logp)
            storage['rewards'].append(torch.tensor(rew).to(Config.DEVICE));
            storage['dones'].append(next_done)
            storage['values'].append(val.flatten());
            storage['global_states'].append(gs_t)
            storage['lstm_h'].append(next_lstm[0].detach());
            storage['lstm_c'].append(next_lstm[1].detach())
            next_obs = torch.Tensor(real_obs).to(Config.DEVICE);
            next_done = torch.Tensor(done).to(Config.DEVICE)
            mask = (1.0 - next_done).view(1, -1, 1)
            next_lstm = (new_lstm[0] * mask, new_lstm[1] * mask)

        # Flatten Helper for Lists
        def flat(x):
            return torch.stack(x).transpose(0, 1).reshape(-1, *x[0].shape[1:])

        # Flatten Lists
        b_obs = flat(storage['obs']);
        b_act = flat(storage['actions']);
        b_logp = flat(storage['logprobs'])
        b_don = flat(storage['dones']);
        b_gs = flat(storage['global_states']);
        b_val = flat(storage['values'])

        # Flatten LSTM States (Already stacked tensors, so manual reshape)
        # Stack: (Time, Layers, Envs, Hidden) -> Permute: (Envs, Time, Layers, Hidden) -> Reshape: (Envs*Time, Layers, Hidden)
        b_lh = torch.stack(storage['lstm_h']).permute(2, 0, 1, 3).reshape(-1, 1, Config.D_MODEL)
        b_lc = torch.stack(storage['lstm_c']).permute(2, 0, 1, 3).reshape(-1, 1, Config.D_MODEL)

        # Explicit GAE Loop
        with torch.no_grad():
            last_val = model.get_value(next_obs, global_state=gs_t, lstm_state=next_lstm, done=next_done).reshape(-1)
            rew_t = torch.stack(storage['rewards']);
            don_t = torch.stack(storage['dones']);
            val_t = torch.stack(storage['values'])

            adv = torch.zeros_like(rew_t).to(Config.DEVICE)
            lastgaelam = 0
            for t in reversed(range(len(rew_t))):
                next_n = 1.0 - next_done if t == len(rew_t) - 1 else 1.0 - don_t[t + 1]
                next_v = last_val if t == len(rew_t) - 1 else val_t[t + 1]
                delta = rew_t[t] + Config.GAMMA * next_v * next_n - val_t[t]
                adv[t] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * next_n * lastgaelam

            # FIX: Flatten 'adv' Tensor directly (do not use flat() which expects list)
            b_adv = adv.transpose(0, 1).reshape(-1)
            b_ret = b_adv + b_val

        loss = agent.update(b_obs, b_act, b_logp, b_ret, b_adv, b_gs, (b_lh, b_lc), b_don, b_val, scaler)

        # === DETAILED TENSORBOARD LOGGING ===
        # 1. Action Breakdown
        writer.add_scalar("actions_mean/0_roll", b_act[:, 0].mean().item(), step_idx)
        writer.add_scalar("actions_mean/1_g_pull", b_act[:, 1].mean().item(), step_idx)
        writer.add_scalar("actions_mean/2_throttle", b_act[:, 2].mean().item(), step_idx)
        writer.add_scalar("actions_mean/3_fire_prob", (b_act[:, 3] > 0).float().mean().item(), step_idx)
        writer.add_scalar("actions_std/roll_std", b_act[:, 0].std().item(), step_idx)

        # 2. Flight Physics
        if "physics_speed" in next_info:
            s = next_info["physics_speed"]
            if isinstance(s, (list, np.ndarray)):
                writer.add_scalar("flight/speed_knots", np.mean(s), step_idx)
                writer.add_scalar("flight/altitude_m", np.mean(next_info["physics_alt"]), step_idx)
                v_ms = np.mean(s) * 0.5144
                h = np.mean(next_info["physics_alt"])
                e_state = (h + (v_ms ** 2) / (2 * 9.81)) / 1000.0
                writer.add_scalar("flight/specific_energy", e_state, step_idx)

        # 3. Reward Breakdown
        if "rew_existence" in next_info:
            writer.add_scalar("rewards/1_existence", np.mean(next_info["rew_existence"]), step_idx)
            writer.add_scalar("rewards/2_instructor", np.mean(next_info["rew_instructor"]), step_idx)
            writer.add_scalar("rewards/3_penalty", np.mean(next_info["rew_penalty"]), step_idx)
            writer.add_scalar("rewards/4_guidance", np.mean(next_info["rew_guidance"]), step_idx)
            writer.add_scalar("rewards/5_combat", np.mean(next_info["rew_combat"]), step_idx)
            writer.add_scalar("rewards/total", b_ret.mean().item(), step_idx)

        # 4. Training Health
        writer.add_scalar("train/loss", loss, step_idx)
        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        exp_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        writer.add_scalar("train/explained_variance", exp_var, step_idx)

        # 5. Tactics & ATA
        if b_obs.shape[1] >= 40:
            ata = b_obs[:, 39]
            writer.add_scalar("tactics/mean_ata_deg", ata.abs().mean().item() * 180.0, step_idx)

        # Checkpoint
        if update % Config.SAVE_INTERVAL == 0:
            ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
                    'update': update, 'phase': current_phase}
            torch.save(ckpt, "checkpoints/model_latest.pt")
            save_validation_gif(model, update)
            if current_phase >= 3:
                if sp_manager.evaluate_candidate(model, make_env, current_phase):
                    torch.save(ckpt, f"checkpoints/model_{update}.pt")
                    sp_manager.opponent_pool.append({'path': f"checkpoints/model_{update}.pt", 'win_rate': 0.5})
                sp_manager.sample_opponent(step_idx)

    envs.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1)
    args = parser.parse_args()
    print("=== Training Start ===")
    train(start_phase=args.phase)