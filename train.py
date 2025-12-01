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
from collections import Counter

from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.ppo import PPOAgent
from src.self_play import SelfPlayManager
from src.utils.scenario_plotter import ScenarioPlotter, Airplane, Missile, StatusMessage, ColorRGBA
from config import Config


# --- HARDWARE MONITOR ---
class SystemMonitor:
    def __init__(self):
        self.pynvml = None
        self.psutil = None
        self.handle = None
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            pass
        try:
            import psutil
            self.psutil = psutil
        except:
            pass

    def get_stats(self):
        stats = {}
        if self.pynvml and self.handle:
            try:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                temp = self.pynvml.nvmlDeviceGetTemperature(self.handle, 0)
                mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                stats['hw/gpu_util'] = util.gpu
                stats['hw/gpu_mem_used_mb'] = mem.used / 1024 / 1024
                stats['hw/gpu_temp_c'] = temp
            except:
                pass
        if self.psutil:
            try:
                stats['hw/cpu_util'] = self.psutil.cpu_percent()
                stats['hw/ram_util'] = self.psutil.virtual_memory().percent
            except:
                pass
        return stats


class CurriculumManager:
    def __init__(self, sp_manager):
        self.sp_manager = sp_manager
        self.phase = 1
        self.survival_buffer = []
        self.win_buffer = []
        self.buffer_size = 50

    def update(self, infos, global_step):
        if isinstance(infos, tuple): infos = list(infos)
        survived = [1.0 if i.get("termination_reason") not in ["crash", "floor_violation", "shot"] else 0.0 for i in
                    infos if i and "termination_reason" in i]
        won = [1.0 if i.get("termination_reason") == "win" else 0.0 for i in infos if i and "termination_reason" in i]

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


def save_validation_gif(model, step):
    print("Rendering Replay...")
    env = make_env()
    plotter = ScenarioPlotter(env.unwrapped.map_limits, dpi=100, width=600, height=600)
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
                            objects.append(Airplane(e.x, e.y, e.heading, edge_color=ColorRGBA(0, 1, 1, 1),
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
                    objects.append(StatusMessage(f"Step: {i} | Phase {env.unwrapped.phase}"))
                    fname = f"{tmp_dir}/{i:04d}.png"
                    plotter.to_png(fname, objects)
                    frames.append(fname)

                obs_t = torch.tensor(obs, dtype=torch.float32).to(Config.DEVICE)
                action_t, _, _, _, lstm_state = model.get_action_and_value(obs_t, global_state=None,
                                                                           lstm_state=lstm_state)
                action = action_t.cpu().numpy()
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
    run_name = f"AirCombat_MA_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"Log: {run_name}")
    sys_mon = SystemMonitor()
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(Config.NUM_ENVS)])
    model = AgentTransformer().to(Config.DEVICE)
    agent = PPOAgent(model)
    scaler = torch.amp.GradScaler('cuda', enabled=(Config.DEVICE.type == 'cuda'))
    sp_manager = SelfPlayManager(phase=start_phase)
    curr_manager = CurriculumManager(sp_manager)
    curr_manager.phase = start_phase
    start_update = load_latest_checkpoint(model, agent.optimizer)

    next_obs, next_info = envs.reset()
    total_agents = Config.NUM_ENVS * Config.N_AGENTS
    next_obs = torch.Tensor(next_obs).to(Config.DEVICE).view(total_agents, -1)

    h0 = torch.zeros(1, total_agents, Config.D_MODEL).to(Config.DEVICE)
    c0 = torch.zeros(1, total_agents, Config.D_MODEL).to(Config.DEVICE)
    next_lstm = (h0, c0)
    next_done = torch.zeros(total_agents).to(Config.DEVICE)

    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE
    print(f"Training: {num_updates} updates | Multi-Agent Mode ({Config.N_AGENTS}v{Config.N_ENEMIES})")

    for update in tqdm(range(start_update, num_updates + 1)):
        step_idx = update * Config.BATCH_SIZE
        storage = {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': [],
                   'global_states': [], 'lstm_h': [], 'lstm_c': []}

        batch_outcomes = []
        batch_stall_ratios = []
        batch_g_loads = []
        batch_fired = 0
        batch_kills = 0

        # Note: infos here is from the LAST step of the PREVIOUS loop (or reset)
        # We need to collect stats inside the loop to catch all terminations.

        # Set Curriculum
        envs.call("set_phase", curr_manager.phase)
        writer.add_scalar("curriculum/phase", curr_manager.phase, step_idx)

        steps_per_update = Config.BATCH_SIZE // total_agents

        for step in range(steps_per_update):
            with torch.no_grad():
                raw_gs = next_info.get("global_state")
                if raw_gs is not None:
                    if isinstance(raw_gs, np.ndarray) and raw_gs.dtype == np.object_:
                        try:
                            raw_gs = np.stack(raw_gs).astype(np.float32)
                        except ValueError:
                            raw_gs = next_obs.cpu().numpy()[:Config.NUM_ENVS]
                    gs_expanded = np.repeat(raw_gs, Config.N_AGENTS, axis=0)
                    gs_t = torch.tensor(gs_expanded, dtype=torch.float32).to(Config.DEVICE)
                else:
                    gs_t = next_obs

                act, logp, _, val, new_lstm = model.get_action_and_value(next_obs, global_state=gs_t,
                                                                         lstm_state=next_lstm, done=next_done)

            env_act = act.cpu().numpy().reshape(Config.NUM_ENVS, Config.N_AGENTS, -1)
            real_obs, rew, term, trunc, next_info = envs.step(env_act)

            # --- OUTCOME TRACKING (FIXED) ---
            # Check for episodes that ended during this step
            if "final_info" in next_info:
                # AsyncVectorEnv puts info of terminated envs in 'final_info'
                for info in next_info["final_info"]:
                    if info is not None:
                        # 1. Outcomes
                        if "termination_reason" in info:
                            batch_outcomes.append(info["termination_reason"])
                        # 2. Combat Stats (accumulated)
                        batch_fired += info.get("stat_missiles_fired", 0)
                        batch_kills += info.get("stat_kills", 0)

            # --- PHYSICS TRACKING (LIVE) ---
            if "physics_stall_ratio" in next_info:
                stalls = next_info["physics_stall_ratio"]
                if isinstance(stalls, np.ndarray) and stalls.dtype == np.object_:
                    stalls = np.concatenate(stalls).flatten()
                else:
                    stalls = np.array(stalls).flatten()
                batch_stall_ratios.extend(stalls)

            if "physics_g" in next_info:
                gs = next_info["physics_g"]
                if isinstance(gs, np.ndarray) and gs.dtype == np.object_:
                    gs = np.concatenate(gs).flatten()
                else:
                    gs = np.array(gs).flatten()
                batch_g_loads.extend(gs)

            # Returns
            obs_t_pre_step = next_obs.clone()
            next_obs = torch.Tensor(real_obs).to(Config.DEVICE).view(total_agents, -1)
            rew_t = torch.tensor(rew).view(-1).to(Config.DEVICE)

            if "agent_dones" in next_info:
                raw_dones = next_info["agent_dones"]
                if isinstance(raw_dones, np.ndarray) and raw_dones.dtype == np.object_:
                    try:
                        raw_dones = np.stack(raw_dones).flatten()
                    except:
                        raw_dones = np.concatenate(raw_dones).flatten()
                else:
                    raw_dones = np.array(raw_dones).flatten()
                done_t = torch.tensor(raw_dones, dtype=torch.float32).to(Config.DEVICE)
            else:
                done_arr = np.logical_or(term, trunc)
                done_exp = np.repeat(done_arr[:, None], Config.N_AGENTS, axis=1).flatten()
                done_t = torch.tensor(done_exp, dtype=torch.float32).to(Config.DEVICE)

            storage['obs'].append(obs_t_pre_step)
            storage['actions'].append(act)
            storage['logprobs'].append(logp)
            storage['rewards'].append(rew_t)
            storage['dones'].append(next_done)
            storage['values'].append(val.flatten())
            storage['global_states'].append(gs_t)
            storage['lstm_h'].append(next_lstm[0].detach())
            storage['lstm_c'].append(next_lstm[1].detach())

            next_done = done_t
            mask = (1.0 - next_done).view(1, -1, 1)
            next_lstm = (new_lstm[0] * mask, new_lstm[1] * mask)

        def flat(x):
            return torch.stack(x).transpose(0, 1).reshape(-1, *x[0].shape[1:])

        b_obs = flat(storage['obs'])
        b_act = flat(storage['actions'])
        b_logp = flat(storage['logprobs'])
        b_don = flat(storage['dones'])
        b_gs = flat(storage['global_states'])
        b_val = flat(storage['values'])
        b_lh = torch.stack(storage['lstm_h']).permute(2, 0, 1, 3).reshape(-1, 1, Config.D_MODEL)
        b_lc = torch.stack(storage['lstm_c']).permute(2, 0, 1, 3).reshape(-1, 1, Config.D_MODEL)

        with torch.no_grad():
            last_val = model.get_value(next_obs, global_state=gs_t, lstm_state=next_lstm, done=next_done).reshape(-1)
            rew_t = torch.stack(storage['rewards'])
            don_t = torch.stack(storage['dones'])
            val_t = torch.stack(storage['values'])

            adv = torch.zeros_like(rew_t).to(Config.DEVICE)
            lastgaelam = 0
            for t in reversed(range(len(rew_t))):
                next_n = 1.0 - next_done if t == len(rew_t) - 1 else 1.0 - don_t[t + 1]
                next_v = last_val if t == len(rew_t) - 1 else val_t[t + 1]
                delta = rew_t[t] + Config.GAMMA * next_v * next_n - val_t[t]
                adv[t] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * next_n * lastgaelam

            b_adv = adv.transpose(0, 1).reshape(-1)
            b_ret = b_adv + b_val

        train_stats = agent.update(b_obs, b_act, b_logp, b_ret, b_adv, b_gs, (b_lh, b_lc), b_don, b_val, scaler)

        # Update Curriculum based on collected outcomes
        # Construct a dummy list of infos from batch_outcomes to satisfy update() signature
        dummy_infos = [{"termination_reason": r} for r in batch_outcomes]
        curr_manager.update(dummy_infos, step_idx)

        # === LOGGING ===
        hw_stats = sys_mon.get_stats()
        for k, v in hw_stats.items(): writer.add_scalar(k, v, step_idx)

        if batch_stall_ratios:
            avg_stall = np.mean(batch_stall_ratios)
            writer.add_scalar("flight/percent_stalled", np.mean(np.array(batch_stall_ratios) > 0.5), step_idx)
            writer.add_scalar("flight/control_authority", 1.0 - avg_stall, step_idx)
            writer.add_scalar("flight/avg_g_load", np.mean(batch_g_loads), step_idx)

        outcome_counts = Counter(batch_outcomes)
        total_finished = sum(outcome_counts.values())
        if total_finished > 0:
            writer.add_scalar("outcomes/win", outcome_counts.get("win", 0) / total_finished, step_idx)
            writer.add_scalar("outcomes/loss_crash", (
                        outcome_counts.get("crash", 0) + outcome_counts.get("floor_violation", 0)) / total_finished,
                              step_idx)
            writer.add_scalar("outcomes/loss_shot", outcome_counts.get("shot", 0) / total_finished, step_idx)
            writer.add_scalar("outcomes/timeout", outcome_counts.get("timeout", 0) / total_finished, step_idx)

        if batch_fired > 0:
            writer.add_scalar("combat/hit_rate", batch_kills / batch_fired, step_idx)

        writer.add_scalar("train/loss", train_stats["loss"], step_idx)
        writer.add_scalar("train/policy_loss", train_stats["policy_loss"], step_idx)
        writer.add_scalar("train/value_loss", train_stats["value_loss"], step_idx)
        writer.add_scalar("train/entropy", train_stats["entropy"], step_idx)
        writer.add_scalar("train/approx_kl", train_stats["approx_kl"], step_idx)
        writer.add_scalar("rewards/total", b_ret.mean().item(), step_idx)

        if update % Config.SAVE_INTERVAL == 0:
            ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
                    'update': update, 'phase': curr_manager.phase}
            torch.save(ckpt, "checkpoints/model_latest.pt")
            save_validation_gif(model, update)
            if curr_manager.phase >= 3:
                if sp_manager.evaluate_candidate(model, make_env, curr_manager.phase):
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