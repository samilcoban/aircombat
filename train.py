# ================================================
# FILE: train.py
# ================================================
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import glob
import re
import argparse
import imageio
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

# PyG Imports
from torch_geometric.data import Data, Batch

# Local Imports
from src.env import AirCombatEnv
from src.model import HybridActorCritic
from src.self_play import SelfPlayManager
from src.utils.scenario_plotter import ScenarioPlotter, Airplane, Missile, StatusMessage, ColorRGBA
from config import Config


# --- HARDWARE MONITOR ---
class SystemMonitor:
    def __init__(self):
        self.pynvml = None;
        self.psutil = None;
        self.handle = None
        try:
            import pynvml
            self.pynvml = pynvml;
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            pass
        try:
            import psutil; self.psutil = psutil
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


# --- CUSTOM VECTOR ENV ---
class MultiAgentVectorEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)

    def reset(self):
        obs_list, infos = [], []
        for env in self.envs:
            o, i = env.reset()
            obs_list.append(o);
            infos.append(i)
        return np.stack(obs_list), infos

    def step(self, actions):
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, t, tr, info = env.step(actions[i])
            obs_list.append(o);
            rew_list.append(r);
            term_list.append(t);
            trunc_list.append(tr);
            info_list.append(info)
            if t or tr:
                o_reset, i_reset = env.reset()
                obs_list[i] = o_reset
                info_list[i]["graph_data"] = i_reset["graph_data"]
        return np.stack(obs_list), np.stack(rew_list), np.array(term_list), np.array(trunc_list), info_list

    def call(self, method_name, *args, **kwargs):
        return [getattr(env.unwrapped, method_name)(*args, **kwargs) for env in self.envs]

    def close(self):
        for env in self.envs: env.close()


# --- MANAGERS ---
class CurriculumManager:
    def __init__(self, sp_manager):
        self.sp_manager = sp_manager;
        self.phase = 1
        self.survival_buffer = [];
        self.win_buffer = [];
        self.buffer_size = 50

    def update(self, outcomes, global_step):
        if not outcomes: return self.phase

        survived = [1.0 if r not in ["crash", "floor_violation", "shot"] else 0.0 for r in outcomes]
        won = [1.0 if r == "win" else 0.0 for r in outcomes]

        if survived: self.survival_buffer.append(np.mean(survived))
        if won: self.win_buffer.append(np.mean(won))

        if len(self.survival_buffer) > self.buffer_size: self.survival_buffer.pop(0)
        if len(self.win_buffer) > self.buffer_size: self.win_buffer.pop(0)

        avg_surv = np.mean(self.survival_buffer) if self.survival_buffer else 0.0
        avg_win = np.mean(self.win_buffer) if self.win_buffer else 0.0

        # Phase thresholds
        if self.phase == 1 and avg_win > 0.60:
            print(f"\n🚀 Phase 1 -> 2 (Completed Training Range)");
            self.phase = 2;
            self.win_buffer = []
        elif self.phase == 2 and avg_win > 0.50:
            print(f"\n🚀 Phase 2 -> 3 (Combat Ready)");
            self.phase = 3
        elif self.phase == 3 and avg_win > 0.50 and global_step > 300_000:
            print(f"\n🚀 Phase 3 -> 4 (Mastery)");
            self.phase = 4
        return self.phase


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env): super().__init__(env)

    def set_phase(self, p): self.env.unwrapped.set_phase(p)

    def set_kappa(self, k): self.env.unwrapped.set_kappa(k)

    def set_global_step(self, s): self.env.unwrapped.set_global_step(s)

    def step(self, action, **kwargs): return self.env.step(action, **kwargs)


def make_env():
    return CurriculumWrapper(AirCombatEnv())


def load_latest_checkpoint(model, optimizer):
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    files = glob.glob("checkpoints/model_*.pt")
    if not files: return 1

    numbered = [f for f in files if re.search(r'model_(\d+).pt', f)]
    if not numbered: return 1

    latest = max(numbered, key=lambda f: int(re.search(r'model_(\d+).pt', f).group(1)))
    update = int(re.search(r'model_(\d+).pt', latest).group(1))

    print(f"Loading {latest}...")
    ckpt = torch.load(latest, map_location=Config.DEVICE)
    state_dict = ckpt['model_state_dict']
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    if 'optimizer_state_dict' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return update + 1


def save_validation_gif(model, step):
    pass


# --- TRAIN ---
def train(start_phase=1):
    run_name = f"AirCombat_Hybrid_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"Log: {run_name}")

    sys_mon = SystemMonitor()
    envs = MultiAgentVectorEnv([make_env for _ in range(Config.NUM_ENVS)])
    model = HybridActorCritic().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, eps=1e-5)

    sp_manager = SelfPlayManager(phase=start_phase)
    curr_manager = CurriculumManager(sp_manager)
    curr_manager.phase = start_phase
    start_update = load_latest_checkpoint(model, optimizer)

    total_agents = Config.NUM_ENVS * Config.N_AGENTS
    gru_state = torch.zeros(1, total_agents, Config.D_MODEL).to(Config.DEVICE)
    obs_np, info = envs.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32).to(Config.DEVICE)
    done_flags = torch.zeros(total_agents).to(Config.DEVICE)

    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE
    print(f"Training: {num_updates} updates | Hybrid (Transformer Actor / GNN Critic)")

    for update in tqdm(range(start_update, num_updates + 1)):
        step_idx = update * Config.BATCH_SIZE

        b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values = [], [], [], [], [], []
        b_gru_states = []
        b_graph_lists = []

        batch_outcomes = []
        batch_stall_ratios = []
        batch_g_loads = []
        batch_fired = 0
        batch_cannons = 0
        batch_kills = 0
        batch_lock_duration = 0

        # UPDATE ENV GLOBAL STEP & PHASE
        envs.call("set_phase", curr_manager.phase)
        envs.call("set_global_step", step_idx)
        writer.add_scalar("curriculum/phase", curr_manager.phase, step_idx)

        # === ROLLOUT ===
        steps_per_update = Config.BATCH_SIZE // total_agents

        for step in range(steps_per_update):
            step_graphs = []
            env_infos = info
            for env_info in env_infos:
                if env_info:
                    if "termination_reason" in env_info and env_info["termination_reason"] != "none":
                        batch_outcomes.append(env_info["termination_reason"])
                    if "physics_stall_ratio" in env_info: batch_stall_ratios.append(env_info["physics_stall_ratio"])
                    if "physics_g" in env_info: batch_g_loads.append(env_info["physics_g"])
                    batch_fired += env_info.get("stat_missiles_fired", 0)
                    batch_cannons += env_info.get("stat_cannons_fired", 0)
                    batch_kills += env_info.get("stat_kills", 0)
                    batch_lock_duration += env_info.get("is_locking", 0)

                if env_info and "graph_data" in env_info and env_info["graph_data"] is not None:
                    gd = env_info["graph_data"]
                    step_graphs.append(Data(x=torch.tensor(gd['x']), edge_index=torch.tensor(gd['edge_index']),
                                            edge_attr=torch.tensor(gd['edge_attr'])))
                else:
                    step_graphs.append(Data(x=torch.zeros(1, 12), edge_index=torch.zeros(2, 0, dtype=torch.long),
                                            edge_attr=torch.zeros(0, 6)))

            graph_batch = Batch.from_data_list(step_graphs).to(Config.DEVICE)
            flat_obs = obs.view(total_agents, -1)

            with torch.no_grad():
                values = model.get_value(graph_batch)
                values_expanded = values.repeat_interleave(Config.N_AGENTS, dim=0)
                action, logprob, _, _, next_gru = model.get_action_and_value(flat_obs, action=None, gru_state=gru_state,
                                                                             done=done_flags)

            env_act = action.cpu().numpy().reshape(Config.NUM_ENVS, Config.N_AGENTS, -1)
            next_obs_np, rew, term, trunc, next_info = envs.step(env_act)

            rew_t = torch.tensor(rew, dtype=torch.float32).to(Config.DEVICE).view(-1)
            dones_np = np.logical_or(term, trunc)
            dones_expanded = np.repeat(dones_np[:, np.newaxis], Config.N_AGENTS, axis=1).flatten()
            done_flags = torch.tensor(dones_expanded, dtype=torch.float32).to(Config.DEVICE)

            b_obs.append(flat_obs)
            b_actions.append(action)
            b_logprobs.append(logprob)
            b_rewards.append(rew_t)
            b_dones.append(done_flags)
            b_values.append(values_expanded.view(-1))
            b_gru_states.append(gru_state.detach())
            b_graph_lists.append(step_graphs)

            obs = torch.tensor(next_obs_np, dtype=torch.float32).to(Config.DEVICE)
            gru_state = next_gru
            info = next_info

        curr_manager.update(batch_outcomes, step_idx)

        # === UPDATE ===
        t_obs = torch.stack(b_obs).view(-1, Config.OBS_DIM)
        t_actions = torch.stack(b_actions).view(-1, Config.ACTION_DIM)
        t_logprobs = torch.stack(b_logprobs).view(-1)
        t_rewards = torch.stack(b_rewards).view(-1)
        t_dones = torch.stack(b_dones).view(-1)
        t_values = torch.stack(b_values).view(-1)
        t_gru_states = torch.stack(b_gru_states).view(-1, 1, Config.D_MODEL)

        flat_graph_list = []
        for step_g_list in b_graph_lists:
            for g_data in step_g_list:
                for _ in range(Config.N_AGENTS): flat_graph_list.append(g_data)

        with torch.no_grad():
            last_graphs = []
            for inf in next_info:
                if inf and "graph_data" in inf:
                    gd = inf["graph_data"]
                    last_graphs.append(Data(x=torch.tensor(gd['x']), edge_index=torch.tensor(gd['edge_index']),
                                            edge_attr=torch.tensor(gd['edge_attr'])))
                else:
                    last_graphs.append(Data(x=torch.zeros(1, 12), edge_index=torch.zeros(2, 0, dtype=torch.long),
                                            edge_attr=torch.zeros(0, 6)))
            last_batch = Batch.from_data_list(last_graphs).to(Config.DEVICE)
            next_val = model.get_value(last_batch).repeat_interleave(Config.N_AGENTS, dim=0).view(-1)

            advantages = torch.zeros_like(t_rewards).to(Config.DEVICE)
            lastgaelam = 0

            r_rewards = t_rewards.view(steps_per_update, total_agents)
            r_dones = t_dones.view(steps_per_update, total_agents)
            r_values = t_values.view(steps_per_update, total_agents)

            for t in reversed(range(steps_per_update)):
                if t == steps_per_update - 1:
                    nextnonterminal = 1.0 - done_flags;
                    nextvalues = next_val
                else:
                    nextnonterminal = 1.0 - r_dones[t + 1];
                    nextvalues = r_values[t + 1]
                delta = r_rewards[t] + Config.GAMMA * nextvalues * nextnonterminal - r_values[t]
                advantages[t * total_agents: (
                                                         t + 1) * total_agents] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + t_values

        b_inds = np.arange(len(t_obs))

        pg_losses, v_losses, ent_losses = [], [], []

        for epoch in range(Config.UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, len(t_obs), Config.MINIBATCH_SIZE):
                end = start + Config.MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                mb_obs = t_obs[mb_inds];
                mb_actions = t_actions[mb_inds]
                mb_logprobs = t_logprobs[mb_inds];
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds];
                mb_gru = t_gru_states[mb_inds].permute(1, 0, 2)
                mb_graphs = [flat_graph_list[i] for i in mb_inds]
                mb_graph_batch = Batch.from_data_list(mb_graphs).to(Config.DEVICE)

                new_values = model.get_value(mb_graph_batch).view(-1)
                _, new_logprob, entropy, _, _ = model.get_action_and_value(mb_obs, action=mb_actions, gru_state=mb_gru)

                logratio = new_logprob - mb_logprobs;
                ratio = logratio.exp()
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - Config.CLIP_COEF, 1 + Config.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - Config.ENT_COEF * entropy_loss + Config.VF_COEF * v_loss

                optimizer.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM);
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())

        # === METRICS LOGGING ===
        hw_stats = sys_mon.get_stats()
        for k, v in hw_stats.items(): writer.add_scalar(k, v, step_idx)

        if batch_stall_ratios:
            writer.add_scalar("flight/percent_stalled", np.mean(np.array(batch_stall_ratios) > 0.5), step_idx)
            writer.add_scalar("flight/control_authority", 1.0 - np.mean(batch_stall_ratios), step_idx)
            writer.add_scalar("flight/avg_g_load", np.mean(batch_g_loads), step_idx)

        outcome_counts = Counter(batch_outcomes)
        total_finished = sum(outcome_counts.values())
        if total_finished > 0:
            writer.add_scalar("outcomes/win", outcome_counts.get("win", 0) / total_finished, step_idx)
            writer.add_scalar("outcomes/win_passive", outcome_counts.get("win_passive", 0) / total_finished, step_idx)
            writer.add_scalar("outcomes/loss_crash", (
                        outcome_counts.get("crash", 0) + outcome_counts.get("floor_violation", 0)) / total_finished,
                              step_idx)
            writer.add_scalar("outcomes/loss_shot", outcome_counts.get("shot", 0) / total_finished, step_idx)
            writer.add_scalar("outcomes/timeout", outcome_counts.get("timeout", 0) / total_finished, step_idx)
            writer.add_scalar("combat/missiles_per_episode", batch_fired / total_finished, step_idx)
            writer.add_scalar("combat/cannons_per_episode", batch_cannons / total_finished, step_idx)

        writer.add_scalar("combat/missiles_fired_total", batch_fired, step_idx)
        writer.add_scalar("combat/cannons_fired_total", batch_cannons, step_idx)
        writer.add_scalar("combat/kills_total", batch_kills, step_idx)
        writer.add_scalar("combat/lock_duration", batch_lock_duration, step_idx)

        # Calculate Hit Rate
        total_shots = batch_fired + (batch_cannons / 10.0)
        if total_shots > 0:
            writer.add_scalar("combat/hit_rate", batch_kills / total_shots, step_idx)
        else:
            writer.add_scalar("combat/hit_rate", 0.0, step_idx)

        writer.add_scalar("actions/fire_mean", t_actions[:, 3].mean().item(), step_idx)
        writer.add_scalar("actions/throttle_mean", t_actions[:, 2].mean().item(), step_idx)

        writer.add_scalar("train/loss", loss.item(), step_idx)
        writer.add_scalar("train/policy_loss", np.mean(pg_losses), step_idx)
        writer.add_scalar("train/value_loss", np.mean(v_losses), step_idx)
        writer.add_scalar("train/entropy", np.mean(ent_losses), step_idx)
        writer.add_scalar("rewards/total", torch.mean(t_rewards).item(), step_idx)

        if update % Config.SAVE_INTERVAL == 0:
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'update': update, 'phase': curr_manager.phase}, "checkpoints/model_latest.pt")
            if curr_manager.phase >= 3 and sp_manager.evaluate_candidate(model, make_env, curr_manager.phase):
                torch.save({'model_state_dict': model.state_dict()}, f"checkpoints/model_{update}.pt")
                sp_manager.opponent_pool.append({'path': f"checkpoints/model_{update}.pt", 'win_rate': 0.5})
            sp_manager.sample_opponent(step_idx)

    envs.close();
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1)
    args = parser.parse_args()
    train(start_phase=args.phase)