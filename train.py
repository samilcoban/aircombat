# ================================================
# FILE: train.py (Enhanced Logging)
# ================================================
import sys
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
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import random

from torch_geometric.data import Data, Batch
from src.env import AirCombatEnv
from src.model import HybridActorCritic
from src.ppo import PPOAgent
from src.self_play import SelfPlayManager
from config import Config


class SystemMonitor:
    def __init__(self):
        self.pynvml = None
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            pass

    def get_stats(self):
        stats = {}
        if self.pynvml and getattr(self, 'handle', None):
            try:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                stats['hw/gpu_util'] = util.gpu
                stats['hw/gpu_mem_used_mb'] = mem.used / 1024 / 1024
            except:
                pass
        return stats


def worker(remote, parent_remote, env_fn_wrapper, seed):
    import random
    import numpy as np
    import torch
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    parent_remote.close()
    env = env_fn_wrapper()
    env.reset(seed=seed)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                blue_act, red_act = data
                ob, reward, term, trunc, info = env.step(blue_act, red_actions=red_act)
                if term or trunc:
                    ob_reset, info_reset = env.reset()
                    info['red_obs'] = info_reset.get('red_obs')
                    info['graph_data'] = info_reset.get('graph_data')
                    ob = ob_reset
                remote.send((ob, reward, term, trunc, info))
            elif cmd == 'reset':
                ob, info = env.reset()
                remote.send((ob, info))
            elif cmd == 'call':
                remote.send(getattr(env.unwrapped, data[0])(*data[1], **data[2]))
            elif cmd == 'close':
                env.close();
                remote.close();
                break
        except EOFError:
            break


class ParallelMultiAgentEnv:
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        base_seed = int(time.time())
        self.ps = []
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            p = mp.Process(target=worker, args=(work_remote, remote, env_fn, base_seed + i))
            self.ps.append(p);
            p.daemon = True;
            p.start()
        for remote in self.work_remotes: remote.close()

    def reset(self):
        for remote in self.remotes: remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def step(self, blue_actions, red_actions_batch=None):
        for i, remote in enumerate(self.remotes):
            r_act = red_actions_batch[i] if red_actions_batch is not None else None
            remote.send(('step', (blue_actions[i], r_act)))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, terms, truncs, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.array(terms), np.array(truncs), infos

    def call(self, method_name, *args, **kwargs):
        for remote in self.remotes: remote.send(('call', (method_name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes: remote.send(('close', None))
        for p in self.ps: p.join()


class CurriculumManager:
    def __init__(self, sp_manager):
        self.sp_manager = sp_manager;
        self.phase = 1;
        self.win_buffer = []

    def update(self, outcomes, global_step):
        if not outcomes: return self.phase
        # Only count strict active 'win' as a win for curriculum progression
        won = [1.0 if r == "win" else 0.0 for r in outcomes]
        if won: self.win_buffer.append(np.mean(won))
        if len(self.win_buffer) > 50: self.win_buffer.pop(0)
        avg_win = np.mean(self.win_buffer) if self.win_buffer else 0.0
        if self.phase == 1 and avg_win > 0.85:
            print(f"\n🚀 Advancing to Phase 2 (Basic Combat)");
            self.phase = 2;
            self.win_buffer = []
        elif self.phase == 2 and avg_win > 0.70:
            print(f"\n🚀 Advancing to Phase 3 (Advanced/Self-Play)");
            self.phase = 3;
            self.win_buffer = []
        return self.phase


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env): super().__init__(env)

    def set_phase(self, p): self.env.unwrapped.set_phase(p)

    def set_kappa(self, k): self.env.unwrapped.set_kappa(k)

    def set_global_step(self, s): self.env.unwrapped.set_global_step(s)

    def step(self, action, **kwargs): return self.env.step(action, **kwargs)


def make_env(): return CurriculumWrapper(AirCombatEnv())


def load_latest_checkpoint(model, optimizer):
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    files = glob.glob("checkpoints/model_*.pt")
    if not files: return 1
    latest = max(files, key=os.path.getctime)
    print(f"Loading {latest}...")
    try:
        ckpt = torch.load(latest, map_location=Config.DEVICE)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt['model_state_dict'].items()})
        if 'optimizer_state_dict' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt.get('update', 0) + 1
    except Exception as e:
        print(f"Error loading checkpoint: {e}"); return 1


def train(start_phase=1):
    run_name = f"AirCombat_Metrics_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"Log: {run_name}")

    sys_mon = SystemMonitor()
    envs = ParallelMultiAgentEnv([make_env for _ in range(Config.NUM_ENVS)])
    model = HybridActorCritic().to(Config.DEVICE)
    agent = PPOAgent(model)
    sp_manager = SelfPlayManager(phase=start_phase)
    curr_manager = CurriculumManager(sp_manager)

    start_update = load_latest_checkpoint(model, agent.optimizer)
    total_agents = Config.NUM_ENVS * Config.N_AGENTS

    obs_np, info = envs.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32).to(Config.DEVICE)
    gru_state = torch.zeros(1, total_agents, Config.D_MODEL).to(Config.DEVICE)
    dones_flags = torch.zeros(total_agents).to(Config.DEVICE)

    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE

    for update in tqdm(range(start_update, num_updates + 1)):
        step_idx = update * Config.BATCH_SIZE
        b_obs, b_actions, b_logprobs, b_rewards, b_dones = [], [], [], [], []
        b_terms, b_masks, b_graphs, b_gru_states = [], [], [], []
        b_values = []

        metrics = {
            "out_wins": 0, "out_loss": 0, "out_draw": 0, "out_crash": 0, "out_passive_win": 0,
            "tac_kills": 0, "tac_fired": 0, "tac_locked_steps": 0,
            "phy_stall_steps": 0
        }
        total_steps_batch = 0
        batch_outcomes = []
        batch_breakdown = Counter()

        envs.call("set_phase", curr_manager.phase)
        envs.call("set_global_step", step_idx)
        writer.add_scalar("curriculum/phase", curr_manager.phase, step_idx)

        steps_per_update = Config.BATCH_SIZE // total_agents

        for step in range(steps_per_update):
            total_steps_batch += total_agents
            step_graphs = []
            current_red_obs = []

            for env_info in info:
                if env_info:
                    if "termination_reason" in env_info:
                        reason = env_info["termination_reason"]
                        if reason != "none": batch_outcomes.append(reason)

                        if reason == "win":
                            metrics["out_wins"] += 1
                        elif reason == "shot":
                            metrics["out_loss"] += 1
                        elif reason in ["crash", "floor_violation"]:
                            metrics["out_crash"] += 1
                        elif reason == "timeout":
                            metrics["out_draw"] += 1
                        elif reason in ["win_passive", "enemy_crash"]:
                            metrics["out_passive_win"] += 1

                    metrics["tac_kills"] += env_info.get("stat_kills", 0)
                    metrics["tac_fired"] += env_info.get("stat_missiles_fired", 0)
                    metrics["tac_locked_steps"] += env_info.get("stat_locked", 0)
                    metrics["phy_stall_steps"] += int(env_info.get("physics_stall_ratio", 0) > 0.1)

                    if "reward_breakdown" in env_info:
                        for k, v in env_info["reward_breakdown"].items(): batch_breakdown[k] += v

                    r_obs = env_info.get("red_obs")
                    if r_obs is None: r_obs = np.zeros((1, Config.OBS_DIM), dtype=np.float32)
                    current_red_obs.append(r_obs)

                if env_info and "graph_data" in env_info and env_info["graph_data"] is not None:
                    gd = env_info["graph_data"]
                    step_graphs.append(
                        Data(x=torch.tensor(gd['x']), edge_index=torch.tensor(gd['edge_index'], dtype=torch.long),
                             edge_attr=torch.tensor(gd['edge_attr'])))
                else:
                    step_graphs.append(Data(x=torch.zeros(1, 12), edge_index=torch.zeros(2, 0, dtype=torch.long),
                                            edge_attr=torch.zeros(0, Config.GNN_EDGE_DIM)))

            graph_batch = Batch.from_data_list(step_graphs).to(Config.DEVICE)
            b_graphs.append(step_graphs)

            flat_obs = obs.view(total_agents, -1)
            with torch.no_grad():
                action, logprob, _, values, next_gru = model.get_action_and_value(
                    flat_obs, graph_data=graph_batch, action=None, gru_state=gru_state, done=dones_flags
                )

            red_actions_batch = None
            if current_red_obs:
                try:
                    red_obs_np = np.stack(current_red_obs)
                    env_dones = dones_flags.view(Config.NUM_ENVS, Config.N_AGENTS)[:, 0]
                    red_actions_batch = sp_manager.get_action(red_obs_np, dones=env_dones)
                except:
                    pass

            env_act = action.cpu().numpy().reshape(Config.NUM_ENVS, Config.N_AGENTS, -1)
            next_obs_np, rew, term, trunc, next_info = envs.step(env_act, red_actions_batch)

            is_alive = (flat_obs[:, 0] > 0.5).float()
            b_masks.append(is_alive)
            b_obs.append(flat_obs)
            b_actions.append(action)
            b_logprobs.append(logprob)
            b_rewards.append(torch.tensor(rew, dtype=torch.float32).to(Config.DEVICE).view(-1))
            b_values.append(values.view(-1))
            b_gru_states.append(gru_state)

            dones_np = np.logical_or(term, trunc)
            dones_expanded = np.repeat(dones_np[:, np.newaxis], Config.N_AGENTS, axis=1).flatten()
            terms_expanded = np.repeat(term[:, np.newaxis], Config.N_AGENTS, axis=1).flatten()
            dones_flags = torch.tensor(dones_expanded, dtype=torch.float32).to(Config.DEVICE)
            term_flags = torch.tensor(terms_expanded, dtype=torch.float32).to(Config.DEVICE)

            b_dones.append(dones_flags)
            b_terms.append(term_flags)

            obs = torch.tensor(next_obs_np, dtype=torch.float32).to(Config.DEVICE)
            gru_state = next_gru
            info = next_info

        curr_manager.update(batch_outcomes, step_idx)

        def align_buffer(buf_list):
            stacked = torch.stack(buf_list)
            permuted = stacked.permute(1, 0, 2) if len(stacked.shape) > 2 else stacked.permute(1, 0)
            return permuted.reshape(-1, *permuted.shape[2:])

        t_obs = align_buffer(b_obs)
        t_actions = align_buffer(b_actions)
        t_logprobs = align_buffer(b_logprobs).flatten()
        t_rewards = align_buffer(b_rewards).flatten()
        t_values = align_buffer(b_values).flatten()
        t_dones = align_buffer(b_dones).flatten()
        t_terms = align_buffer(b_terms).flatten()
        t_masks = align_buffer(b_masks).flatten()

        t_gru_states = torch.stack(b_gru_states).view(-1, Config.D_MODEL)

        flat_time_major = [g for step_gs in b_graphs for g in step_gs for _ in range(Config.N_AGENTS)]
        flat_agent_major = []
        for i in range(total_agents): flat_agent_major.extend(flat_time_major[i::total_agents])

        with torch.no_grad():
            last_graphs = []
            for inf in next_info:
                if inf and "graph_data" in inf:
                    gd = inf["graph_data"]
                    last_graphs.append(
                        Data(x=torch.tensor(gd['x']), edge_index=torch.tensor(gd['edge_index'], dtype=torch.long),
                             edge_attr=torch.tensor(gd['edge_attr'])))
                else:
                    last_graphs.append(Data(x=torch.zeros(1, 12), edge_index=torch.zeros(2, 0, dtype=torch.long),
                                            edge_attr=torch.zeros(0, Config.GNN_EDGE_DIM)))

            last_batch = Batch.from_data_list(last_graphs).to(Config.DEVICE)
            next_val = model.get_value(last_batch, obs.view(total_agents, -1), gru_state, dones_flags).view(-1)

            advantages = torch.zeros_like(t_rewards).to(Config.DEVICE)
            lastgaelam = 0

            r_rew = t_rewards.view(total_agents, steps_per_update)
            r_val = t_values.view(total_agents, steps_per_update)
            r_term = t_terms.view(total_agents, steps_per_update)
            r_adv = torch.zeros_like(r_rew)

            for t in reversed(range(steps_per_update)):
                nextvalues = next_val if t == steps_per_update - 1 else r_val[:, t + 1]
                nextnonterminal = 1.0 - (term_flags if t == steps_per_update - 1 else r_term[:, t + 1])
                delta = r_rew[:, t] + Config.GAMMA * nextvalues * nextnonterminal - r_val[:, t]
                r_adv[:, t] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * nextnonterminal * lastgaelam

            advantages = r_adv.flatten()
            returns = advantages + t_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        train_stats = agent.update(
            obs=t_obs, actions=t_actions, logprobs=t_logprobs, returns=returns, advantages=advantages,
            global_states=flat_agent_major, gru_states=t_gru_states, dones=t_dones,
            old_values=t_values, active_masks=t_masks
        )

        # ===============================================================
        # METRICS LOGGING - 4 DASHBOARDS
        # ===============================================================
        
        total_episodes = metrics["out_wins"] + metrics["out_loss"] + metrics["out_draw"] + metrics["out_crash"] + \
                         metrics["out_passive_win"]
        
        # A. OUTCOME DASHBOARD (Did we win?)
        if total_episodes > 0:
            # Consolidate all wins (active + passive)
            total_wins = metrics["out_wins"] + metrics["out_passive_win"]
            writer.add_scalar("outcome/win_rate", total_wins / total_episodes, step_idx)
            writer.add_scalar("outcome/loss_rate", metrics["out_loss"] / total_episodes, step_idx)
            writer.add_scalar("outcome/draw_rate", metrics["out_draw"] / total_episodes, step_idx)
            writer.add_scalar("outcome/crash_rate", metrics["out_crash"] / total_episodes, step_idx)
        
        # B. TACTICS DASHBOARD (How are we fighting?)
        if total_episodes > 0:
            # Kill ratio: kills / deaths (loss means death)
            kill_ratio = metrics["tac_kills"] / max(metrics["out_loss"], 1)  # Avoid div by 0
            writer.add_scalar("tactics/kill_ratio", kill_ratio, step_idx)
            
            # Missile hit rate (efficiency)
            if metrics["tac_fired"] > 0:
                writer.add_scalar("tactics/missile_hit_rate", metrics["tac_kills"] / metrics["tac_fired"], step_idx)
            
            # Aggression: missiles per episode
            writer.add_scalar("tactics/aggression", metrics["tac_fired"] / total_episodes, step_idx)
        
        # Lock duration: % of time locked on enemy
        if total_steps_batch > 0:
            writer.add_scalar("tactics/lock_duration", metrics["tac_locked_steps"] / total_steps_batch, step_idx)
        
        # C. PHYSICS DASHBOARD (How are we flying?)
        if total_steps_batch > 0:
            # Stall rate: % of steps in stall
            writer.add_scalar("physics/stall_rate", metrics["phy_stall_steps"] / total_steps_batch, step_idx)
        
        # D. TRAINING DASHBOARD (Is the brain healthy?)
        writer.add_scalar("training/entropy", train_stats['entropy'], step_idx)
        writer.add_scalar("training/approx_kl", train_stats['kl'], step_idx)
        writer.add_scalar("training/clip_fraction", train_stats['clip_frac'], step_idx)
        writer.add_scalar("training/explained_variance", train_stats['explained_var'], step_idx)
        
        # Optional: Keep detailed reward breakdown
        writer.add_scalar("rewards/total", torch.mean(t_rewards).item(), step_idx)
        if total_steps_batch > 0:
            for k, v in batch_breakdown.items(): 
                writer.add_scalar(f"rewards/{k}", v / total_steps_batch, step_idx)
        
        # Hardware metrics
        hw = sys_mon.get_stats()
        for k, v in hw.items(): writer.add_scalar(k, v, step_idx)

        if update % Config.SAVE_INTERVAL == 0:
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'update': update}, "checkpoints/model_latest.pt")
            if curr_manager.phase >= 3 and sp_manager.evaluate_candidate(model, make_env, curr_manager.phase):
                save_path = f"checkpoints/model_{update}.pt"
                torch.save({'model_state_dict': model.state_dict()}, save_path)
                sp_manager.opponent_pool.append({'path': save_path, 'win_rate': 0.5, 'step': step_idx})
                sp_manager.save_pool_metadata()

        if update % 5 == 0: sp_manager.sample_opponent(step_idx)

    envs.close();
    writer.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser();
    parser.add_argument('--phase', type=int, default=1)
    args = parser.parse_args()
    train(start_phase=args.phase)