# ================================================
# FILE: train.py (PARALLELIZED)
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

# PyG Imports
from torch_geometric.data import Data, Batch

# Local Imports
from src.env import AirCombatEnv
from src.model import HybridActorCritic
from src.self_play import SelfPlayManager
from config import Config


# --- HARDWARE MONITOR ---
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


# --- PARALLEL VECTOR ENV (Truth 2) ---
# Custom implementation because standard AsyncVectorEnv doesn't support 'red_actions' arg in step()
def worker(remote, parent_remote, env_fn_wrapper, seed):
    """
    Worker process for parallel environment execution.
    FIX: Now accepts a 'seed' argument to ensure data diversity.
    """
    # 1. Set global seeds for this specific process
    # Intuition: Ensure that any random call (numpy, torch, random) inside this process
    # diverges from other processes.
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parent_remote.close()

    # 2. Instantiate the environment
    env = env_fn_wrapper()

    # 3. Initial Reset with Seed
    # Intuition: Gym environments need an initial seed passed to reset() to
    # initialize their internal RNG (self.np_random).
    env.reset(seed=seed)

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                blue_act, red_act = data
                ob, reward, term, trunc, info = env.step(blue_act, red_actions=red_act)
                if term or trunc:
                    # Auto-reset logic
                    # Note: We don't need to pass seed here again; Gym maintains the RNG state
                    # from the initial seed.
                    ob_reset, info_reset = env.reset()

                    # Propagate info needed for next step
                    info['red_obs'] = info_reset.get('red_obs')
                    info['graph_data'] = info_reset.get('graph_data')
                    ob = ob_reset
                remote.send((ob, reward, term, trunc, info))
            elif cmd == 'reset':
                # Allow manual reset with new random state if requested,
                # otherwise continue stream
                ob, info = env.reset()
                remote.send((ob, info))
            elif cmd == 'call':
                name, args, kwargs = data
                res = getattr(env.unwrapped, name)(*args, **kwargs)
                remote.send(res)
            elif cmd == 'close':
                env.close()
                remote.close()
                break
        except EOFError:
            break


class ParallelMultiAgentEnv:
    """
    Custom Parallel Environment Manager.
    FIX: Passes unique seeds to workers upon initialization.
    """

    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])

        # Generate a base seed based on time to ensure different runs have different seeds

        base_seed = int(time.time())

        self.ps = []
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            # Calculate unique seed for this worker (Base + Rank)
            worker_seed = base_seed + i

            # Start process with the unique seed
            p = mp.Process(target=worker, args=(work_remote, remote, env_fn, worker_seed))
            self.ps.append(p)

        for p in self.ps:
            p.daemon = True  # Clean up if main process dies
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
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
        for remote in self.remotes:
            remote.send(('call', (method_name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


# --- MANAGERS ---
class CurriculumManager:
    def __init__(self, sp_manager):
        self.sp_manager = sp_manager
        self.phase = 1
        self.win_buffer = []

    def update(self, outcomes, global_step):
        if not outcomes: return self.phase
        won = [1.0 if r == "win" else 0.0 for r in outcomes]
        if won: self.win_buffer.append(np.mean(won))
        if len(self.win_buffer) > 50: self.win_buffer.pop(0)

        avg_win = np.mean(self.win_buffer) if self.win_buffer else 0.0

        if self.phase == 1 and avg_win > 0.80:
            print(f"\n🚀 Phase 1 -> 2")
            self.phase = 2
            self.win_buffer = []
        elif self.phase == 2 and avg_win > 0.60:
            print(f"\n🚀 Phase 2 -> 3")
            self.phase = 3
            self.win_buffer = []
        elif self.phase == 3 and avg_win > 0.60 and global_step > 500_000:
            print(f"\n🚀 Phase 3 -> 4")
            self.phase = 4
            self.win_buffer = []
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
    latest = max(files, key=os.path.getctime)
    print(f"Loading {latest}...")
    try:
        ckpt = torch.load(latest, map_location=Config.DEVICE)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt['model_state_dict'].items()})
        if 'optimizer_state_dict' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt.get('update', 0) + 1
    except:
        return 1


# --- TRAIN ---
def train(start_phase=1):
    run_name = f"AirCombat_Parallel_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"Log: {run_name}")

    sys_mon = SystemMonitor()
    # Updated: Use Parallel Env
    envs = ParallelMultiAgentEnv([make_env for _ in range(Config.NUM_ENVS)])

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
    print(f"Training: {num_updates} updates | Parallelized")

    for update in tqdm(range(start_update, num_updates + 1)):
        step_idx = update * Config.BATCH_SIZE

        # Buffers
        # Added b_terms to track true terminations (death/win) separately from resets
        b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values = [], [], [], [], [], []
        b_terms = []  # <--- NEW BUFFER
        b_gru_states, b_graph_lists = [], []

        batch_outcomes = []
        batch_stats = Counter()
        batch_breakdown = Counter()

        red_obs_list = []

        envs.call("set_phase", curr_manager.phase)
        envs.call("set_global_step", step_idx)
        writer.add_scalar("curriculum/phase", curr_manager.phase, step_idx)

        steps_per_update = Config.BATCH_SIZE // total_agents

        for step in range(steps_per_update):
            step_graphs = []
            current_red_obs = []

            for env_info in info:
                if env_info:
                    if "termination_reason" in env_info and env_info["termination_reason"] != "none":
                        batch_outcomes.append(env_info["termination_reason"])
                    if "stat_kills" in env_info: batch_stats['kills'] += env_info["stat_kills"]
                    if "stat_missiles_fired" in env_info: batch_stats['fired'] += env_info["stat_missiles_fired"]

                    if "reward_breakdown" in env_info:
                        for k, v in env_info["reward_breakdown"].items():
                            batch_breakdown[k] += v

                    r_obs = env_info.get("red_obs")
                    if r_obs is None: r_obs = np.zeros((1, Config.OBS_DIM), dtype=np.float32)
                    current_red_obs.append(r_obs)

                if env_info and "graph_data" in env_info and env_info["graph_data"] is not None:
                    gd = env_info["graph_data"]
                    x_t = torch.tensor(gd['x'], dtype=torch.float32)
                    edge_index_t = torch.tensor(gd['edge_index'], dtype=torch.long)
                    edge_attr_t = torch.tensor(gd['edge_attr'], dtype=torch.float32)
                    if edge_index_t.ndim == 1 or edge_index_t.numel() == 0:
                        edge_index_t = torch.zeros((2, 0), dtype=torch.long)
                        edge_attr_t = torch.zeros((0, 6), dtype=torch.float32)
                    step_graphs.append(Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t))
                else:
                    step_graphs.append(Data(x=torch.zeros(1, 12), edge_index=torch.zeros(2, 0, dtype=torch.long),
                                            edge_attr=torch.zeros(0, 6)))

            red_obs_list.extend(current_red_obs)
            graph_batch = Batch.from_data_list(step_graphs).to(Config.DEVICE)
            flat_obs = obs.view(total_agents, -1)

            with torch.no_grad():
                action, logprob, _, values, next_gru = model.get_action_and_value(
                    flat_obs, graph_data=graph_batch, action=None, gru_state=gru_state, done=done_flags
                )

            red_actions_batch = None
            if current_red_obs:
                try:
                    red_obs_np = np.stack(current_red_obs)
                    red_actions_batch = sp_manager.get_action(red_obs_np)
                except:
                    red_actions_batch = None

            env_act = action.cpu().numpy().reshape(Config.NUM_ENVS, Config.N_AGENTS, -1)
            next_obs_np, rew, term, trunc, next_info = envs.step(env_act, red_actions_batch)

            b_obs.append(flat_obs)
            b_actions.append(action)
            b_logprobs.append(logprob)
            b_rewards.append(torch.tensor(rew, dtype=torch.float32).to(Config.DEVICE).view(-1))

            # Logic:
            # 'term' = True Death/Win. Value should be 0 (masked).
            # 'trunc' = Time Limit. Value should be V(next) (not masked).
            # 'done' = Reset trigger. Used for GRU reset and Environment reset.

            dones_np = np.logical_or(term, trunc)  # For GRU reset

            # Expand dimensions to match agents
            dones_expanded = np.repeat(dones_np[:, np.newaxis], Config.N_AGENTS, axis=1).flatten()
            terms_expanded = np.repeat(term[:, np.newaxis], Config.N_AGENTS, axis=1).flatten()  # <--- NEW

            done_flags = torch.tensor(dones_expanded, dtype=torch.float32).to(Config.DEVICE)
            term_flags = torch.tensor(terms_expanded, dtype=torch.float32).to(Config.DEVICE)  # <--- NEW

            b_dones.append(done_flags)  # Used for GRU resetting
            b_terms.append(term_flags)  # Used for GAE masking <--- NEW

            b_values.append(values.view(-1))
            b_gru_states.append(gru_state.detach())
            b_graph_lists.append(step_graphs)

            obs = torch.tensor(next_obs_np, dtype=torch.float32).to(Config.DEVICE)
            gru_state = next_gru
            info = next_info

        curr_manager.update(batch_outcomes, step_idx)

        t_obs = torch.stack(b_obs).view(-1, Config.OBS_DIM)
        t_actions = torch.stack(b_actions).view(-1, Config.ACTION_DIM)
        t_logprobs = torch.stack(b_logprobs).view(-1)
        t_rewards = torch.stack(b_rewards).view(-1)

        # --- MISSING LINE ---
        t_terms = torch.stack(b_terms).view(-1)
        # --------------------

        t_values = torch.stack(b_values).view(-1)
        t_gru_states = torch.stack(b_gru_states).view(-1, 1, Config.D_MODEL)

        flat_graph_list = []
        for step_g_list in b_graph_lists:
            for g_data in step_g_list:
                for _ in range(Config.N_AGENTS):
                    flat_graph_list.append(g_data.clone())

        # GAE
        with torch.no_grad():
            last_graphs = []
            for inf in next_info:
                if inf and "graph_data" in inf:
                    gd = inf["graph_data"]
                    x_t = torch.tensor(gd['x'], dtype=torch.float32)
                    edge_index_t = torch.tensor(gd['edge_index'], dtype=torch.long)
                    if edge_index_t.numel() == 0: edge_index_t = torch.zeros((2, 0), dtype=torch.long)
                    edge_attr_t = torch.tensor(gd['edge_attr'], dtype=torch.float32)
                    last_graphs.append(Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t))
                else:
                    last_graphs.append(Data(x=torch.zeros(1, 12), edge_index=torch.zeros(2, 0, dtype=torch.long),
                                            edge_attr=torch.zeros(0, 6)))

            last_batch = Batch.from_data_list(last_graphs).to(Config.DEVICE)
            # Get next value
            next_val = model.get_value(last_batch, obs.view(total_agents, -1), gru_state=gru_state,
                                       done=done_flags).view(-1)

            advantages = torch.zeros_like(t_rewards).to(Config.DEVICE)
            lastgaelam = 0

            r_rewards = t_rewards.view(steps_per_update, total_agents)
            r_terms = t_terms.view(steps_per_update, total_agents)  # <--- USE TERMS NOT DONES
            r_values = t_values.view(steps_per_update, total_agents)

            # IMPORTANT: For bootstrapping, we use the done_flags (reset) to know if we switch episodes,
            # BUT for value masking, we use term_flags.
            # Actually, standard PPO implementation:
            # Mask = 1 - done (if done, next val is 0).
            # Here, if truncated, we WANT next val. So Mask = 1 - term.

            # Correct logic:
            # If term=True (died), mask=0. We don't care about next value.
            # If trunc=True (timeout), term=False, mask=1. We add gamma * next_val.

            for t in reversed(range(steps_per_update)):
                # If t is last step, use next_val from outside loop.
                # Else use value from t+1
                nextvalues = next_val if t == steps_per_update - 1 else r_values[t + 1]

                # FIX: We mask ONLY if it was a true termination (death/win).
                # If it was a truncation (time limit), term is False, so nonterminal is 1.0.
                # We effectively bootstrap across the time limit boundary using the Critic's prediction.
                nextnonterminal = 1.0 - (term_flags if t == steps_per_update - 1 else r_terms[t + 1])

                delta = r_rewards[t] + Config.GAMMA * nextvalues * nextnonterminal - r_values[t]
                advantages[t * total_agents: (t + 1) * total_agents] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * nextnonterminal * lastgaelam

            returns = advantages + t_values

        y_pred = t_values.cpu().numpy()
        y_true = returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        b_inds = np.arange(len(t_obs))
        pg_losses, v_losses, ent_losses, approx_kls = [], [], [], []
        # Global Advantage Normalization ---
        # Intuition: Normalize across the full batch (3840 steps) to stabilize the signal.
        # Doing this inside the minibatch loop introduces high variance noise.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # --- FIX END ---

        for epoch in range(Config.UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, len(t_obs), Config.MINIBATCH_SIZE):
                end = start + Config.MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                mb_graphs = [flat_graph_list[i] for i in mb_inds]
                mb_graph_batch = Batch.from_data_list(mb_graphs).to(Config.DEVICE)

                _, new_logprob, entropy, new_values, _ = model.get_action_and_value(
                    t_obs[mb_inds], graph_data=mb_graph_batch,
                    action=t_actions[mb_inds], gru_state=t_gru_states[mb_inds].permute(1, 0, 2)
                )

                logratio = new_logprob - t_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kls.append(approx_kl.item())

                mb_adv = advantages[mb_inds]

                pg_loss = -torch.min(mb_adv * ratio,
                                     mb_adv * torch.clamp(ratio, 1 - Config.CLIP_COEF, 1 + Config.CLIP_COEF)).mean()
                v_loss = 0.5 * ((new_values.view(-1) - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - Config.ENT_COEF * entropy_loss + Config.VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())

        hw = sys_mon.get_stats()
        for k, v in hw.items(): writer.add_scalar(k, v, step_idx)

        writer.add_scalar("combat/kills_total", batch_stats['kills'], step_idx)
        writer.add_scalar("combat/missiles_fired", batch_stats['fired'], step_idx)
        writer.add_scalar("train/loss", np.mean(pg_losses), step_idx)
        writer.add_scalar("rewards/total", torch.mean(t_rewards).item(), step_idx)
        writer.add_scalar("train/opponent_type", 1.0 if sp_manager.current_opponent_type == "model" else 0.0, step_idx)
        writer.add_scalar("losses/explained_variance", explained_var, step_idx)
        writer.add_scalar("losses/approx_kl", np.mean(approx_kls), step_idx)
        writer.add_scalar("losses/entropy", np.mean(ent_losses), step_idx)
        writer.add_scalar("losses/value_loss", np.mean(v_losses), step_idx)

        total_steps = len(t_rewards)
        writer.add_scalar("rewards/component_kill", batch_breakdown['rew_kill'] / total_steps, step_idx)
        writer.add_scalar("rewards/component_pos", batch_breakdown['rew_pos'] / total_steps, step_idx)
        writer.add_scalar("rewards/component_survival", batch_breakdown['rew_survival'] / total_steps, step_idx)
        writer.add_scalar("rewards/component_penalty", batch_breakdown['rew_penalty'] / total_steps, step_idx)

        if update % Config.SAVE_INTERVAL == 0:
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'update': update}, "checkpoints/model_latest.pt")

            # Only add to pool if we are in Self-Play phase (3+) AND the model is good
            if curr_manager.phase >= 3 and sp_manager.evaluate_candidate(model, make_env, curr_manager.phase):
                save_path = f"checkpoints/model_{update}.pt"
                torch.save({'model_state_dict': model.state_dict()}, save_path)
                # Add to pool with 0.5 win rate (default assumption)
                sp_manager.opponent_pool.append({'path': save_path, 'win_rate': 0.5})
                # Refresh file list in manager
                sp_manager.load_checkpoints_list()

        OPPONENT_SWITCH_INTERVAL = 5

        if update % OPPONENT_SWITCH_INTERVAL == 0:
            sp_manager.sample_opponent(step_idx)

    envs.close();
    writer.close()


if __name__ == "__main__":
    # Required for multiprocessing on Windows/MacOS
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1)
    args = parser.parse_args()
    train(start_phase=args.phase)