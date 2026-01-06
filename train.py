# ================================================
# FILE: train.py
# ================================================
import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from src.model import HybridActorCritic, AirCombatDiscriminator
from src.ppo import PPOAgent
from src.self_play import SelfPlayManager
from config import Config
from torch.utils.data import DataLoader
from pretrain import load_or_collect_data, SequenceDataset, collate_sequences


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
                    info["terminal_observation"] = ob.copy() if isinstance(ob, np.ndarray) else ob
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
        won = [1.0 if r in ["win", "win_passive"] else 0.0 for r in outcomes]
        if won: self.win_buffer.extend(won)
        if len(self.win_buffer) > 100: self.win_buffer = self.win_buffer[-100:]
        avg_win = np.mean(self.win_buffer) if self.win_buffer else 0.0

        if self.phase == 1:
            if avg_win > 0.60:  # Lowered threshold
                print(f"\n🚀 PROMOTION: Phase 2");
                self.phase = 2;
                self.win_buffer = [];
                self.sp_manager.kappa = 0.5
        elif self.phase == 2:
            if avg_win > 0.55 and global_step > 200_000:  # Lowered threshold
                print(f"\n🚀 PROMOTION: Phase 3");
                self.phase = 3;
                self.win_buffer = [];
                self.sp_manager.kappa = 0.0
        return self.phase


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def set_phase(self, p):
        self.env.unwrapped.set_phase(p)

    def set_kappa(self, k):
        self.env.unwrapped.set_kappa(k)

    def set_global_step(self, s):
        self.env.unwrapped.set_global_step(s)

    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # NOTE: Logic to strip ammo removed as discussed.
        return obs, info


def make_env(): return CurriculumWrapper(AirCombatEnv())


def load_latest_checkpoint(model, optimizer):
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    files = glob.glob("checkpoints/model_*.pt")
    if not files: return 1
    latest = None
    numbered_files = []
    for f in files:
        match = re.search(r'model_(\d+).pt', f)
        if match: numbered_files.append((int(match.group(1)), f))
    if numbered_files:
        _, latest_file = max(numbered_files, key=lambda x: x[0]);
        latest = latest_file
    elif os.path.exists("checkpoints/model_latest.pt"):
        latest = "checkpoints/model_latest.pt"
    elif os.path.exists("checkpoints/model_pretrained.pt"):
        latest = "checkpoints/model_pretrained.pt"
    else:
        latest = max(files, key=os.path.getctime)

    print(f"Loading {latest}...")
    try:
        ckpt = torch.load(latest, map_location=Config.DEVICE)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        opt_dict = ckpt.get('optimizer_state_dict', None) if isinstance(ckpt, dict) else None
        update = ckpt.get('update', 0) if isinstance(ckpt, dict) else 0
        is_pretrained = "pretrained" in latest
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        if opt_dict is not None and not is_pretrained:
            try:
                optimizer.load_state_dict(opt_dict);
                print("✅ Optimizer state restored.")
            except:
                print("⚠️ Optimizer load failed")
        elif is_pretrained:
            print("✨ Loaded Pretrained Weights. Resetting Optimizer.")
            return 1
        return update + 1
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return 1


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
    if start_phase != 1: curr_manager.phase = start_phase

    # --- GAIL SETUP ---
    discriminator = AirCombatDiscriminator().to(Config.DEVICE)
    opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4)

    print("Loading expert data for GAIL...")
    phase_files = load_or_collect_data()

    # UPDATED: Load sharded data preserving sequence structure
    full_obs_list, full_graphs_list, full_acts_list, full_rets_list, full_masks_list = [], [], [], [], []

    for _, fpath in phase_files:
        d = torch.load(fpath, weights_only=False)
        full_obs_list.extend(d[0])
        full_graphs_list.extend(d[1])
        full_acts_list.extend(d[2])
        full_rets_list.extend(d[3])
        full_masks_list.extend(d[4])

    # DO NOT flatten lists. Keep them as lists of sequences/graphs for SequenceDataset
    full_obs = full_obs_list
    full_graphs = full_graphs_list
    full_acts = full_acts_list
    full_rets = full_rets_list
    full_masks = full_masks_list

    # Clear temp lists
    del full_obs_list, full_graphs_list, full_acts_list, full_rets_list, full_masks_list

    # PPO interprets 'BATCH_SIZE' as total TIMESTEPS.
    # We must divide by SEQ_LEN to align memory usage.
    expert_seq_batch_size = max(1, Config.BATCH_SIZE // Config.SEQ_LEN)

    print(f"GAIL: Loading {expert_seq_batch_size} sequences per batch (approx {Config.BATCH_SIZE} steps)")

    gail_dataset = SequenceDataset(full_obs, full_graphs, full_acts, full_rets, full_masks)
    # Ensure drop_last=True to avoid small batches if data is tight
    expert_loader = DataLoader(gail_dataset, batch_size=expert_seq_batch_size, shuffle=True,
                               collate_fn=collate_sequences, drop_last=True)
    expert_iter = iter(expert_loader)
    # ------------------

    total_agents = Config.NUM_ENVS * Config.N_AGENTS
    obs_np, info = envs.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32).to(Config.DEVICE)
    gru_state = torch.zeros(1, total_agents, Config.D_MODEL).to(Config.DEVICE)
    dones_flags = torch.zeros(total_agents).to(Config.DEVICE)

    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE

    for update in tqdm(range(start_update, num_updates + 1)):
        step_idx = update * Config.BATCH_SIZE

        b_obs, b_next_obs, b_actions, b_logprobs, b_rewards, b_dones = [], [], [], [], [], []
        b_terms, b_masks, b_graphs, b_gru_states = [], [], [], []
        b_values = []

        metrics = {"out_wins": 0, "out_loss": 0, "out_draw": 0, "out_crash": 0, "out_passive_win": 0,
                   "tac_kills": 0, "tac_fired": 0, "tac_locked_steps": 0, "phy_stall_steps": 0}
        total_steps_batch = 0;
        batch_outcomes = [];
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

                # --- FIX: GRAPH REPLICATION ---
                gd_data = None
                if env_info and "graph_data" in env_info and env_info["graph_data"] is not None:
                    gd = env_info["graph_data"]
                    gd_data = Data(x=torch.tensor(gd['x'], dtype=torch.float32),
                                   edge_index=torch.tensor(gd['edge_index'], dtype=torch.long),
                                   edge_attr=torch.tensor(gd['edge_attr'], dtype=torch.float32))
                else:
                    gd_data = Data(x=torch.zeros(1, Config.NODE_DIM, dtype=torch.float32),
                                   edge_index=torch.zeros(2, 0, dtype=torch.long),
                                   edge_attr=torch.zeros(0, Config.EDGE_DIM, dtype=torch.float32))

                # Replicate for all agents in this env
                for _ in range(Config.N_AGENTS):
                    step_graphs.append(gd_data)

            b_graphs.append(step_graphs)
            graph_batch = Batch.from_data_list(step_graphs).to(Config.DEVICE)

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
            real_next_obs = next_obs_np.copy()
            for i, inf in enumerate(next_info):
                if (dones_np[i].any()) and "terminal_observation" in inf:
                    real_next_obs[i] = inf["terminal_observation"]

            t_real_next = torch.tensor(real_next_obs, dtype=torch.float32).to(Config.DEVICE)
            b_next_obs.append(t_real_next.view(total_agents, -1))

            dones_expanded = np.repeat(dones_np[:, np.newaxis], Config.N_AGENTS, axis=1).flatten()
            terms_expanded = np.repeat(term[:, np.newaxis], Config.N_AGENTS, axis=1).flatten()
            dones_flags = torch.tensor(dones_expanded, dtype=torch.float32).to(Config.DEVICE)
            term_flags = torch.tensor(terms_expanded, dtype=torch.float32).to(Config.DEVICE)

            b_dones.append(dones_flags);
            b_terms.append(term_flags)
            obs = torch.tensor(next_obs_np, dtype=torch.float32).to(Config.DEVICE)
            info = next_info

        curr_manager.update(batch_outcomes, step_idx)

        # Buffer Alignment
        def align_buffer(buf_list):
            stacked = torch.stack(buf_list)
            if stacked.ndim == 2: stacked = stacked.unsqueeze(-1)
            if stacked.ndim == 4 and stacked.shape[1] == 1: stacked = stacked.squeeze(1)
            # Reshape to (Steps, Envs, Agents, Dim)
            if stacked.ndim == 3: stacked = stacked.view(len(buf_list), Config.NUM_ENVS, Config.N_AGENTS, -1)
            # Permute to (Envs, Agents, Steps, Dim) -> Matches Agent-Major Order
            permuted = stacked.permute(1, 2, 0, 3)
            return permuted.reshape(-1, *permuted.shape[3:])

        t_obs = align_buffer(b_obs)
        t_next_obs = align_buffer(b_next_obs)
        t_actions = align_buffer(b_actions)
        t_logprobs = align_buffer(b_logprobs).flatten()
        t_rewards = align_buffer(b_rewards).flatten()
        t_values = align_buffer(b_values).flatten()
        t_dones = align_buffer(b_dones).flatten()
        t_terms = align_buffer(b_terms).flatten()
        t_masks = align_buffer(b_masks).flatten()
        t_gru_states = align_buffer(b_gru_states)

        # --- FIX: ALIGN GRAPHS ---
        flat_agent_graphs = []
        n_steps_collected = len(b_graphs)
        if n_steps_collected > 0:
            n_total_agents = len(b_graphs[0])
            for a in range(n_total_agents):
                for t in range(n_steps_collected):
                    flat_agent_graphs.append(b_graphs[t][a])

        # Re-batch graphs for GAIL and PPO
        t_graphs = Batch.from_data_list(flat_agent_graphs).to(Config.DEVICE)

        # --- GAIL REWARD CALCULATION ---
        with torch.no_grad():
            disc_logits = discriminator(t_graphs, t_obs, t_actions)
            prob_expert = torch.sigmoid(disc_logits)
            # -log(1 - D(s,a)) -> Higher reward if D predicts Expert (1)
            r_gail = -torch.log(1.0 - prob_expert + 1e-8)
            r_gail = r_gail.view(-1) * 0.1  # Lambda

        # Fuse Rewards: Env + GAIL
        t_total_rewards = t_rewards + r_gail

        # --- GAE RE-CALCULATION with Fused Rewards ---
        with torch.no_grad():
            # Get next value
            last_graphs = []
            for env_info in next_info:
                gd_data = None
                if env_info and "graph_data" in env_info:
                    gd = env_info["graph_data"]
                    gd_data = Data(x=torch.tensor(gd['x'], dtype=torch.float32),
                                   edge_index=torch.tensor(gd['edge_index'], dtype=torch.long),
                                   edge_attr=torch.tensor(gd['edge_attr'], dtype=torch.float32))
                else:
                    gd_data = Data(x=torch.zeros(1, Config.NODE_DIM, dtype=torch.float32),
                                   edge_index=torch.zeros(2, 0, dtype=torch.long),
                                   edge_attr=torch.zeros(0, Config.EDGE_DIM, dtype=torch.float32))
                # Replicate for GAE next_val calculation
                for _ in range(Config.N_AGENTS):
                    last_graphs.append(gd_data)

            last_batch = Batch.from_data_list(last_graphs).to(Config.DEVICE)
            next_val = model.get_value(last_batch, obs.view(total_agents, -1)).view(-1)

            r_val = t_values.view(total_agents, steps_per_update)
            r_rew = t_total_rewards.view(total_agents, steps_per_update)
            r_term = t_terms.view(total_agents, steps_per_update)

            r_adv = torch.zeros_like(r_rew)
            lastgaelam = 0
            for t in reversed(range(steps_per_update)):
                nextvalues = next_val if t == steps_per_update - 1 else r_val[:, t + 1]
                nextnonterminal = 1.0 - (term_flags if t == steps_per_update - 1 else r_term[:, t + 1])
                delta = r_rew[:, t] + Config.GAMMA * nextvalues * nextnonterminal - r_val[:, t]
                r_adv[:, t] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * nextnonterminal * lastgaelam

            advantages = r_adv.flatten()
            returns = advantages + t_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- UPDATE DISCRIMINATOR ---
        try:
            exp_batch = next(expert_iter)
        except StopIteration:
            expert_iter = iter(expert_loader)
            exp_batch = next(expert_iter)

        exp_obs = exp_batch[0].to(Config.DEVICE).view(-1, Config.OBS_DIM)
        exp_graphs = exp_batch[1].to(Config.DEVICE)
        exp_acts = exp_batch[2].to(Config.DEVICE).view(-1, Config.ACTION_DIM)
        exp_mask = exp_batch[4].to(Config.DEVICE).view(-1)

        valid_idx = exp_mask > 0.5
        if valid_idx.sum() > 0:
            real_logits = discriminator(exp_graphs, exp_obs, exp_acts)
            loss_real = F.binary_cross_entropy_with_logits(real_logits.view(-1), torch.ones_like(real_logits.view(-1)),
                                                           reduction='none')
            loss_real = (loss_real * exp_mask).sum() / (exp_mask.sum() + 1e-8)

            fake_logits = discriminator(t_graphs, t_obs.detach(), t_actions.detach())
            loss_fake = F.binary_cross_entropy_with_logits(fake_logits.view(-1), torch.zeros_like(fake_logits.view(-1)),
                                                           reduction='none')
            loss_fake = (loss_fake * t_masks).sum() / (t_masks.sum() + 1e-8)

            loss_disc = loss_real + loss_fake
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            writer.add_scalar("gail/disc_loss", loss_disc.item(), step_idx)
            writer.add_scalar("gail/reward_mean", r_gail.mean().item(), step_idx)

        # --- FREEZE LOGIC ---
        update_actor = (update > getattr(Config, 'FREEZE_ACTOR_STEPS', 0))
        if not update_actor and update % 10 == 0:
            print(f"❄️  Actor Frozen (Critic Warmup) - Step {update}/{getattr(Config, 'FREEZE_ACTOR_STEPS', 0)}")

        # --- UPDATE AGENT (PPO) ---
        train_stats = agent.update(
            obs=t_obs,
            next_obs=t_next_obs,
            actions=t_actions,
            logprobs=t_logprobs,
            returns=returns,
            advantages=advantages,
            global_states=flat_agent_graphs,
            gru_states=t_gru_states,
            dones=t_dones,
            old_values=t_values,
            active_masks=t_masks,
            update_actor=update_actor
        )
        episodes_this_batch = len(batch_outcomes)

        # Logging
        if episodes_this_batch > 0:
            total_wins = metrics["out_wins"] + metrics["out_passive_win"]
            writer.add_scalar("outcome/win_rate", total_wins / episodes_this_batch, step_idx)
            writer.add_scalar("outcome/loss_rate", metrics["out_loss"] / episodes_this_batch, step_idx)
            writer.add_scalar("outcome/crash_rate", metrics["out_crash"] / episodes_this_batch, step_idx)
            writer.add_scalar("tactics/aggression", metrics["tac_fired"] / episodes_this_batch, step_idx)

        writer.add_scalar("training/approx_kl", train_stats['kl'], step_idx)
        writer.add_scalar("training/clip_fraction", train_stats['clip_frac'], step_idx)
        writer.add_scalar("rewards/total", torch.mean(t_rewards).item(), step_idx)

        hw = sys_mon.get_stats()
        for k, v in hw.items(): writer.add_scalar(k, v, step_idx)

        if update % Config.SAVE_INTERVAL == 0:
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
                        'update': update}, "checkpoints/model_latest.pt")
            if curr_manager.phase >= 3 and sp_manager.evaluate_candidate(model, make_env, curr_manager.phase):
                save_path = f"checkpoints/model_{update}.pt"
                torch.save({'model_state_dict': model.state_dict()}, save_path)
                sp_manager.opponent_pool.append({'path': save_path, 'win_rate': 0.5, 'step': step_idx})
                sp_manager.save_pool_metadata()

        if update % 5 == 0:
            sp_manager.sample_opponent(step_idx, phase=curr_manager.phase)

    envs.close();
    writer.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2) # Default to Phase 2 (Fast-Track)
    args = parser.parse_args()
    train(start_phase=args.phase)