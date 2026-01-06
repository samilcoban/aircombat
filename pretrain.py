# ================================================
# FILE: pretrain.py
# ================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp
import numpy as np
import math
import os
import sys
import time
import glob
import re
import multiprocessing as mp
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
import gymnasium as gym
import random

# Add root directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.env import AirCombatEnv
from src.model import HybridActorCritic
from src.bot import HardcodedAce

# === CONFIGURATION ===
DATA_DIR = "data"
BATCH_SIZE = 8  # Local batch size for 4GB VRAM
GRAD_ACCUM_STEPS = 8  # Effective Batch Size = 64
SEQ_LEN = Config.SEQ_LEN
TOTAL_EPOCHS = 15  # Consolidated Epochs
MAX_PRETRAIN_STEPS = 2000
DEVICE = Config.DEVICE
LR = Config.LEARNING_RATE

# Defined globally for sharing between collect and load
PHASES = [
    ('recovery', 200_000),
    ('nav', 200_000),
    ('tail_chase', 200_000),
    ('head_on', 200_000),
    ('disadvantage', 200_000)
]


# ================================================
# 1. SCENARIO WRAPPER (The Director)
# ================================================
class ScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scenario_type = "combat"
        self.step_counter = 0

    def step(self, action, **kwargs):
        obs, reward, term, trunc, info = self.env.step(action, **kwargs)
        self.step_counter += 1

        # Short-circuit Nav/Recovery success
        if self.scenario_type in ["nav", "recovery"]:
            if self.step_counter >= 300 and not term:
                trunc = True

        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        self.step_counter = 0
        obs, info = self.env.reset(**kwargs)

        # 1. Variable Team Sizes
        n_blue = np.random.randint(1, Config.N_AGENTS + 1)
        n_red = np.random.randint(1, Config.N_ENEMIES_MAX + 1)

        active_blue = self.env.unwrapped.blue_ids[:n_blue]
        active_red = self.env.unwrapped.red_ids[:n_red]

        # Banish inactive agents
        inactive_blue = self.env.unwrapped.blue_ids[n_blue:]
        inactive_red = self.env.unwrapped.red_ids[n_red:]
        self._teleport_formation(inactive_blue, -200000, -200000, 10000, 0, 0)
        self._teleport_formation(inactive_red, 200000, 200000, 10000, 0, 0)

        # 2. Setup Scenario
        guns_only = (np.random.rand() < 0.5)

        def strip_all_ammo():
            for uid in self.env.unwrapped.blue_ids + self.env.unwrapped.red_ids:
                if uid in self.env.unwrapped.core.entities:
                    self.env.unwrapped.core.entities[uid].ammo = 0

        if guns_only:
            strip_all_ammo()

        if self.scenario_type == "recovery":
            self._teleport_formation(active_red, 200000, 200000, 10000, 0, 0)
            self._setup_recovery(active_blue)

        elif self.scenario_type == "nav":
            self._teleport_formation(active_red, 200000, 200000, 10000, 0, 0)
            self._setup_navigation(active_blue)

        elif self.scenario_type == "tail_chase":
            self._setup_tail_chase(active_blue, active_red, guns_only)

        elif self.scenario_type == "head_on":
            self._setup_head_on(active_blue, active_red, guns_only)

        elif self.scenario_type == "disadvantage":
            if not guns_only: strip_all_ammo()
            self._setup_disadvantage(active_blue, active_red, guns_only)

        # 3. Update Physics Cache
        self.env.unwrapped.core.update_spatial_cache()
        self.env.unwrapped._compute_frame_data()
        obs = self.env.unwrapped._get_all_blue_obs()
        info["red_obs"] = self.env.unwrapped._get_all_red_obs()
        info["graph_data"] = self.env.unwrapped._get_graph_state()

        info["scenario_mode"] = self.scenario_type
        info["active_blue_count"] = n_blue

        return obs, info

    def _teleport_entity(self, uid, x, y, alt, heading, speed):
        if uid not in self.env.unwrapped.core.entities: return
        ent = self.env.unwrapped.core.entities[uid]
        ent.x = x
        ent.y = y
        ent.alt = alt
        ent.heading = math.radians(heading)
        ent.speed = speed
        ent.roll = 0.0
        ent.pitch = 0.0
        # Reset derivatives to 0 to prevent physics explosion
        ent.prev_heading = ent.heading
        ent.prev_pitch = 0.0
        ent.prev_roll = 0.0
        ent.prev_speed = speed
        ent.d_heading = 0.0
        ent.d_pitch = 0.0
        ent.d_roll = 0.0
        ent.d_speed = 0.0

    def _teleport_formation(self, uids, center_x, center_y, alt, heading, speed, spacing=1000.0):
        if not uids: return
        n = len(uids)
        perp_rad = math.radians(heading + 90)
        off_x, off_y = math.cos(perp_rad), math.sin(perp_rad)
        for i, uid in enumerate(uids):
            offset = (i - (n - 1) / 2.0) * spacing
            tx = center_x + off_x * offset
            ty = center_y + off_y * offset
            self._teleport_entity(uid, tx, ty, alt, heading, speed)

    def _setup_recovery(self, blues):
        for uid in blues:
            if np.random.rand() < 0.5:
                # Dive Recovery
                self._teleport_entity(uid, np.random.uniform(-5000, 5000), np.random.uniform(-5000, 5000),
                                      4000, np.random.uniform(0, 360), 600)
                if uid in self.env.unwrapped.core.entities:
                    self.env.unwrapped.core.entities[uid].pitch = math.radians(np.random.uniform(-40, -70))
            else:
                # Stall Recovery
                self._teleport_entity(uid, 0, 0, 5000, 0, 180)

    def _setup_navigation(self, blues):
        self._teleport_formation(blues, 0, 0, 6000, np.random.uniform(0, 360), 600)

    def _setup_tail_chase(self, blues, reds, guns_only):
        dist = 2000 if guns_only else 6000
        self._teleport_formation(reds, 0, dist, 6000, 90, 600)
        self._teleport_formation(blues, 0, 0, 6000, 90, 800)

    def _setup_head_on(self, blues, reds, guns_only):
        dist = 6000 if guns_only else 15000
        self._teleport_formation(reds, 0, dist, 7000, 270, 700)
        self._teleport_formation(blues, 0, -dist, 7000, 90, 700)

    def _setup_disadvantage(self, blues, reds, guns_only):
        dist = 1500 if guns_only else 8000
        self._teleport_formation(reds, 0, 0, 6000, 90, 800)
        self._teleport_formation(blues, 0, dist, 6000, 90, 600)


# ================================================
# 2. PARALLEL INFRASTRUCTURE
# ================================================
class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=MAX_PRETRAIN_STEPS):
        super().__init__(env)
        self._max_steps = max_steps
        self._elapsed_steps = 0

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):
        self._elapsed_steps += 1
        obs, reward, term, trunc, info = self.env.step(action, **kwargs)
        if self._elapsed_steps >= self._max_steps:
            trunc = True
        return obs, reward, term, trunc, info


def make_env():
    env = AirCombatEnv()
    env.set_phase(3)  # Train against full physics/enemies for data collection
    return TimeLimitWrapper(env, max_steps=MAX_PRETRAIN_STEPS)


def worker(remote, parent_remote, env_fn_wrapper, seed):
    try:
        import random
        import numpy as np
        import torch
        # Important: Ensure path is correct for worker processes
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        parent_remote.close()

        env = ScenarioWrapper(env_fn_wrapper())
        env.reset(seed=seed)

        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                blue_act, red_act = data
                ob, reward, term, trunc, info = env.step(blue_act, red_actions=red_act)
                if term or trunc:
                    final_stats = {
                        'termination_reason': info.get('termination_reason', 'unknown'),
                        'scenario_mode': env.scenario_type
                    }
                    ob_reset, info_reset = env.reset()
                    info_reset.update(final_stats)
                    ob = ob_reset
                    remote.send((ob, reward, term, trunc, info_reset))
                else:
                    remote.send((ob, reward, term, trunc, info))

            elif cmd == 'reset':
                ob, info = env.reset()
                remote.send((ob, info))

            elif cmd == 'set_mode':
                env.scenario_type = data
                ob, info = env.reset()
                remote.send((ob, info))

            elif cmd == 'close':
                env.close()
                remote.close()
                break
    except Exception as e:
        print(f"❌ WORKER CRASH: {e}")
        raise e


class ParallelMultiAgentEnv:
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        base_seed = int(time.time())
        self.ps = []
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            p = mp.Process(target=worker, args=(work_remote, remote, env_fn, base_seed + i))
            self.ps.append(p)
            p.daemon = True
            p.start()
        for remote in self.work_remotes: remote.close()

    def reset(self):
        for remote in self.remotes: remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def set_mode(self, mode):
        for remote in self.remotes: remote.send(('set_mode', mode))
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

    def close(self):
        for remote in self.remotes: remote.send(('close', None))
        for p in self.ps: p.join()


# ================================================
# 3. DATA COLLECTION & TRAINING
# ================================================

class SequenceDataset(Dataset):
    def __init__(self, obs_chunks, graph_chunks, act_chunks, ret_chunks, mask_chunks):
        self.obs = obs_chunks
        self.graphs = graph_chunks
        self.actions = act_chunks
        self.returns = ret_chunks
        self.masks = mask_chunks

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (self.obs[idx], self.graphs[idx], self.actions[idx],
                self.returns[idx], self.masks[idx])


def collate_sequences(batch):
    obs_list, graph_list_seqs, act_list, ret_list, mask_list = zip(*batch)

    b_obs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    b_act = torch.tensor(np.stack(act_list), dtype=torch.float32)
    b_ret = torch.tensor(np.stack(ret_list), dtype=torch.float32).unsqueeze(-1)
    b_mask = torch.tensor(np.stack(mask_list), dtype=torch.float32).unsqueeze(-1)

    flat_graphs = []
    for seq_graphs in graph_list_seqs:
        for g in seq_graphs:
            if g is None:
                # Empty graph placeholder
                flat_graphs.append(Data(x=torch.zeros(1, Config.NODE_DIM),
                                        edge_index=torch.zeros(2, 0, dtype=torch.long),
                                        edge_attr=torch.zeros(0, Config.EDGE_DIM)))
            else:
                if isinstance(g, dict):
                    flat_graphs.append(Data(x=torch.tensor(g['x'], dtype=torch.float32),
                                            edge_index=torch.tensor(g['edge_index'], dtype=torch.long),
                                            edge_attr=torch.tensor(g['edge_attr'], dtype=torch.float32)))
                else:
                    flat_graphs.append(g)

    b_graphs = Batch.from_data_list(flat_graphs)
    return b_obs, b_graphs, b_act, b_ret, b_mask


def get_bot_actions(bot, obs_batch):
    num_envs, n_agents, _ = obs_batch.shape
    actions = np.zeros((num_envs, n_agents, Config.ACTION_DIM), dtype=np.float32)
    for e in range(num_envs):
        for a in range(n_agents):
            actions[e, a] = bot.get_action(obs_batch[e, a])
    return actions


def collect_data_parallel():
    print(f"🚀 Initializing {Config.NUM_ENVS} Parallel Scenarios...")
    envs = ParallelMultiAgentEnv([make_env for _ in range(Config.NUM_ENVS)])

    print("✅ Workers Started. Initializing HardcodedAce...")
    bot = HardcodedAce()

    # Create data directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    collected_files = []

    for mode, target in PHASES:
        pbar = tqdm(total=target, desc=f"Collecting: {mode.upper()}", unit="step")
        obs, infos = envs.set_mode(mode)

        # Buffers for current phase
        phase_obs = []
        phase_graphs = []
        phase_acts = []
        phase_rets = []
        phase_masks = []

        agent_states = [[{'buffer': {'obs': [], 'graphs': [], 'acts': [], 'rews': []}, 'kills': 0}
                         for _ in range(Config.N_AGENTS)] for _ in range(Config.NUM_ENVS)]

        active_counts = [inf.get('active_blue_count', Config.N_AGENTS) for inf in infos]
        phase_collected_count = 0

        while phase_collected_count < target:
            blue_actions = get_bot_actions(bot, obs)

            current_red_obs = []
            for inf in infos:
                if inf and 'red_obs' in inf:
                    current_red_obs.append(inf['red_obs'])
                else:
                    current_red_obs.append(np.zeros((Config.N_ENEMIES_MAX, Config.OBS_DIM)))
            red_actions = get_bot_actions(bot, np.stack(current_red_obs))

            for i in range(Config.NUM_ENVS):
                step_graph = infos[i]['graph_data'] if (infos[i] and 'graph_data' in infos[i]) else None
                for a in range(Config.N_AGENTS):
                    state = agent_states[i][a]
                    state['buffer']['obs'].append(obs[i, a])
                    state['buffer']['acts'].append(blue_actions[i, a])
                    state['buffer']['graphs'].append(step_graph)

            next_obs, rewards, terms, truncs, next_infos = envs.step(blue_actions, red_actions)
            dones = np.logical_or(terms, truncs)

            for i in range(Config.NUM_ENVS):
                for a in range(Config.N_AGENTS):
                    agent_states[i][a]['buffer']['rews'].append(rewards[i, a])
                    if rewards[i, a] >= 2.5:
                        agent_states[i][a]['kills'] += 1

            for i in range(Config.NUM_ENVS):
                if dones[i]:
                    active_count = active_counts[i]
                    term_reason = next_infos[i].get('termination_reason', 'unknown')
                    nav_success = (mode in ['nav', 'recovery']) and (term_reason != 'crash' and term_reason != 'shot')

                    for a in range(Config.N_AGENTS):
                        if a >= active_count:
                            agent_states[i][a]['buffer'] = {'obs': [], 'graphs': [], 'acts': [], 'rews': []}
                            agent_states[i][a]['kills'] = 0
                            continue

                        state = agent_states[i][a]
                        buf = state['buffer']

                        keep = False
                        crashed = (buf['rews'][-1] <= -4.0)

                        if not crashed:
                            if mode in ['tail_chase', 'head_on', 'disadvantage']:
                                if state['kills'] > 0: keep = True
                            else:
                                if nav_success: keep = True

                        # --- STABILITY FILTER ---
                        if len(buf['obs']) < 20: keep = False

                        if keep:
                            ep_obs = buf['obs']
                            ep_graphs = buf['graphs']
                            ep_acts = buf['acts']
                            g = 0
                            ep_rets = []
                            for r in reversed(buf['rews']):
                                g = r + Config.GAMMA * g
                                ep_rets.insert(0, g)

                            L = len(ep_obs)
                            for start in range(0, L, SEQ_LEN):
                                end = min(start + SEQ_LEN, L)
                                length = end - start
                                if length < SEQ_LEN // 2: continue

                                pad = SEQ_LEN - length
                                if pad > 0:
                                    c_obs = ep_obs[start:end] + [np.zeros_like(ep_obs[0])] * pad
                                    c_graphs = ep_graphs[start:end] + [None] * pad
                                    c_acts = ep_acts[start:end] + [np.zeros_like(ep_acts[0])] * pad
                                    c_rets = ep_rets[start:end] + [0.0] * pad
                                    c_mask = [1.0] * length + [0.0] * pad
                                else:
                                    c_obs = ep_obs[start:end]
                                    c_graphs = ep_graphs[start:end]
                                    c_acts = ep_acts[start:end]
                                    c_rets = ep_rets[start:end]
                                    c_mask = [1.0] * length

                                phase_obs.append(np.array(c_obs))
                                phase_graphs.append(c_graphs)
                                phase_acts.append(np.array(c_acts))
                                phase_rets.append(np.array(c_rets))
                                phase_masks.append(np.array(c_mask))

                                phase_collected_count += length
                                pbar.update(length)

                        state['buffer'] = {'obs': [], 'graphs': [], 'acts': [], 'rews': []}
                        state['kills'] = 0

                    if next_infos[i]:
                        active_counts[i] = next_infos[i].get('active_blue_count', Config.N_AGENTS)

            obs = next_obs
            infos = next_infos

        # Save Phase Data
        file_path = os.path.join(DATA_DIR, f"phase_{mode}.pt")
        print(f"💾 Saving {mode} to {file_path}...")
        torch.save((phase_obs, phase_graphs, phase_acts, phase_rets, phase_masks), file_path)
        collected_files.append((mode, file_path))
        pbar.close()

    envs.close()
    print(f"✅ Collection Complete.")
    return collected_files


def load_or_collect_data():
    """
    Checks for phase files. If missing, runs collection.
    Returns: List of (mode, file_path) tuples.
    """
    existing_files = []
    missing_any = False

    for mode, _ in PHASES:
        path = os.path.join(DATA_DIR, f"phase_{mode}.pt")
        if os.path.exists(path):
            existing_files.append((mode, path))
        else:
            missing_any = True
            break

    if not missing_any and existing_files:
        print(f"\n📂 Found existing phase files in {DATA_DIR}")
        return existing_files

    print("\n📡 Missing datasets. Starting High-Quality Collection...")
    return collect_data_parallel()


def load_phase_data_in_memory(file_path):
    """Loads raw list data into memory without tensor conversion yet to save RAM."""
    print(f"   📂 Reading {file_path}...")
    # Load to CPU
    data = torch.load(file_path, weights_only=False, map_location='cpu')
    return data  # (obs, graphs, acts, rets, masks)


def train_supervised():
    phase_files = load_or_collect_data()

    print(f"Initializing Model on {Config.DEVICE}...")
    model = HybridActorCritic().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=Config.WEIGHT_DECAY)

    # One Cycle LR for superconvergence and stability
    # Steps = Total Epochs * Steps per epoch. We estimate steps per epoch roughly.
    # We will step the scheduler every batch.

    scaler = amp.GradScaler()

    print(f"\n🧠 Starting Unified Mixed Training (All Phases)...")
    print(f"⚡ Batch Size: {BATCH_SIZE}")
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    # 1. Load ALL data into memory (Dictionary keyed by mode)
    # This might take ~4-6GB RAM for 1M steps total.
    database = {}
    total_samples_available = 0
    for mode, fpath in phase_files:
        database[mode] = load_phase_data_in_memory(fpath)
        total_samples_available += len(database[mode][0])

    print(f"📚 Total Expert Sequences Available: {total_samples_available}")

    # Estimate steps for scheduler
    # We will aim for a fixed number of samples per epoch to keep training time predictable
    SAMPLES_PER_EPOCH = 10000

    # CORRECTED CALCULATION:
    # steps_per_epoch is determined by how many times scheduler.step() is called.
    # We call it once every GRAD_ACCUM_STEPS batches.
    # Number of batches = SAMPLES_PER_EPOCH / BATCH_SIZE
    # Number of steps = Number of batches / GRAD_ACCUM_STEPS

    batches_per_epoch = SAMPLES_PER_EPOCH // BATCH_SIZE
    steps_per_epoch = max(1, batches_per_epoch // GRAD_ACCUM_STEPS)

    total_steps = steps_per_epoch * TOTAL_EPOCHS

    print(f"📅 Scheduler Config: {TOTAL_EPOCHS} Epochs, ~{steps_per_epoch} Steps/Epoch, Total Steps: {total_steps}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos'
    )

    for epoch in range(TOTAL_EPOCHS):
        # -------------------------------------------------------------
        # DYNAMIC CURRICULUM MIXING
        # -------------------------------------------------------------
        # Progress: 0.0 -> 1.0
        progress = epoch / (TOTAL_EPOCHS - 1) if TOTAL_EPOCHS > 1 else 1.0

        # SAFETY FLOOR CURRICULUM (Min 20% Safety Data)
        # Start: 30% Rec, 30% Nav, 40% Combat
        # End:   10% Rec, 10% Nav, 80% Combat

        pct_rec = 0.30 - (0.20 * progress)  # 0.30 -> 0.10
        pct_nav = 0.30 - (0.20 * progress)  # 0.30 -> 0.10
        pct_combat = 1.0 - (pct_rec + pct_nav)  # 0.40 -> 0.80

        # Split combat pct among the 3 combat modes equally
        pct_tail = pct_combat / 3.0
        pct_head = pct_combat / 3.0
        pct_disadv = pct_combat / 3.0

        ratios = {
            'recovery': pct_rec,
            'nav': pct_nav,
            'tail_chase': pct_tail,
            'head_on': pct_head,
            'disadvantage': pct_disadv
        }

        # Build the Mixed Dataset for this Epoch
        epoch_obs, epoch_graphs, epoch_acts, epoch_rets, epoch_masks = [], [], [], [], []

        print(f"\nEpoch {epoch + 1}/{TOTAL_EPOCHS} Distribution:")

        for mode in database:
            # How many samples for this mode?
            n_target = int(SAMPLES_PER_EPOCH * ratios[mode])

            # Source data
            src_obs, src_graphs, src_acts, src_rets, src_masks = database[mode]
            n_available = len(src_obs)

            # Sample with replacement if needed, or truncate
            if n_available > 0:
                indices = np.random.choice(n_available, n_target, replace=(n_target > n_available))

                # Append to epoch buffers
                # List comprehension is faster than appending one by one
                epoch_obs.extend([src_obs[i] for i in indices])
                epoch_graphs.extend([src_graphs[i] for i in indices])
                epoch_acts.extend([src_acts[i] for i in indices])
                epoch_rets.extend([src_rets[i] for i in indices])
                epoch_masks.extend([src_masks[i] for i in indices])

                print(f"  - {mode:<12}: {n_target} seqs ({ratios[mode] * 100:.1f}%)")

        # Create DataLoader
        dataset = SequenceDataset(epoch_obs, epoch_graphs, epoch_acts, epoch_rets, epoch_masks)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_sequences, pin_memory=True,
                            num_workers=2)  # Workers help with graph collation

        # Training Loop
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Train Ep {epoch + 1}")

        for i, (b_obs, b_graphs, b_act, b_ret, b_mask) in enumerate(pbar):
            b_obs = b_obs.to(DEVICE)
            b_act = b_act.to(DEVICE)
            b_mask = b_mask.to(DEVICE)
            b_graphs = b_graphs.to(DEVICE)

            # Input Noise
            noise = torch.randn_like(b_obs) * 0.02
            noise[:, :, 0:3] = 0.0  # Don't noise flags
            b_obs_noisy = b_obs + noise

            b_obs_flat = b_obs_noisy.reshape(-1, Config.OBS_DIM)
            b_act_flat = b_act.reshape(-1, Config.ACTION_DIM)
            b_mask_flat = b_mask.reshape(-1)

            # Helper to flatten targets for loss
            b_ret_flat = b_ret.reshape(-1)

            with amp.autocast():
                # 1. ACTOR LOSS
                history_y = model.get_action_history(b_obs_flat)
                actor_loss_sum = 0
                for y_pred in history_y:
                    l_flight = (y_pred[:, :3] - b_act_flat[:, :3]) ** 2
                    target_weap = b_act_flat[:, 3:]
                    bce_weights = 1.0 + (target_weap * 9.0)
                    l_weap = F.binary_cross_entropy_with_logits(
                        y_pred[:, 3:], target_weap, weight=bce_weights, reduction='none'
                    )
                    raw_loss = l_flight.sum(dim=1) + l_weap.sum(dim=1)
                    masked_loss = (raw_loss * b_mask_flat).sum() / (b_mask_flat.sum() + 1e-8)
                    actor_loss_sum += masked_loss

                loss_actor = actor_loss_sum / len(history_y)

                # 2. CRITIC LOSS
                # Predict Value based on the pretraining Graphs + Observations
                # b_graphs is already a batch of (Batch * Seq) graphs
                values = model.get_value(b_graphs, b_obs)

                # Calculate MSE against the recorded Returns (b_ret)
                # b_ret contains the discounted sum of rewards the Expert actually got.
                l_critic_raw = (values.view(-1) - b_ret_flat) ** 2

                # Mask out padding
                loss_critic = (l_critic_raw * b_mask_flat).sum() / (b_mask_flat.sum() + 1e-8)

                # 3. TOTAL LOSS
                # We weight critic loss (usually 0.5 or 1.0)
                loss = loss_actor + (0.5 * loss_critic)

                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix({"L": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"})

        # Save Checkpoint
        save_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'update': epoch
        }
        torch.save(save_data, f"checkpoints/model_pretrained_epoch{epoch}.pt")
        torch.save(save_data, "checkpoints/model_latest.pt")
        torch.save(save_data, "checkpoints/model_pretrained.pt")

        # Cleanup epoch memory
        del dataset, loader, epoch_obs, epoch_graphs, epoch_acts, epoch_rets, epoch_masks
        gc.collect()

    print("✅ Unified Mixed Pretraining Complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_supervised()