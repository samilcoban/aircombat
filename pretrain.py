# ==============================================================================
# FILE: pretrain.py
# ==============================================================================
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
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
SEQ_LEN = Config.SEQ_LEN
TOTAL_EPOCHS = 20
MAX_PRETRAIN_STEPS = 2000
DEVICE = Config.DEVICE
LR = 3e-4

# Reduced count for Disadvantage since it's harder to collect, but usually
# we want equal amounts. Let's keep it equal but fix the collection speed.
PHASES = [
    ('recovery', 200_000),
    ('nav', 200_000),
    ('tail_chase', 200_000),
    ('head_on', 200_000),
    ('disadvantage', 200_000)
]


# ==============================================================================
# 0. VICTIM BOT (PACIFIST)
# ==============================================================================
class VictimAce(HardcodedAce):
    """
    A Target Drone version of the Ace.
    1. Blind to Missiles (Won't Notch/Defend).
    2. Pacifist (Won't Fire).

    This ensures Blue has time to turn the tables in a Disadvantage scenario
    without getting shot in the back immediately.
    """

    def get_action(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        # 1. Blindness (Mask Missiles)
        blind_obs = obs.copy()
        track_data = blind_obs[Config.NODE_DIM:]
        num_tracks = len(track_data) // Config.EDGE_DIM

        for i in range(num_tracks):
            start = i * Config.EDGE_DIM
            type_idx = start + 9
            # If missile, zero out range
            if track_data[type_idx] < -0.5:
                track_data[start] = 0.0

                # 2. Get Standard Action
        act = super().get_action(blind_obs)

        # 3. Pacifism (Disable Weapons)
        # Act format: [Roll, G, Throttle, Fire, CM]
        act[3] = 0.0  # Never Fire

        return act


# ================================================
# 1. SCENARIO WRAPPER
# ================================================
class ScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scenario_type = "combat"
        self.step_counter = 0

    def step(self, action, **kwargs):
        obs, reward, term, trunc, info = self.env.step(action, **kwargs)
        self.step_counter += 1

        if self.scenario_type in ["nav", "recovery"]:
            if self.step_counter >= 300 and not term:
                trunc = True

        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        self.step_counter = 0
        obs, info = self.env.reset(**kwargs)

        n_blue = np.random.randint(1, Config.N_AGENTS + 1)
        n_red = np.random.randint(1, Config.N_ENEMIES_MAX + 1)

        active_blue = self.env.unwrapped.blue_ids[:n_blue]
        active_red = self.env.unwrapped.red_ids[:n_red]

        inactive_blue = self.env.unwrapped.blue_ids[n_blue:]
        inactive_red = self.env.unwrapped.red_ids[n_red:]
        self._teleport_formation(inactive_blue, -200000, -200000, 10000, 0, 0)
        self._teleport_formation(inactive_red, 200000, 200000, 10000, 0, 0)

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
            # NOTE: We allow missiles here so Blue can turn the tables
            self._setup_disadvantage(active_blue, active_red, guns_only)

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
        # FIX: Ensure Visual Contact
        # If dist > 5000, Blue cannot see Red behind it, so Blue flies straight and dies.
        # We must keep distance < 5000 for Blue to "Sense" the threat visually.

        dist = 1500 if guns_only else 4500  # Was 8000

        # Red: Behind
        self._teleport_formation(reds, 0, 0, 6000, 90, 600)

        # Blue: Ahead (Distance 'dist' along Y)
        # Give Blue slight offset/angle so it isn't pure 6 o'clock
        self._teleport_formation(blues, 500, dist, 6000, 110, 700)


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

    print("✅ Workers Started.")
    print("   -> Blue Team: HardcodedAce (Expert)")
    print("   -> Red Team:  VictimAce (Pacifist Target Drone)")

    bot = HardcodedAce()
    victim = VictimAce()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    collected_files = []

    for mode, target in PHASES:
        # Resume Check
        file_path = os.path.join(DATA_DIR, f"phase_{mode}.pt")
        if os.path.exists(file_path):
            print(f"⏩ Skipping {mode.upper()} (File exists: {file_path})")
            collected_files.append((mode, file_path))
            continue

        pbar = tqdm(total=target, desc=f"Collecting: {mode.upper()}", unit="step")
        obs, infos = envs.set_mode(mode)

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

            # Use VictimAce for Red (Will NOT fire)
            red_actions = get_bot_actions(victim, np.stack(current_red_obs))

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

        print(f"💾 Saving {mode} to {file_path}...")
        torch.save((phase_obs, phase_graphs, phase_acts, phase_rets, phase_masks), file_path)
        collected_files.append((mode, file_path))
        pbar.close()

    envs.close()
    print(f"✅ Collection Complete.")
    return collected_files


def load_or_collect_data():
    """
    Checks for phase files.
    OPTIMIZATION: Checks existance BEFORE spinning up workers.
    """
    collected_files = []
    missing_any = False

    # 1. Pre-check loop
    for mode, _ in PHASES:
        path = os.path.join(DATA_DIR, f"phase_{mode}.pt")
        if os.path.exists(path):
            collected_files.append((mode, path))
        else:
            missing_any = True
            # Don't break yet, we want to know exactly what we have,
            # but we know we need to run collection.

    # 2. If everything exists, return immediately (0 seconds)
    if not missing_any:
        print(f"\n📂 Found ALL existing phase files in {DATA_DIR}. Skipping worker init.")
        return collected_files

    # 3. If missing something, NOW we spin up the workers
    print("\n📡 Missing some datasets. Starting High-Quality Collection...")
    return collect_data_parallel()


def load_phase_data_in_memory(file_path):
    print(f"   📂 Reading {file_path}...")
    data = torch.load(file_path, weights_only=False, map_location='cpu')
    return data


def train_supervised():
    phase_files = load_or_collect_data()

    print(f"Initializing Model on {Config.DEVICE}...")
    model = HybridActorCritic().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=Config.WEIGHT_DECAY)
    scaler = amp.GradScaler()

    print(f"\n🧠 Starting Unified Mixed Training (All Phases)...")
    print(f"⚡ Batch Size: {BATCH_SIZE}")
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    database = {}
    total_samples_available = 0
    for mode, fpath in phase_files:
        database[mode] = load_phase_data_in_memory(fpath)
        total_samples_available += len(database[mode][0])

    print(f"📚 Total Expert Sequences Available: {total_samples_available}")

    SAMPLES_PER_EPOCH = 10000
    batches_per_epoch = SAMPLES_PER_EPOCH // BATCH_SIZE
    steps_per_epoch = max(1, batches_per_epoch // GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * TOTAL_EPOCHS

    print(f"📅 Scheduler Config: {TOTAL_EPOCHS} Epochs, ~{steps_per_epoch} Steps/Epoch, Total Steps: {total_steps}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    for epoch in range(TOTAL_EPOCHS):
        progress = epoch / (TOTAL_EPOCHS - 1) if TOTAL_EPOCHS > 1 else 1.0

        pct_rec = 0.30 - (0.20 * progress)
        pct_nav = 0.30 - (0.20 * progress)
        pct_combat = 1.0 - (pct_rec + pct_nav)

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

        epoch_obs, epoch_graphs, epoch_acts, epoch_rets, epoch_masks = [], [], [], [], []

        print(f"\nEpoch {epoch + 1}/{TOTAL_EPOCHS} Distribution:")

        for mode in database:
            n_target = int(SAMPLES_PER_EPOCH * ratios[mode])
            src_obs, src_graphs, src_acts, src_rets, src_masks = database[mode]
            n_available = len(src_obs)

            if n_available > 0:
                indices = np.random.choice(n_available, n_target, replace=(n_target > n_available))
                epoch_obs.extend([src_obs[i] for i in indices])
                epoch_graphs.extend([src_graphs[i] for i in indices])
                epoch_acts.extend([src_acts[i] for i in indices])
                epoch_rets.extend([src_rets[i] for i in indices])
                epoch_masks.extend([src_masks[i] for i in indices])
                print(f"  - {mode:<12}: {n_target} seqs ({ratios[mode] * 100:.1f}%)")

        dataset = SequenceDataset(epoch_obs, epoch_graphs, epoch_acts, epoch_rets, epoch_masks)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_sequences, pin_memory=True,
                            num_workers=2)

        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Train Ep {epoch + 1}")

        for i, (b_obs, b_graphs, b_act, b_ret, b_mask) in enumerate(pbar):
            b_obs = b_obs.to(DEVICE)
            b_act = b_act.to(DEVICE)
            b_mask = b_mask.to(DEVICE)
            b_graphs = b_graphs.to(DEVICE)
            b_ret = b_ret.to(DEVICE)

            # Input Noise
            noise = torch.randn_like(b_obs) * 0.02
            noise[:, :, 0:3] = 0.0
            b_obs_noisy = b_obs + noise

            b_act_flat = b_act.reshape(-1, Config.ACTION_DIM)
            b_mask_flat = b_mask.reshape(-1)
            b_ret_flat = b_ret.reshape(-1)

            with amp.autocast():
                # 1. ACTOR LOSS
                # FIX: Pass 3D tensor [Batch, Seq, Dim] so Model unrolls GRU statefully
                history_y = model.get_action_history(b_obs_noisy)

                actor_loss_sum = 0
                for y_pred in history_y:
                    # FIX: Flatten output [Batch, Seq, Act] -> [Batch*Seq, Act] to match targets
                    y_pred_flat = y_pred.reshape(-1, Config.ACTION_DIM)

                    # Instead of MSE (Distance), we use NLL (Probability).
                    # This teaches the model to match the Expert's Distribution, not just the Mean.

                    # 1. Get the exact distribution PPO will see (Parity)
                    dist = model.get_policy_distribution(y_pred_flat)

                    # 2. Calculate Log Probability of Expert Actions
                    log_prob = dist.log_prob(b_act_flat).sum(dim=-1)

                    # 3. Calculate Entropy (To prevent collapse on deterministic expert)
                    entropy = dist.entropy().sum(dim=-1)

                    # 4. Masking (Ignore padding)
                    # Denominator +1e-8 prevents div/0
                    active_count = b_mask_flat.sum() + 1e-8
                    masked_nll = (-log_prob * b_mask_flat).sum() / active_count
                    masked_entropy = (entropy * b_mask_flat).sum() / active_count

                    # 5. Total Step Loss
                    # Minimize NLL (Fit Expert) - Maximize Entropy (Stay Humble)
                    step_loss = masked_nll - (Config.ENT_COEF * masked_entropy)

                    actor_loss_sum += step_loss

                loss_actor = actor_loss_sum / len(history_y)

                # 2. CRITIC LOSS
                # Note: get_value handles 3D inputs internally by flattening for GNN query, so b_obs (3D) is fine here
                values = model.get_value(b_graphs, b_obs)
                l_critic_raw = (values.view(-1) - b_ret_flat) ** 2
                loss_critic = (l_critic_raw * b_mask_flat).sum() / (b_mask_flat.sum() + 1e-8)

                # 3. TOTAL LOSS
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


                # Calculate actual loss for this batch (undoing the gradient accumulation division)
                current_batch_loss = loss.item() * GRAD_ACCUM_STEPS
                total_loss += current_batch_loss

                # Calculate running average for this epoch
                avg_loss_so_far = total_loss / (i + 1)

                # Update progress bar with Average Loss
                pbar.set_postfix({"L_avg": f"{avg_loss_so_far:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"})

        save_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'update': epoch
        }
        torch.save(save_data, f"checkpoints/model_pretrained_epoch{epoch}.pt")
        torch.save(save_data, "checkpoints/model_latest.pt")
        torch.save(save_data, "checkpoints/model_pretrained.pt")

        del dataset, loader, epoch_obs, epoch_graphs, epoch_acts, epoch_rets, epoch_masks
        gc.collect()

    print("✅ Unified Mixed Pretraining Complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_supervised()