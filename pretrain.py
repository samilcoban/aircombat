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
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
import gymnasium as gym

# Add root directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.env import AirCombatEnv
from src.model import HybridActorCritic
from src.bot import HardcodedAce

# === CONFIGURATION ===
DATA_PATH = "data/pretrain_dataset.pt"
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8  # Effective Batch Size = 64
SEQ_LEN = Config.SEQ_LEN
EPOCHS_PER_PHASE = 3
MAX_PRETRAIN_STEPS = 2000


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
        ent.x = x;
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
        ent.d_heading = 0.0;
        ent.d_pitch = 0.0;
        ent.d_roll = 0.0;
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

    master_obs_chunks = []
    master_graph_chunks = []
    master_act_chunks = []
    master_ret_chunks = []
    master_mask_chunks = []

    phase_indices = []  # Stores (start_idx, end_idx, name)

    PHASES = [
        ('recovery', 200_000),
        ('nav', 200_000),
        ('tail_chase', 200_000),
        ('head_on', 200_000),
        ('disadvantage', 200_000)
    ]

    total_target = sum(p[1] for p in PHASES)
    global_collected = 0
    pbar = tqdm(total=total_target, desc="Pretraining Progress", unit="step")

    current_chunk_start = 0

    for mode, target in PHASES:
        pbar.set_description(f"Collecting: {mode.upper()}")
        obs, infos = envs.set_mode(mode)

        agent_states = [[{'buffer': {'obs': [], 'graphs': [], 'acts': [], 'rews': []}, 'kills': 0}
                         for _ in range(Config.N_AGENTS)] for _ in range(Config.NUM_ENVS)]

        active_counts = [inf.get('active_blue_count', Config.N_AGENTS) for inf in infos]
        phase_collected = 0

        while phase_collected < target:
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
                            agent_states[i][a]['buffer'] = {'obs': [], 'graphs': [], 'acts': [], 'rews': []};
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
                        # Discard spawn-die loops
                        if len(buf['obs']) < 20: keep = False

                        if keep:
                            ep_obs = buf['obs']
                            ep_graphs = buf['graphs']
                            ep_acts = buf['acts']
                            g = 0;
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

                                master_obs_chunks.append(np.array(c_obs))
                                master_graph_chunks.append(c_graphs)
                                master_act_chunks.append(np.array(c_acts))
                                master_ret_chunks.append(np.array(c_rets))
                                master_mask_chunks.append(np.array(c_mask))

                                phase_collected += length
                                global_collected += length
                                pbar.update(length)

                        state['buffer'] = {'obs': [], 'graphs': [], 'acts': [], 'rews': []}
                        state['kills'] = 0

                    if next_infos[i]:
                        active_counts[i] = next_infos[i].get('active_blue_count', Config.N_AGENTS)

            obs = next_obs
            infos = next_infos

        chunk_end = len(master_obs_chunks)
        phase_indices.append((current_chunk_start, chunk_end, mode))
        current_chunk_start = chunk_end

    envs.close()
    pbar.close()
    print(f"✅ Collection Complete. Total Chunks: {len(master_obs_chunks)}")
    return (master_obs_chunks, master_graph_chunks, master_act_chunks, master_ret_chunks, master_mask_chunks,
            phase_indices)


def load_or_collect_data():
    if os.path.exists(DATA_PATH):
        print(f"\n📂 Found existing dataset at: {DATA_PATH}")
        print("   Loading data into CPU memory...")
        try:
            data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)

            # Legacy handling
            if len(data) == 5:
                print("⚠️  Legacy dataset detected (no phase indices). Treating as single phase.")
                total_chunks = len(data[0])
                data = list(data)
                data.append([(0, total_chunks, "mixed")])
                data = tuple(data)

            obs, _, _, _, _, _ = data
            print(f"✅ Loaded {len(obs)} chunks.")
            return data
        except Exception as e:
            print(f"❌ Error loading file: {e}. Re-running collection.")

    print("\n📡 No dataset found. Starting High-Quality Collection...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    data = collect_data_parallel()
    print(f"💾 Saving dataset to {DATA_PATH}...")
    torch.save(data, DATA_PATH)
    print("✅ Data saved.")
    return data


def train_supervised():
    data = load_or_collect_data()
    all_obs, all_graphs, all_acts, all_rets, all_masks, phase_indices = data

    print(f"Initializing Model on {Config.DEVICE}...")
    model = HybridActorCritic().to(Config.DEVICE)
    # L2 Regularization (Weight Decay)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=Config.WEIGHT_DECAY)

    start_phase_idx = 0
    if os.path.exists("checkpoints"):
        files = glob.glob("checkpoints/model_pretrained_phase*.pt")
        if files:
            phases = []
            for f in files:
                match = re.search(r'phase(\d+).pt', f)
                if match: phases.append(int(match.group(1)))
            if phases:
                latest_phase = max(phases)
                ckpt_path = f"checkpoints/model_pretrained_phase{latest_phase}.pt"
                print(f"🔄 Resuming from Checkpoint: {ckpt_path}")
                try:
                    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_phase_idx = latest_phase + 1
                except Exception as e:
                    print(f"⚠️ Failed to resume: {e}")

    scaler = amp.GradScaler()

    print(f"\n🧠 Starting Incremental Supervised Training (Deep Supervision + Hybrid Loss)...")
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    for p_idx, (start_idx, end_idx, phase_name) in enumerate(phase_indices):
        if p_idx < start_phase_idx:
            continue

        print(f"\n🎓 TEACHING PHASE {p_idx + 1}/{len(phase_indices)}: {phase_name.upper()}")
        print(f"   Chunks: {start_idx} to {end_idx} (Total: {end_idx - start_idx})")

        phase_obs = all_obs[start_idx:end_idx]
        phase_graphs = all_graphs[start_idx:end_idx]
        phase_acts = all_acts[start_idx:end_idx]
        phase_rets = all_rets[start_idx:end_idx]
        phase_masks = all_masks[start_idx:end_idx]

        dataset = SequenceDataset(phase_obs, phase_graphs, phase_acts, phase_rets, phase_masks)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_sequences, pin_memory=True)

        for epoch in range(EPOCHS_PER_PHASE):
            model.train()
            total_loss = 0
            optimizer.zero_grad()

            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")

            for i, (b_obs, b_graphs, b_act, b_ret, b_mask) in enumerate(pbar):
                b_obs = b_obs.to(DEVICE)
                b_graphs = b_graphs.to(DEVICE)
                b_act = b_act.to(DEVICE)
                b_mask = b_mask.to(DEVICE)

                # --- 1. INPUT NOISE (Robustness) ---
                if model.training:
                    noise = torch.randn_like(b_obs) * 0.02
                    # Mask noise for categorical flags (Indices 0, 1, 2)
                    noise[:, :, 0:3] = 0.0
                    b_obs_noisy = b_obs + noise
                else:
                    b_obs_noisy = b_obs

                # Flatten for TRM input (since TRM expects Batch, Dim)
                # But here we have (Batch, Seq, Dim). We process everything flat.
                b_obs_flat = b_obs_noisy.reshape(-1, Config.OBS_DIM)
                b_act_flat = b_act.reshape(-1, Config.ACTION_DIM)
                b_mask_flat = b_mask.reshape(-1)

                with amp.autocast():
                    # --- 2. DEEP SUPERVISION ---
                    # Get list of actions [y0, y1, y2...]
                    history_y = model.get_action_history(b_obs_flat)

                    # --- 3. HYBRID LOSS CALCULATION ---
                    loss_sum = 0
                    for y_pred in history_y:
                        # A. Flight Controls (MSE)
                        l_flight = (y_pred[:, :3] - b_act_flat[:, :3]) ** 2

                        # B. Weapons (Weighted BCE)
                        # Punish missing a shot (target=1) 10x more than firing at nothing
                        target_weap = b_act_flat[:, 3:]
                        bce_weights = 1.0 + (target_weap * 9.0)
                        l_weap = F.binary_cross_entropy_with_logits(
                            y_pred[:, 3:], target_weap, weight=bce_weights, reduction='none'
                        )

                        # Sum dims
                        raw_loss = l_flight.sum(dim=1) + l_weap.sum(dim=1)

                        # Mask Padding
                        masked_loss = (raw_loss * b_mask_flat).sum() / (b_mask_flat.sum() + 1e-8)
                        loss_sum += masked_loss

                    loss = loss_sum / len(history_y)  # Average deep supervision steps
                    loss = loss / GRAD_ACCUM_STEPS

                scaler.scale(loss).backward()

                if (i + 1) % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * GRAD_ACCUM_STEPS
                pbar.set_postfix({"Loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})

        # Save
        save_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'update': 0
        }
        torch.save(save_data, f"checkpoints/model_pretrained_phase{p_idx}.pt")
        torch.save(save_data, "checkpoints/model_latest.pt")
        torch.save(save_data, "checkpoints/model_pretrained.pt")

    print("✅ Incremental Pretraining Complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_supervised()