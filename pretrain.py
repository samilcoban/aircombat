# ================================================
# FILE: pretrain.py
# ================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp  # <--- VRAM FIX
import numpy as np
import math
import os
import sys
import time
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
PRETRAIN_STEPS = 200_000

# --- VRAM OPTIMIZATION START ---
# Effective Batch Size = 8 * 4 = 32. Fits in 4GB VRAM.
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
# --- VRAM OPTIMIZATION END ---

SEQ_LEN = Config.SEQ_LEN
EPOCHS = 10
LR = 3e-4
DEVICE = Config.DEVICE
MAX_PRETRAIN_STEPS = 600


# ================================================
# 1. SCENARIO WRAPPER (The Director)
# ================================================
class ScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scenario_type = "random"

    # --- CRITICAL FIX: Allow red_actions to pass through ---
    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 1. Randomize Loadout
        guns_only = (np.random.rand() < 0.5)

        if guns_only:
            for uid in self.env.unwrapped.blue_ids:
                if uid in self.env.unwrapped.core.entities:
                    self.env.unwrapped.core.entities[uid].ammo = 0
            for uid in self.env.unwrapped.red_ids:
                if uid in self.env.unwrapped.core.entities:
                    self.env.unwrapped.core.entities[uid].ammo = 0

        # 2. Determine Scenario
        rand = np.random.rand()
        scenario_active = False

        if rand < 0.30:
            self.scenario_type = "tail_chase"
            self._setup_tail_chase(guns_only)
            scenario_active = True
        elif rand < 0.60:
            self.scenario_type = "head_on"
            self._setup_head_on(guns_only)
            scenario_active = True
        elif rand < 0.80:
            self.scenario_type = "disadvantage"
            self._setup_disadvantage()
            scenario_active = True
        else:
            self.scenario_type = "random"

        # 3. Cache Coherency Fix
        if scenario_active or guns_only:
            self.env.unwrapped.core.update_spatial_cache()
            self.env.unwrapped._compute_frame_data()
            obs = self.env.unwrapped._get_all_blue_obs()
            info["red_obs"] = self.env.unwrapped._get_all_red_obs()
            info["graph_data"] = self.env.unwrapped._get_graph_state()

        return obs, info

    def _teleport_entity(self, uid, x, y, alt, heading, speed):
        """Helper to move an entity to a specific state."""
        if uid not in self.env.unwrapped.core.entities: return
        ent = self.env.unwrapped.core.entities[uid]

        if hasattr(self.env.unwrapped.core, 'dist_matrix'):  # Flat
            ent.x = x
            ent.y = y
        else:
            # Geodetic approx placeholder
            pass

        ent.alt = alt
        ent.heading = math.radians(heading)
        ent.speed = speed
        ent.roll = 0.0
        ent.pitch = 0.0

    def _setup_tail_chase(self, guns_only):
        if not self.env.unwrapped.blue_ids or not self.env.unwrapped.red_ids: return
        bid = self.env.unwrapped.blue_ids[0]
        rid = self.env.unwrapped.red_ids[0]
        dist = 2000 if guns_only else 5000
        # Increased alt slightly to prevent ground crashes
        self._teleport_entity(rid, 0, dist, 6000, 90, 600)
        self._teleport_entity(bid, 0, 0, 6000, 90, 800)

    def _setup_head_on(self, guns_only):
        if not self.env.unwrapped.blue_ids or not self.env.unwrapped.red_ids: return
        bid = self.env.unwrapped.blue_ids[0]
        rid = self.env.unwrapped.red_ids[0]
        half_dist = 5000 if guns_only else 15000
        self._teleport_entity(rid, 0, half_dist, 7000, 270, 700)
        self._teleport_entity(bid, 0, -half_dist, 7000, 90, 700)

    def _setup_disadvantage(self):
        if not self.env.unwrapped.blue_ids or not self.env.unwrapped.red_ids: return
        bid = self.env.unwrapped.blue_ids[0]
        rid = self.env.unwrapped.red_ids[0]
        self._teleport_entity(rid, 0, 0, 6000, 90, 800)
        self._teleport_entity(bid, 0, 3000, 6000, 90, 600)


# ================================================
# 2. PARALLEL INFRASTRUCTURE
# ================================================
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
                    # --- FIX: CAPTURE STATS BEFORE RESET ---
                    # The original code reset immediately, losing 'stat_kills'.
                    final_stats = {
                        'stat_kills': info.get('stat_kills', 0),
                        'stat_missiles_fired': info.get('stat_missiles_fired', 0)
                    }
                    ob_reset, info_reset = env.reset()
                    # Inject old stats into new info so main process sees them
                    info_reset.update(final_stats)

                    # Carry over persistent info
                    info_reset['red_obs'] = info_reset.get('red_obs')
                    info_reset['graph_data'] = info_reset.get('graph_data')
                    ob = ob_reset
                    # Return the RESET info (which now contains the kills)
                    remote.send((ob, reward, term, trunc, info_reset))
                else:
                    remote.send((ob, reward, term, trunc, info))

            elif cmd == 'reset':
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
    # CRITICAL: Expert needs Phase 3 physics
    env.set_phase(3)
    return TimeLimitWrapper(env)


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
    b_mask = torch.tensor(np.stack(mask_list), dtype=torch.float32).unsqueeze(-1)  # Added unsqueeze for mask

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

    print("✅ Workers Started. Initializing Hardcoded Ace...")
    bot = HardcodedAce()

    master_obs_chunks = []
    master_graph_chunks = []
    master_act_chunks = []
    master_ret_chunks = []
    master_mask_chunks = []

    env_buffers = [{'obs': [], 'graphs': [], 'acts': [], 'rews': []} for _ in range(Config.NUM_ENVS)]

    obs, infos = envs.reset()

    valid_steps_collected = 0
    total_simulated = 0
    discarded_episodes = 0

    stats_kills = 0
    stats_fired = 0
    debug_returns = []

    pbar = tqdm(total=PRETRAIN_STEPS, desc="Valid Steps", unit="step")

    while valid_steps_collected < PRETRAIN_STEPS:
        blue_actions = get_bot_actions(bot, obs)

        current_red_obs = []
        for inf in infos:
            if inf and 'red_obs' in inf:
                current_red_obs.append(inf['red_obs'])
            else:
                current_red_obs.append(np.zeros((Config.N_AGENTS, Config.OBS_DIM)))
        red_obs_np = np.stack(current_red_obs)
        red_actions = get_bot_actions(bot, red_obs_np)

        for i in range(Config.NUM_ENVS):
            env_buffers[i]['obs'].append(obs[i, 0])
            env_buffers[i]['acts'].append(blue_actions[i, 0])
            if infos[i] and 'graph_data' in infos[i]:
                env_buffers[i]['graphs'].append(infos[i]['graph_data'])
            else:
                env_buffers[i]['graphs'].append(None)

        next_obs, rewards, terms, truncs, next_infos = envs.step(blue_actions, red_actions)
        dones = np.logical_or(terms, truncs)

        total_simulated += Config.NUM_ENVS

        # --- FIX: LOGGING CORRECTLY FROM WORKER ---
        for inf in next_infos:
            if inf:
                stats_kills += inf.get('stat_kills', 0)
                stats_fired += inf.get('stat_missiles_fired', 0)

        for i in range(Config.NUM_ENVS):
            env_buffers[i]['rews'].append(rewards[i, 0])

            if np.any(dones[i]):
                total_return = sum(env_buffers[i]['rews'])
                debug_returns.append(total_return)

                # --- FIX: NEW REWARD FILTER ---
                # Win(+3) + Kill(+4) = +7. Costs reduce this.
                # Threshold > 2.0 ensures we only keep Wins.
                if total_return > 2.0:
                    ep_obs = env_buffers[i]['obs']
                    ep_graphs = env_buffers[i]['graphs']
                    ep_acts = env_buffers[i]['acts']

                    g = 0
                    ep_rets = []
                    for r in reversed(env_buffers[i]['rews']):
                        g = r + Config.GAMMA * g
                        ep_rets.insert(0, g)

                    L = len(ep_obs)
                    for start in range(0, L, SEQ_LEN):
                        end = min(start + SEQ_LEN, L)
                        length = end - start

                        if length < SEQ_LEN // 2: continue

                        pad_len = SEQ_LEN - length
                        if pad_len > 0:
                            c_obs = ep_obs[start:end] + [np.zeros_like(ep_obs[0])] * pad_len
                            c_graphs = ep_graphs[start:end] + [None] * pad_len
                            c_acts = ep_acts[start:end] + [np.zeros_like(ep_acts[0])] * pad_len
                            c_rets = ep_rets[start:end] + [0.0] * pad_len
                            c_mask = [1.0] * length + [0.0] * pad_len
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

                        valid_steps_collected += length
                        pbar.update(length)
                else:
                    discarded_episodes += 1

                env_buffers[i] = {'obs': [], 'graphs': [], 'acts': [], 'rews': []}

        obs = next_obs
        infos = next_infos

        avg_ret = np.mean(debug_returns[-100:]) if debug_returns else 0.0
        pbar.set_postfix({
            "Drop": discarded_episodes,
            "AvgR": f"{avg_ret:.2f}",
            "Kills": stats_kills,
            "Fire": stats_fired,
            "Chunks": len(master_obs_chunks)
        })

    envs.close()
    pbar.close()
    print(f"✅ Collection Complete. Total Chunks: {len(master_obs_chunks)}")
    return master_obs_chunks, master_graph_chunks, master_act_chunks, master_ret_chunks, master_mask_chunks


def train_supervised():
    data = collect_data_parallel()
    if not data[0]:
        print("❌ No valid episodes! Check Bot logic or rewards.")
        return

    dataset = SequenceDataset(*data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sequences)

    print(f"Initializing Model on {DEVICE}...")
    model = HybridActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- VRAM OPTIMIZATION ---
    scaler = amp.GradScaler()

    actor_criterion = nn.MSELoss(reduction='none')
    critic_criterion = nn.MSELoss(reduction='none')

    print("\n🧠 Starting Supervised Training (Sequence Mode) with Mixed Precision...")
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # Initialize gradients

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i, (b_obs, b_graphs, b_act, b_ret, b_mask) in enumerate(pbar):
            b_obs = b_obs.to(DEVICE)
            b_graphs = b_graphs.to(DEVICE)
            b_act = b_act.to(DEVICE)
            b_ret = b_ret.to(DEVICE)
            b_mask = b_mask.to(DEVICE)  # already unsqueezed in collate

            batch_dim = b_obs.shape[0]
            gru_state = torch.zeros(1, batch_dim, Config.D_MODEL).to(DEVICE)

            # --- MIXED PRECISION FORWARD ---
            with amp.autocast():
                pred_act, _, _, pred_val, _ = model.get_action_and_value(
                    b_obs, graph_data=b_graphs, action=None, gru_state=gru_state
                )

                loss_a = (actor_criterion(pred_act, b_act) * b_mask).sum() / (b_mask.sum() + 1e-8)
                loss_c = (critic_criterion(pred_val, b_ret) * b_mask).sum() / (b_mask.sum() + 1e-8)

                # Divide by Accumulation Steps for gradient normalization
                loss = (loss_a + 0.5 * loss_c) / GRAD_ACCUM_STEPS

            # --- MIXED PRECISION BACKWARD ---
            scaler.scale(loss).backward()

            # --- GRADIENT ACCUMULATION STEP ---
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS  # Scale back for display
            pbar.set_postfix({"L_Act": f"{loss_a.item():.4f}", "L_Crit": f"{loss_c.item():.4f}"})

        print(f"  Avg Loss: {total_loss / len(loader):.4f}")

        save_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'update': 0
        }
        torch.save(save_data, f"checkpoints/model_pretrained_ep{epoch}.pt")

    torch.save(save_data, "checkpoints/model_latest.pt")
    torch.save(save_data, "checkpoints/model_pretrained.pt")
    print("✅ Pretraining Complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_supervised()