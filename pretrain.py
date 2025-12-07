# ================================================
# FILE: pretrain.py
# ================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import time
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
import gymnasium as gym

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.env import AirCombatEnv
from src.model import HybridActorCritic
from src.bot import HardcodedAce

# === CONFIGURATION ===
PRETRAIN_STEPS = 500_000
BATCH_SIZE = 128
EPOCHS = 10
LR = 3e-4
DEVICE = Config.DEVICE
MAX_PRETRAIN_STEPS = 400


# ================================================
# PRETRAIN BOT - Safe Flying + Safe Combat
# ================================================
class PretrainBot:
    """
    Bot for pretraining that teaches safe flying AND combat.
    Removes aggressive missile evasion to avoid teaching the agent how to die.
    """

    def __init__(self):
        self.cfg = Config

    def get_action(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        # Parse Ego State (Unified Node: NODE_DIM = 16)
        # Indices: [Exist, Team, Type, X, Y, Alt, CosH, SinH, SinP, SinR, Spd, G, Fuel, Ammo, Chaff, CM]
        ego_vec = obs[0:self.cfg.NODE_DIM]

        # Check existence
        if ego_vec[0] < 0.5:
            return np.array([0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)

        ego_alt_norm = ego_vec[5]
        ego_speed = ego_vec[10]
        ego_ammo = ego_vec[13]

        # === SPEED RECOVERY (Prevent Stalling) ===
        if ego_speed < 0.25:
            # Roll level, push nose down, max throttle to recover speed
            return np.array([0.0, -0.5, 1.0, 0.0, 0.0], dtype=np.float32)

        # Recover roll angle
        current_roll_rad = math.asin(np.clip(ego_vec[9], -1, 1))

        # === PARSE TRACKS (for enemy detection) ===
        track_data = obs[self.cfg.NODE_DIM:]
        num_tracks = len(track_data) // self.cfg.EDGE_DIM

        enemies = []
        for i in range(num_tracks):
            start = i * self.cfg.EDGE_DIM
            end = start + self.cfg.EDGE_DIM
            vec = track_data[start:end]
            
            if vec[0] < 1e-5: continue  # Empty track
            
            is_missile = (vec[9] < -0.5)
            is_enemy = (vec[10] < -0.5)
            
            # We SKIP missiles - no aggressive evasion to avoid teaching dying
            if is_missile:
                continue
            
            if is_enemy and not is_missile:
                # Parse target geometry
                lx, ly, lz = vec[1], vec[2], vec[3]
                dist_flat = math.hypot(lx, ly)
                az_rad = math.atan2(ly, lx) if dist_flat > 1e-6 else 0.0
                az_deg = math.degrees(az_rad)
                
                real_z = lz * 10000.0
                real_dist = vec[0] * 60000.0
                el_sin = np.clip(real_z / (real_dist + 1e-5), -1, 1)
                
                enemies.append({
                    'range_norm': vec[0],
                    'azimuth_deg': az_deg,
                    'elevation_sin': el_sin,
                    'closure': vec[7]
                })

        # === SAFE COMBAT: Engage Enemies ===
        if enemies:
            # Pick closest target with good aspect
            target = min(enemies, key=lambda e: abs(e['azimuth_deg']) + e['range_norm'] * 100)
            ata = target['azimuth_deg']
            
            # Fire control
            fire = 0.0
            if abs(ata) < 15.0 and target['range_norm'] < 0.5 and ego_ammo > 0:
                if np.random.rand() < 0.15:
                    fire = 1.0
            
            # Gentle maneuvering (no aggressive turns)
            roll_cmd = np.clip(ata / 45.0, -1.0, 1.0)
            g_cmd = np.clip(abs(ata) / 30.0, 0.0, 0.8)  # Max 0.8 G (was 1.0)
            
            # Elevation adjustment
            if target['elevation_sin'] > 0.1:
                g_cmd += 0.3
            elif target['elevation_sin'] < -0.1:
                g_cmd -= 0.2
            
            return np.array([roll_cmd, np.clip(g_cmd, -0.2, 0.8), 1.0, fire, 0.0], dtype=np.float32)

        # === SAFE PATROL FLIGHT ===
        roll_cmd = np.clip(-current_roll_rad * 2.0, -1.0, 1.0)
        target_alt = 0.4
        alt_err = target_alt - ego_alt_norm
        g_cmd = np.clip(alt_err * 5.0, -0.2, 0.5)
        
        return np.array([roll_cmd, g_cmd, 0.8, 0.0, 0.0], dtype=np.float32)


# --- Parallel Infrastructure ---
def worker(remote, parent_remote, env_fn_wrapper, seed):
    # Move imports inside to catch import errors in workers
    try:
        import random
        import numpy as np
        import torch
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        parent_remote.close()
        env = env_fn_wrapper()
        env.reset(seed=seed)

        while True:
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

    def call(self, method_name, *args, **kwargs):
        for remote in self.remotes: remote.send(('call', (method_name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]

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

    def set_phase(self, p): self.env.unwrapped.set_phase(p)


def make_env():
    return TimeLimitWrapper(AirCombatEnv())


# --- DATASET & HELPERS ---

class ExpertDataset(Dataset):
    def __init__(self, obs, graphs, actions, returns):
        self.obs = obs
        self.graphs = graphs
        self.actions = actions
        self.returns = returns

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.graphs[idx], self.actions[idx], self.returns[idx]


def collate_fn(batch):
    obs_list, graph_list, act_list, ret_list = zip(*batch)
    b_obs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    b_act = torch.tensor(np.stack(act_list), dtype=torch.float32)
    b_ret = torch.tensor(np.stack(ret_list), dtype=torch.float32).view(-1, 1)

    clean_graphs = []
    for g in graph_list:
        if g is None:
            clean_graphs.append(Data(x=torch.zeros(1, Config.NODE_DIM),
                                     edge_index=torch.zeros(2, 0, dtype=torch.long),
                                     edge_attr=torch.zeros(0, Config.EDGE_DIM)))
        else:
            if isinstance(g, dict):
                clean_graphs.append(Data(x=torch.tensor(g['x']),
                                         edge_index=torch.tensor(g['edge_index'], dtype=torch.long),
                                         edge_attr=torch.tensor(g['edge_attr'])))
            else:
                clean_graphs.append(g)

    b_graphs = Batch.from_data_list(clean_graphs)
    return b_obs, b_graphs, b_act, b_ret


def get_bot_actions(bot, obs_batch):
    num_envs, n_agents, _ = obs_batch.shape
    actions = np.zeros((num_envs, n_agents, Config.ACTION_DIM), dtype=np.float32)
    for e in range(num_envs):
        for a in range(n_agents):
            actions[e, a] = bot.get_action(obs_batch[e, a])
    return actions


def collect_data_parallel():
    print(f"🚀 Initializing {Config.NUM_ENVS} Parallel Environments...")

    envs = ParallelMultiAgentEnv([make_env for _ in range(Config.NUM_ENVS)])
    envs.call("set_phase", 2)

    print("✅ Workers Started. Initializing Bot...")
    bot = PretrainBot()

    master_obs = []
    master_graphs = []
    master_actions = []
    master_returns = []

    env_buffers = [{'obs': [], 'graphs': [], 'acts': [], 'rews': []} for _ in range(Config.NUM_ENVS)]

    obs, infos = envs.reset()

    print(f"🎥 Starting Data Collection (Target: {PRETRAIN_STEPS} Valid Steps)...")

    # Progress Bar tracks VALID steps, but we use postfix for diagnostics
    pbar = tqdm(total=PRETRAIN_STEPS, desc="Valid Data", unit="step")

    valid_steps_collected = 0
    total_simulated = 0
    discarded_episodes = 0

    while valid_steps_collected < PRETRAIN_STEPS:
        # 1. Bot Inference
        blue_actions = get_bot_actions(bot, obs)

        current_red_obs = []
        for inf in infos:
            if inf and 'red_obs' in inf:
                current_red_obs.append(inf['red_obs'])
            else:
                current_red_obs.append(np.zeros((Config.N_AGENTS, Config.OBS_DIM)))
        red_obs_np = np.stack(current_red_obs)
        red_actions = get_bot_actions(bot, red_obs_np)

        # 2. Buffer Storage
        for i in range(Config.NUM_ENVS):
            env_buffers[i]['obs'].append(obs[i, 0])
            env_buffers[i]['acts'].append(blue_actions[i, 0])
            if infos[i] and 'graph_data' in infos[i]:
                env_buffers[i]['graphs'].append(infos[i]['graph_data'])
            else:
                env_buffers[i]['graphs'].append(None)

        # 3. Physics Step
        next_obs, rewards, terms, truncs, next_infos = envs.step(blue_actions, red_actions)
        dones = np.logical_or(terms, truncs)

        # Increment raw counter
        total_simulated += Config.NUM_ENVS

        # 4. Processing
        for i in range(Config.NUM_ENVS):
            env_buffers[i]['rews'].append(rewards[i, 0])

            is_done = np.any(dones[i]) if isinstance(dones[i], np.ndarray) else dones[i]

            if is_done:
                ep_len = len(env_buffers[i]['obs'])
                total_return = sum(env_buffers[i]['rews'])

                # --- FIX: Relax filter to capture "Stalling but Surviving" episodes ---
                # Old: -5.0 (Too strict, requires near-perfect energy management)
                # New: -30.0 (Allows for continuous stalling penalty of -20.0)
                if total_return > -30.0:
                    master_obs.extend(env_buffers[i]['obs'])
                    master_actions.extend(env_buffers[i]['acts'])
                    master_graphs.extend(env_buffers[i]['graphs'])

                    # Returns-to-Go
                    g = 0
                    returns = []
                    for r in reversed(env_buffers[i]['rews']):
                        g = r + 0.99 * g
                        returns.insert(0, g)
                    master_returns.extend(returns)

                    valid_steps_collected += ep_len
                    pbar.update(ep_len)
                else:
                    discarded_episodes += 1

                # Reset Buffer
                env_buffers[i] = {'obs': [], 'graphs': [], 'acts': [], 'rews': []}

        # Update Postfix every step loop so you see it moving
        pbar.set_postfix({
            "Simulated": total_simulated,
            "Discarded": discarded_episodes,
            "LastRew": f"{np.mean(rewards):.3f}"
        })

        obs = next_obs
        infos = next_infos

    envs.close()
    pbar.close()
    print(f"✅ Collection Complete. Total Valid Steps: {len(master_obs)}")
    return master_obs, master_graphs, master_actions, master_returns


def train_supervised():
    obs, graphs, actions, returns = collect_data_parallel()

    if not obs:
        print("❌ No valid episodes! Check Bot logic or rewards.")
        return

    dataset = ExpertDataset(obs, graphs, actions, returns)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print("Initializing Model...")
    model = HybridActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    actor_criterion = nn.MSELoss()
    critic_criterion = nn.MSELoss()

    print("\n🧠 Starting Supervised Training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for b_obs, b_graphs, b_act, b_ret in pbar:
            b_obs = b_obs.to(DEVICE)
            b_graphs = b_graphs.to(DEVICE)
            b_act = b_act.to(DEVICE)
            b_ret = b_ret.to(DEVICE)

            # Simple GRU init (Zero)
            gru_state = torch.zeros(1, b_obs.shape[0], Config.D_MODEL).to(DEVICE)

            # Use deterministic output if possible, but PPO model samples.
            # We minimize MSE between sampled action and expert.
            pred_act, _, _, pred_val, _ = model.get_action_and_value(
                b_obs, graph_data=b_graphs, action=None, gru_state=gru_state
            )

            loss_a = actor_criterion(pred_act, b_act)
            loss_c = critic_criterion(pred_val, b_ret)
            loss = loss_a + 0.5 * loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"L_Act": f"{loss_a.item():.4f}", "L_Crit": f"{loss_c.item():.4f}"})

        print(f"  Avg Loss: {total_loss / len(loader):.4f}")

    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    save_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update': 0
    }
    torch.save(save_data, "checkpoints/model_latest.pt")
    torch.save(save_data, "checkpoints/model_pretrained.pt")
    print("✅ Done!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_supervised()