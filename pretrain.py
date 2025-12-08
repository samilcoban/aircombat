# ================================================
# FILE: pretrain.py
# ================================================
import torch
import torch.nn as nn
import torch.optim as optim
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

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.env import AirCombatEnv
from src.model import HybridActorCritic

# === CONFIGURATION ===
PRETRAIN_STEPS = 500_000
BATCH_SIZE = 128
EPOCHS = 10
LR = 3e-4
DEVICE = Config.DEVICE
MAX_PRETRAIN_STEPS = 600  # Slightly longer to allow BVR engagements to finish


# ================================================
# 1. INSTRUCTOR BOT (The 3-in-1 Expert)
# ================================================
class InstructorBot:
    """
    A unified expert that switches behavior modes based on tactical context.
    Effectively acts as 3 different bots:
    1. Safety Pilot: Smooth flight, altitude hold.
    2. BVR Sniper: Lead pursuit, missile employment.
    3. Dogfighter: Pure pursuit, high-G, cannon employment.
    """

    def __init__(self):
        self.cfg = Config

    def get_action(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        # --- 1. Parse Ego State ---
        # [Exist, Team, Type, X, Y, Alt, CosH, SinH, SinP, SinR, Spd, G, Fuel, Ammo, Chaff, CM]
        ego_vec = obs[0:self.cfg.NODE_DIM]

        if ego_vec[0] < 0.5:  # Dead
            return np.zeros(5, dtype=np.float32)

        ego_alt_norm = ego_vec[5]
        ego_spd_norm = ego_vec[10]
        ego_ammo = ego_vec[13]  # Normalized

        # Recover roll (approx)
        current_roll = math.asin(np.clip(ego_vec[9], -1, 1))

        # --- 2. Parse Tracks (Find Target) ---
        track_data = obs[self.cfg.NODE_DIM:]
        num_tracks = len(track_data) // self.cfg.EDGE_DIM

        target = None
        closest_dist = float('inf')

        for i in range(num_tracks):
            start = i * self.cfg.EDGE_DIM
            vec = track_data[start:start + self.cfg.EDGE_DIM]

            # [Dist, LX, LY, LZ, ATA, AA, Align, Close, TgtSpd, TgtType, TeamRel, Vis]
            if vec[0] < 1e-5: continue  # Empty

            is_enemy = (vec[10] < -0.5)
            is_plane = (vec[9] > 0.5)
            dist_norm = vec[0]

            if is_enemy and is_plane and dist_norm < closest_dist:
                closest_dist = dist_norm
                target = vec

        # --- 3. Mode Selection & Execution ---

        # Safety Override: Stall Protection
        # If speed < 200 kts (approx 0.2 norm), nose down immediately
        if ego_spd_norm < 0.2:
            return self._safety_recovery(current_roll)

        if target is not None:
            dist_km = target[0] * 60.0

            if dist_km < 3.0:
                # MODE: DOGFIGHTER (Close Range)
                return self._dogfight_logic(target, current_roll)
            else:
                # MODE: BVR SNIPER (Long Range)
                return self._bvr_logic(target, current_roll, ego_ammo)
        else:
            # MODE: SAFETY PILOT (Patrol)
            return self._patrol_logic(ego_alt_norm, current_roll)

    def _safety_recovery(self, roll):
        # Full throttle, unload Gs, level wings
        return np.array([-roll, -0.5, 1.0, 0.0, 0.0], dtype=np.float32)

    def _patrol_logic(self, alt_norm, roll):
        """
        Mode 1: Fly Smoothly.
        - Hold altitude ~5000m (0.33 norm)
        - Gentle turns only
        """
        target_alt = 0.33
        alt_err = target_alt - alt_norm

        # Gentle G-pull to correct altitude (Max 2.0G)
        g_cmd = np.clip(alt_err * 2.0, -0.2, 0.2)

        # Level wings
        roll_cmd = np.clip(-roll, -0.5, 0.5)

        return np.array([roll_cmd, g_cmd, 0.6, 0.0, 0.0], dtype=np.float32)

    def _bvr_logic(self, target, current_roll, ammo):
        """
        Mode 2: BVR Intercept.
        - Lead Pursuit (Aim slightly ahead of target)
        - Moderate G (Max 4-5G)
        - Fire Missile if aligned
        """
        # Target Geometry
        # LX, LY are local coordinates.
        # For simple intercept, we want to zero out LY (put target in front)
        ly = target[2]
        ata_cos = target[4]
        dist_km = target[0] * 60.0

        # Guidance: Proportional Navigation roughly approximates to keeping LOS rate low
        # Simple Logic: Bank towards target
        desired_roll = np.clip(ly * 5.0, -1.0, 1.0)
        roll_cmd = np.clip(desired_roll - current_roll, -1.0, 1.0)

        # Pitch/G: Maintain altitude unless close, but pull if turning
        # If we are banked, we need G to turn.
        # Load Gs based on bank angle to maintain level turn
        g_for_turn = abs(desired_roll) * 0.5
        g_cmd = np.clip(g_for_turn, 0.0, 0.5)  # Max ~5G

        # Fire Logic
        fire = 0.0
        # Fire if: Pointing at target, In Range, Have Ammo
        if ata_cos > 0.95 and dist_km < 40.0 and ammo > 0:
            # Randomly fire to simulate human reaction time variance
            if np.random.rand() < 0.1:
                fire = 1.0

        return np.array([roll_cmd, g_cmd, 1.0, fire, 0.0], dtype=np.float32)

    def _dogfight_logic(self, target, current_roll):
        """
        Mode 3: Knife Fight.
        - Pure Pursuit (Nose on target)
        - High G (Max 9G)
        - Cannon usage
        """
        ly = target[2]
        lz = target[3]  # Vertical offset
        ata_cos = target[4]
        dist_km = target[0] * 60.0

        # Aggressive bank to target
        desired_roll = np.clip(ly * 10.0, -1.0, 1.0)
        roll_cmd = np.clip((desired_roll - current_roll) * 2.0, -1.0, 1.0)

        # Pull hard to bring nose around
        # If target is "above" (in local frame, meaning we need to pull up into them), pull G
        # LZ > 0 means target is "above" the nose
        g_cmd = np.clip(lz * 5.0, 0.0, 1.0)  # Max 9G

        # Add Gs to sustain turn if banked
        g_cmd += abs(current_roll) * 0.4
        g_cmd = np.clip(g_cmd, -0.2, 1.0)

        # Cannon Fire
        fire = 0.0
        if ata_cos > 0.98 and dist_km < 1.5:
            fire = 1.0  # Cannon trigger

        return np.array([roll_cmd, g_cmd, 1.0, fire, 0.0], dtype=np.float32)


# ================================================
# 2. SCENARIO WRAPPER (The Director)
# ================================================
class ScenarioWrapper(gym.Wrapper):
    """
    Forces specific scenarios upon reset to ensure diverse training data.
    Overrides the default random spawning of the environment.
    """

    def __init__(self, env):
        super().__init__(env)
        self.scenario_type = "random"

    def step(self, action, **kwargs):
        """Override step to pass through red_actions and other kwargs."""
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        # 1. Standard Reset
        obs, info = self.env.reset(**kwargs)

        # 2. Determine Scenario
        rand = np.random.rand()
        if rand < 0.30:
            self.scenario_type = "tail_chase"
            self._setup_tail_chase()
        elif rand < 0.60:
            self.scenario_type = "head_on"
            self._setup_head_on()
        elif rand < 0.80:
            self.scenario_type = "disadvantage"
            self._setup_disadvantage()
        else:
            self.scenario_type = "random"
            # Keep default env spawn
            pass

        # 3. Refresh Obs after teleportation
        # We need to manually trigger the env to refresh observations based on new positions
        if self.scenario_type != "random":
            self.env.unwrapped.core.update_spatial_cache()
            obs = self.env.unwrapped._get_all_blue_obs()
            # Re-fetch info to update Red obs if necessary
            info["red_obs"] = self.env.unwrapped._get_all_red_obs()

        return obs, info

    def _teleport_entity(self, uid, x, y, alt, heading, speed):
        ent = self.env.unwrapped.core.entities[uid]
        # Handle Flat vs Geodetic
        if hasattr(self.env.unwrapped.core, 'dist_matrix'):  # Flat
            ent.x = x
            ent.y = y
        else:  # Geodetic (Approximate mapping for this scenario logic)
            # Map 0,0 to center of map limits
            limits = self.env.unwrapped.map_limits
            center_lat = (limits.bottom_lat + limits.top_lat) / 2
            center_lon = (limits.left_lon + limits.right_lon) / 2
            # 1 deg lat ~ 111km
            ent.lat = center_lat + (y / 111000.0)
            ent.lon = center_lon + (x / 111000.0)

        ent.alt = alt
        ent.heading = math.radians(heading)
        ent.speed = speed
        ent.roll = 0.0
        ent.pitch = 0.0

    def _setup_tail_chase(self):
        """Blue 2km behind Red. Both fast."""
        if not self.env.unwrapped.blue_ids or not self.env.unwrapped.red_ids: return
        bid = self.env.unwrapped.blue_ids[0]
        rid = self.env.unwrapped.red_ids[0]

        # Red at 0,0, Heading North (90 deg Cartesian)
        self._teleport_entity(rid, 0, 2000, 5000, 90, 600)
        # Blue at 0,-2000, Heading North
        self._teleport_entity(bid, 0, 0, 5000, 90, 800)  # Blue slightly faster

    def _setup_head_on(self):
        """Blue and Red 30km apart, facing each other."""
        if not self.env.unwrapped.blue_ids or not self.env.unwrapped.red_ids: return
        bid = self.env.unwrapped.blue_ids[0]
        rid = self.env.unwrapped.red_ids[0]

        # Red at North, Heading South (270)
        self._teleport_entity(rid, 0, 15000, 6000, 270, 700)
        # Blue at South, Heading North (90)
        self._teleport_entity(bid, 0, -15000, 6000, 90, 700)

    def _setup_disadvantage(self):
        """Blue in front of Red (Defensive)."""
        if not self.env.unwrapped.blue_ids or not self.env.unwrapped.red_ids: return
        bid = self.env.unwrapped.blue_ids[0]
        rid = self.env.unwrapped.red_ids[0]

        # Red at 0,0, Heading North
        self._teleport_entity(rid, 0, 0, 5000, 90, 800)
        # Blue at 0, 3000, Heading North (Run away!)
        self._teleport_entity(bid, 0, 3000, 5000, 90, 600)


# ================================================
# 3. PARALLEL INFRASTRUCTURE
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

        # Instantiate Env with Scenario Wrapper
        env = ScenarioWrapper(env_fn_wrapper())
        env.reset(seed=seed)

        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                blue_act, red_act = data
                ob, reward, term, trunc, info = env.step(blue_act, red_actions=red_act)
                if term or trunc:
                    # Wrapper handles scenario randomization on reset automatically
                    ob_reset, info_reset = env.reset()
                    info['red_obs'] = info_reset.get('red_obs')
                    info['graph_data'] = info_reset.get('graph_data')
                    ob = ob_reset
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
    # Base Env -> TimeLimit
    return TimeLimitWrapper(AirCombatEnv())


# ================================================
# 4. DATA COLLECTION & TRAINING
# ================================================

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
    print(f"🚀 Initializing {Config.NUM_ENVS} Parallel Scenarios...")

    # Using the ScenarioWrapper implicitly via worker logic
    envs = ParallelMultiAgentEnv([make_env for _ in range(Config.NUM_ENVS)])

    print("✅ Workers Started. Initializing Instructor Bot...")
    bot = InstructorBot()

    master_obs = []
    master_graphs = []
    master_actions = []
    master_returns = []

    env_buffers = [{'obs': [], 'graphs': [], 'acts': [], 'rews': []} for _ in range(Config.NUM_ENVS)]

    obs, infos = envs.reset()

    print(f"🎥 Starting Data Collection (Target: {PRETRAIN_STEPS} Valid Steps)...")

    pbar = tqdm(total=PRETRAIN_STEPS, desc="Valid Data", unit="step")
    valid_steps_collected = 0
    total_simulated = 0
    discarded_episodes = 0

    while valid_steps_collected < PRETRAIN_STEPS:
        # 1. Instructor Inference
        blue_actions = get_bot_actions(bot, obs)

        # Red is also the instructor (Self-Play Logic) or simple
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

        total_simulated += Config.NUM_ENVS

        # 4. Processing
        for i in range(Config.NUM_ENVS):
            env_buffers[i]['rews'].append(rewards[i, 0])

            # Check if episode ended
            is_done = np.any(dones[i]) if isinstance(dones[i], np.ndarray) else dones[i]

            if is_done:
                ep_len = len(env_buffers[i]['obs'])
                total_return = sum(env_buffers[i]['rews'])

                # Quality Filter:
                # Discard episodes where the bot crashed immediately or failed miserably
                # With Scenario spawning, -20.0 is a reasonable cutoff (mostly valid flights)
                if total_return > -20.0:
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

        pbar.set_postfix({
            "Sim": total_simulated,
            "Drop": discarded_episodes,
            "MeanR": f"{np.mean(rewards):.2f}"
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

    # Checkpoint Dir
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for b_obs, b_graphs, b_act, b_ret in pbar:
            b_obs = b_obs.to(DEVICE)
            b_graphs = b_graphs.to(DEVICE)
            b_act = b_act.to(DEVICE)
            b_ret = b_ret.to(DEVICE)

            # GRU Init
            gru_state = torch.zeros(1, b_obs.shape[0], Config.D_MODEL).to(DEVICE)

            # Forward
            pred_act, _, _, pred_val, _ = model.get_action_and_value(
                b_obs, graph_data=b_graphs, action=None, gru_state=gru_state
            )

            # Loss
            loss_a = actor_criterion(pred_act, b_act)
            loss_c = critic_criterion(pred_val, b_ret)
            loss = loss_a + 0.5 * loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"L_Act": f"{loss_a.item():.4f}", "L_Crit": f"{loss_c.item():.4f}"})

        print(f"  Avg Loss: {total_loss / len(loader):.4f}")

        # Save every epoch
        save_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'update': 0
        }
        torch.save(save_data, f"checkpoints/model_pretrained_ep{epoch}.pt")

    # Final Save
    torch.save(save_data, "checkpoints/model_latest.pt")
    torch.save(save_data, "checkpoints/model_pretrained.pt")
    print("✅ Done!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_supervised()