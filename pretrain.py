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

# Add root directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.env import AirCombatEnv
from src.model import HybridActorCritic
from src.bot import HardcodedAce  # Using the single source of truth for the expert

# === CONFIGURATION ===
# Total valid timesteps to collect before training
PRETRAIN_STEPS = 500_000
# Number of sequences per batch (Effective batch size = 32 * SEQ_LEN)
BATCH_SIZE = 32
SEQ_LEN = Config.SEQ_LEN
EPOCHS = 10
LR = 3e-4
DEVICE = Config.DEVICE
# Max steps per episode during data collection
MAX_PRETRAIN_STEPS = 600


# ================================================
# 1. SCENARIO WRAPPER (The Director)
# ================================================
class ScenarioWrapper(gym.Wrapper):
    """
    Forces specific tactical scenarios upon reset to ensure diverse training data.
    Overrides the default random spawning of the environment.
    """

    def __init__(self, env):
        super().__init__(env)
        self.scenario_type = "random"

    def step(self, action, **kwargs):
        """Override step to pass through red_actions and other kwargs."""
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 1. Randomize Loadout (The Fix)
        # 50% chance to have NO missiles (Guns Only).
        # This teaches the network: "Check your ammo before deciding tactics."
        # It aligns pretraining with Phase 1 constraints.
        guns_only = (np.random.rand() < 0.5)

        if guns_only:
            for uid in self.env.unwrapped.blue_ids:
                if uid in self.env.unwrapped.core.entities:
                    self.env.unwrapped.core.entities[uid].ammo = 0
            # Red should also follow suit for fairness, or keep missiles to make it harder
            for uid in self.env.unwrapped.red_ids:
                if uid in self.env.unwrapped.core.entities:
                    self.env.unwrapped.core.entities[uid].ammo = 0

        # 2. Determine Scenario
        rand = np.random.rand()
        scenario_active = False

        if rand < 0.30:
            self.scenario_type = "tail_chase"
            self._setup_tail_chase(guns_only)  # Pass flag to adjust distance
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

        # 3. Cache Coherency Fix (Update matrices after changing ammo/pos)
        if scenario_active or guns_only:
            self.env.unwrapped.core.update_spatial_cache()
            self.env.unwrapped._compute_frame_data()
            obs = self.env.unwrapped._get_all_blue_obs()
            info["red_obs"] = self.env.unwrapped._get_all_red_obs()
            info["graph_data"] = self.env.unwrapped._get_graph_state()

        return obs, info

    def _teleport_entity(self, uid, x, y, alt, heading, speed):
        """Helper to move an entity to a specific state."""
        import math  # Explicit import for MP safety

        ent = self.env.unwrapped.core.entities[uid]

        # Handle Flat vs Geodetic Coordinate Systems
        if hasattr(self.env.unwrapped.core, 'dist_matrix'):  # Flat
            ent.x = x
            ent.y = y
        else:  # Geodetic (Approximate mapping for scenario logic)
            limits = self.env.unwrapped.map_limits
            center_lat = (limits.bottom_lat + limits.top_lat) / 2
            center_lon = (limits.left_lon + limits.right_lon) / 2
            # approx 111km per degree
            ent.lat = center_lat + (y / 111000.0)
            ent.lon = center_lon + (x / 111000.0)

        ent.alt = alt
        ent.heading = math.radians(heading)
        ent.speed = speed
        ent.roll = 0.0
        ent.pitch = 0.0

    def _setup_tail_chase(self, guns_only):
        """Blue behind Red."""
        if not self.env.unwrapped.blue_ids or not self.env.unwrapped.red_ids: return
        bid = self.env.unwrapped.blue_ids[0]
        rid = self.env.unwrapped.red_ids[0]

        # If guns only, start closer (2km). If missiles, start further (5km)
        dist = 2000 if guns_only else 5000

        self._teleport_entity(rid, 0, dist, 5000, 90, 600)
        self._teleport_entity(bid, 0, 0, 5000, 90, 800)

    def _setup_head_on(self, guns_only):
        """Head-to-head merge."""
        if not self.env.unwrapped.blue_ids or not self.env.unwrapped.red_ids: return
        bid = self.env.unwrapped.blue_ids[0]
        rid = self.env.unwrapped.red_ids[0]

        # If guns only, start at visual range (10km). If missiles, BVR (30km).
        # This prevents the "Guns Only" dataset from being 90% boring flying.
        half_dist = 5000 if guns_only else 15000

        self._teleport_entity(rid, 0, half_dist, 6000, 270, 700)
        self._teleport_entity(bid, 0, -half_dist, 6000, 90, 700)

    def _setup_disadvantage(self):
        """Blue in front of Red. Teaches defensive maneuvers."""
        if not self.env.unwrapped.blue_ids or not self.env.unwrapped.red_ids: return
        bid = self.env.unwrapped.blue_ids[0]
        rid = self.env.unwrapped.red_ids[0]

        # Red at 0,0, Heading North
        self._teleport_entity(rid, 0, 0, 5000, 90, 800)
        # Blue at 0,3km, Heading North (Run away!)
        self._teleport_entity(bid, 0, 3000, 5000, 90, 600)


# ================================================
# 2. PARALLEL INFRASTRUCTURE
# ================================================
def worker(remote, parent_remote, env_fn_wrapper, seed):
    """
    Multiprocessing worker function.
    Wraps the env in the ScenarioWrapper.
    """
    try:
        import random
        import numpy as np
        import torch
        import math
        import sys
        import os
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
                    # Carry over persistent info for continuous training logic if needed
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
    """Manages multiple environment processes."""

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
    """Enforces a maximum number of steps per episode."""

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
    env = AirCombatEnv()
    # CRITICAL: Set Phase to 3 to unlock full physics/weapons for the Expert during pretraining.
    # Phase 1 locks G-pull to 0.3, making it impossible for the Ace to aim.
    env.set_phase(3)
    return TimeLimitWrapper(env)


# ================================================
# 3. DATA COLLECTION & TRAINING
# ================================================

class SequenceDataset(Dataset):
    """
    Dataset that yields sequences of (SEQ_LEN) steps.
    Ensures the GRU learns temporal dependencies correctly.
    """

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
    """
    Stacks sequences into (Batch, Seq, Dim).
    Handles Graph batching by flattening (Batch * Seq) graphs.
    """
    obs_list, graph_list_seqs, act_list, ret_list, mask_list = zip(*batch)

    # 1. Standard Tensors: (Batch, Seq, Dim)
    b_obs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    b_act = torch.tensor(np.stack(act_list), dtype=torch.float32)
    b_ret = torch.tensor(np.stack(ret_list), dtype=torch.float32).unsqueeze(-1)
    b_mask = torch.tensor(np.stack(mask_list), dtype=torch.float32)

    # 2. Graphs: Flatten list of lists -> Single Batch
    # Input graph_list_seqs is tuple of lists: ([G1, G2..], [G1, G2..])
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
                    # Numpy often sends float64, causing the Double vs Float mismatch error.
                    flat_graphs.append(Data(x=torch.tensor(g['x'], dtype=torch.float32),
                                            edge_index=torch.tensor(g['edge_index'], dtype=torch.long),
                                            edge_attr=torch.tensor(g['edge_attr'], dtype=torch.float32)))
                else:
                    flat_graphs.append(g)

    b_graphs = Batch.from_data_list(flat_graphs)

    return b_obs, b_graphs, b_act, b_ret, b_mask


def get_bot_actions(bot, obs_batch):
    """Get expert actions for a batch of observations."""
    num_envs, n_agents, _ = obs_batch.shape
    actions = np.zeros((num_envs, n_agents, Config.ACTION_DIM), dtype=np.float32)
    for e in range(num_envs):
        for a in range(n_agents):
            actions[e, a] = bot.get_action(obs_batch[e, a])
    return actions


def collect_data_parallel():
    """
    Main data collection loop.
    Runs parallel environments, uses HardcodedAce to generate trajectories,
    filters bad episodes, and chunks data for LSTM/GRU training.
    """
    print(f"🚀 Initializing {Config.NUM_ENVS} Parallel Scenarios...")
    # Using the ScenarioWrapper implicitly via worker logic
    envs = ParallelMultiAgentEnv([make_env for _ in range(Config.NUM_ENVS)])

    print("✅ Workers Started. Initializing Hardcoded Ace...")
    bot = HardcodedAce()

    # Master storage for CHUNKS (Sequences)
    master_obs_chunks = []
    master_graph_chunks = []
    master_act_chunks = []
    master_ret_chunks = []
    master_mask_chunks = []  # 1.0 for valid data, 0.0 for padding

    # Temporary buffers for active episodes
    env_buffers = [{'obs': [], 'graphs': [], 'acts': [], 'rews': []} for _ in range(Config.NUM_ENVS)]

    obs, infos = envs.reset()

    valid_steps_collected = 0
    total_simulated = 0
    discarded_episodes = 0

    # Stats counters
    stats_kills = 0
    stats_fired = 0
    debug_returns = []

    pbar = tqdm(total=PRETRAIN_STEPS, desc="Valid Steps", unit="step")

    while valid_steps_collected < PRETRAIN_STEPS:
        # 1. Get Expert Actions
        blue_actions = get_bot_actions(bot, obs)

        # Red Actions (Self-Play logic or simple)
        current_red_obs = []
        for inf in infos:
            if inf and 'red_obs' in inf:
                current_red_obs.append(inf['red_obs'])
            else:
                current_red_obs.append(np.zeros((Config.N_AGENTS, Config.OBS_DIM)))
        red_obs_np = np.stack(current_red_obs)
        red_actions = get_bot_actions(bot, red_obs_np)

        # 2. Store Step Data
        for i in range(Config.NUM_ENVS):
            env_buffers[i]['obs'].append(obs[i, 0])
            env_buffers[i]['acts'].append(blue_actions[i, 0])
            if infos[i] and 'graph_data' in infos[i]:
                env_buffers[i]['graphs'].append(infos[i]['graph_data'])
            else:
                env_buffers[i]['graphs'].append(None)

        # 3. Step Environment
        next_obs, rewards, terms, truncs, next_infos = envs.step(blue_actions, red_actions)
        dones = np.logical_or(terms, truncs)

        total_simulated += Config.NUM_ENVS

        # Stats Aggregation
        for inf in next_infos:
            if inf:
                stats_kills += inf.get('stat_kills', 0)
                stats_fired += inf.get('stat_missiles_fired', 0)

        # 4. Process Dones
        for i in range(Config.NUM_ENVS):
            env_buffers[i]['rews'].append(rewards[i, 0])

            if np.any(dones[i]):
                # Episode Finished
                total_return = sum(env_buffers[i]['rews'])
                debug_returns.append(total_return)

                # Quality Filter:
                # With normalized rewards, a crash is -5.0.
                # Just surviving without accomplishing much is roughly 0.0 to -2.0.
                # Threshold of > -4.0 ensures we drop hard crashes/failures but keep survival/tactical flying.
                if total_return > -0.3:
                    ep_obs = env_buffers[i]['obs']
                    ep_graphs = env_buffers[i]['graphs']
                    ep_acts = env_buffers[i]['acts']

                    # Compute Returns-to-Go (for Critic training)
                    g = 0
                    ep_rets = []
                    for r in reversed(env_buffers[i]['rews']):
                        g = r + 0.99 * g
                        ep_rets.insert(0, g)

                    # Sequence Chunking
                    # We chop the episode into sequences of length SEQ_LEN
                    L = len(ep_obs)
                    for start in range(0, L, SEQ_LEN):
                        end = min(start + SEQ_LEN, L)
                        length = end - start

                        # Handle Padding for short/final chunks
                        if length < SEQ_LEN:
                            # Skip extremely short residual chunks (< half seq len) to reduce noise
                            if length < SEQ_LEN // 2: continue

                            pad_len = SEQ_LEN - length

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
                            c_mask = [1.0] * SEQ_LEN

                        master_obs_chunks.append(np.array(c_obs))
                        master_graph_chunks.append(c_graphs)
                        master_act_chunks.append(np.array(c_acts))
                        master_ret_chunks.append(np.array(c_rets))
                        master_mask_chunks.append(np.array(c_mask))

                        valid_steps_collected += length
                        pbar.update(length)
                else:
                    discarded_episodes += 1

                # Reset Buffer
                env_buffers[i] = {'obs': [], 'graphs': [], 'acts': [], 'rews': []}

        obs = next_obs
        infos = next_infos

        # Debug Info in Progress Bar
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
    """
    Main training loop.
    Loads data, trains the HybridActorCritic model using Behavioral Cloning (MSE Loss).
    """
    data = collect_data_parallel()
    if not data[0]:
        print("❌ No valid episodes! Check Bot logic or rewards.")
        return

    # Create Dataset and Loader
    dataset = SequenceDataset(*data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sequences)

    print(f"Initializing Model on {DEVICE}...")
    model = HybridActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Reduction='none' allows us to multiply by mask later
    actor_criterion = nn.MSELoss(reduction='none')
    critic_criterion = nn.MSELoss(reduction='none')

    print("\n🧠 Starting Supervised Training (Sequence Mode)...")
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for b_obs, b_graphs, b_act, b_ret, b_mask in pbar:
            # Move to device
            # b_obs: (Batch, Seq, Dim)
            b_obs = b_obs.to(DEVICE)
            b_graphs = b_graphs.to(DEVICE)  # Flattened Batch of Graphs
            b_act = b_act.to(DEVICE)
            b_ret = b_ret.to(DEVICE)
            b_mask = b_mask.to(DEVICE).unsqueeze(-1)  # (Batch, Seq, 1)

            # GRU Initialization
            # Since batches are independent sequence chunks, we initialize with zeros.
            # Ideally we would carry state between chunks of the same episode, but
            # for pretraining random shuffling is standard for stability.
            batch_dim = b_obs.shape[0]
            gru_state = torch.zeros(1, batch_dim, Config.D_MODEL).to(DEVICE)

            # Forward Pass
            # model handles sequence internally because input dim=3
            pred_act, _, _, pred_val, _ = model.get_action_and_value(
                b_obs, graph_data=b_graphs, action=None, gru_state=gru_state
            )

            # Calculate Loss
            # Apply mask to ignore padding steps
            loss_a = (actor_criterion(pred_act, b_act) * b_mask).sum() / (b_mask.sum() + 1e-8)
            loss_c = (critic_criterion(pred_val, b_ret) * b_mask).sum() / (b_mask.sum() + 1e-8)

            # Weighted sum (Actor priority)
            loss = loss_a + 0.5 * loss_c

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"L_Act": f"{loss_a.item():.4f}", "L_Crit": f"{loss_c.item():.4f}"})

        print(f"  Avg Loss: {total_loss / len(loader):.4f}")

        # Checkpoint
        save_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'update': 0
        }
        torch.save(save_data, f"checkpoints/model_pretrained_ep{epoch}.pt")

    # Final Save
    torch.save(save_data, "checkpoints/model_latest.pt")
    torch.save(save_data, "checkpoints/model_pretrained.pt")
    print("✅ Pretraining Complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_supervised()