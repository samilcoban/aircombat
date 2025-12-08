# ================================================
# FILE: src/env_flat.py
# ================================================
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import torch

from config import Config
from src.core_flat import AirCombatCore
from src.utils.map_limits_flat import MapLimits


class AirCombatEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.cfg = Config
        self.core = None
        self.map_limits = MapLimits(*self.cfg.MAP_LIMITS)

        self.n_agents = self.cfg.N_AGENTS

        # UPDATED: Dimension Assertion based on Unified Dims
        expected_obs_dim = self.cfg.NODE_DIM + ((self.cfg.MAX_ENTITIES - 1) * self.cfg.EDGE_DIM)
        assert self.cfg.OBS_DIM == expected_obs_dim, \
            f"OBS_DIM Mismatch! Config:{self.cfg.OBS_DIM} vs Calc:{expected_obs_dim}"

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_agents, self.cfg.ACTION_DIM),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_agents, self.cfg.OBS_DIM),
            dtype=np.float32
        )

        self.blue_ids = []
        self.red_ids = []
        self.phase = 1
        self.kappa = 0.0
        self.global_step = 0

        self.last_actions = {}
        self.last_ammo = {}
        self.dead_agent_ids = set()
        self.prev_potentials = {}

    def set_global_step(self, step):
        self.global_step = step

    def _get_guidance_scale(self):
        decay_horizon = self.cfg.GUIDANCE_DECAY_STEPS
        progress = min(1.0, self.global_step / decay_horizon)
        return 1.0 - progress

    def set_phase(self, phase_id, progress=0.0):
        self.phase = phase_id

    def set_kappa(self, k):
        self.kappa = k

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        self.core = AirCombatCore()
        self.blue_ids = []
        self.red_ids = []

        self.last_actions = {}
        self.last_ammo = {}
        self.dead_agent_ids = set()
        self.prev_potentials = {}

        # Geometry Setup
        cx_rel, cy_rel = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        cx, cy = self.map_limits.absolute_position(cx_rel, cy_rel)
        axis_deg = rng.uniform(0.0, 360.0)
        spawn_alt = 5000.0

        if self.phase == 1:
            sep = rng.uniform(8000.0, 12000.0)
            rx, ry = cx, cy
            r_heading_deg = axis_deg
            r_speed = 400.0

            bx = cx - sep * math.cos(math.radians(axis_deg))
            by = cy - sep * math.sin(math.radians(axis_deg))

            heading_error = rng.uniform(-20.0, 20.0)
            b_heading_deg = (axis_deg + heading_error) % 360.0
            b_speed = 600.0
        else:
            sep = rng.uniform(30000.0, 50000.0)
            bx = cx + (sep / 2) * math.cos(math.radians(axis_deg + 180))
            by = cy + (sep / 2) * math.sin(math.radians(axis_deg + 180))
            b_heading_deg = axis_deg
            b_speed = 900.0

            rx = cx + (sep / 2) * math.cos(math.radians(axis_deg))
            ry = cy + (sep / 2) * math.sin(math.radians(axis_deg))
            r_heading_deg = (axis_deg + 180) % 360.0
            r_speed = 600.0 if self.phase == 2 else 900.0

        perp_rad = math.radians(b_heading_deg + 90)
        off_x, off_y = math.cos(perp_rad), math.sin(perp_rad)

        for i in range(self.n_agents):
            offset = (i - (self.n_agents - 1) / 2.0) * 500.0
            bid = self.core.spawn(
                bx + off_x * offset, by + off_y * offset, spawn_alt,
                math.radians(b_heading_deg), b_speed, "blue", "plane"
            )
            self.blue_ids.append(bid)
            self.last_ammo[bid] = 4
            self.last_actions[bid] = np.zeros(self.cfg.ACTION_DIM)

        n_red = 1
        if self.phase > 3: n_red = rng.integers(1, self.cfg.N_ENEMIES_MAX + 1)

        r_perp_rad = math.radians(r_heading_deg + 90)
        roff_x, roff_y = math.cos(r_perp_rad), math.sin(r_perp_rad)

        for i in range(n_red):
            offset = (i - (n_red - 1) / 2.0) * 500.0
            rid = self.core.spawn(
                rx + roff_x * offset, ry + roff_y * offset, spawn_alt,
                math.radians(r_heading_deg), r_speed, "red", "plane"
            )
            self.red_ids.append(rid)

        self.core.update_spatial_cache()

        for bid in self.blue_ids:
            self.prev_potentials[bid] = self._get_current_potential(bid)

        info = {
            "red_obs": self._get_all_red_obs(),
            "graph_data": self._get_graph_state(),
            "termination_reason": "none",
            "stat_kills": 0
        }
        return self._get_all_blue_obs(), info

    def step(self, action, red_actions=None):
        actions_dict = {}
        alpha = 0.6

        for i, agent_id in enumerate(self.blue_ids):
            if agent_id not in self.core.entities: continue
            if i < len(action):
                raw_act = action[i]
                prev_act = self.last_actions.get(agent_id, np.zeros_like(raw_act))
                smoothed = np.zeros_like(raw_act)
                smoothed[:3] = alpha * raw_act[:3] + (1 - alpha) * prev_act[:3]
                smoothed[3:] = raw_act[3:]
                self.last_actions[agent_id] = smoothed
                actions_dict[agent_id] = smoothed

        if red_actions is not None:
            if isinstance(red_actions, (np.ndarray, list)):
                for i, agent_id in enumerate(self.red_ids):
                    if i < len(red_actions): actions_dict[agent_id] = red_actions[i]
            elif isinstance(red_actions, dict):
                actions_dict.update(red_actions)

        if self.phase == 1:
            for agent_id, act in actions_dict.items():
                if agent_id in self.blue_ids:
                    act[2] = 1.0
                    act[1] = np.clip(act[1], -0.3, 0.3)
                    actions_dict[agent_id] = act

        self.core.step(actions_dict, self.kappa)
        self.core.update_spatial_cache()

        reds_alive = sum(1 for uid in self.red_ids if uid in self.core.entities)
        blues_alive = sum(1 for uid in self.blue_ids if uid in self.core.entities)

        all_enemies_dead = (reds_alive == 0)
        defeat = (blues_alive == 0)
        timeout = (self.core.time >= self.cfg.MAX_DURATION_SEC)

        global_term = all_enemies_dead or defeat
        global_trunc = timeout

        rewards = []
        dones = []
        episode_stats = {'missiles_fired': 0, 'cannons_fired': 0, 'kills': 0, 'locked': 0}
        step_breakdown = {'rew_survival': 0.0, 'rew_pos': 0.0, 'rew_kill': 0.0, 'rew_penalty': 0.0}

        stall_ratio, g_load = 0.0, 0.0

        for i, agent_id in enumerate(self.blue_ids):
            act = actions_dict.get(agent_id, np.zeros(5))
            rew, term, _, stats, breakdown = self._calculate_reward(agent_id, all_enemies_dead, timeout, act)
            rewards.append(rew)
            dones.append(term or global_term or global_trunc)

            for k in episode_stats: episode_stats[k] += stats[k]
            for k in step_breakdown: step_breakdown[k] += breakdown[k]

            if i == 0 and agent_id in self.core.entities:
                a = self.core.entities[agent_id]
                if a.speed < 150: stall_ratio = np.clip((150 - a.speed) / 50, 0, 1)
                g_load = a.g_load

        term_reason = "none"
        if all_enemies_dead:
            if episode_stats['kills'] > 0:
                term_reason = "win"
            else:
                term_reason = "enemy_crash"
        elif defeat:
            term_reason = "crash"
        elif timeout:
            term_reason = "timeout"

        info = {
            "termination_reason": term_reason,
            "red_obs": self._get_all_red_obs(),
            "graph_data": self._get_graph_state(),
            "agent_dones": np.array(dones, dtype=bool),
            "physics_stall_ratio": float(stall_ratio),
            "physics_g": float(g_load),
            "stat_kills": int(episode_stats['kills']),
            "stat_missiles_fired": int(episode_stats['missiles_fired']),
            "stat_locked": int(episode_stats['locked']),
            "reward_breakdown": step_breakdown
        }

        return self._get_all_blue_obs(), np.array(rewards, dtype=np.float32), global_term, global_trunc, info

    def _get_current_potential(self, agent_id):
        if agent_id not in self.core.entities: return 0.0
        min_dist = float('inf')
        best_data = None
        living_enemies = 0
        for rid in self.red_ids:
            if rid not in self.core.entities: continue
            living_enemies += 1
            data = self.core.get_relative_data(agent_id, rid)
            if data and data[0] < min_dist:
                min_dist = data[0]
                best_data = data
        if living_enemies == 0: return 1.0
        if best_data:
            dist, _, _, ata_cos, _, _ = best_data
            dist_norm = np.clip(1.0 - (dist / 40000.0), 0.0, 1.0)
            align_norm = np.clip((ata_cos + 1.0) / 2.0, 0.0, 1.0)
            return (0.4 * dist_norm) + (0.6 * align_norm)
        return 0.0

    def _calculate_reward(self, agent_id, win_condition, timeout, action):
        stats = {'missiles_fired': 0, 'cannons_fired': 0, 'kills': 0, 'locked': 0}
        breakdown = {'rew_survival': 0.0, 'rew_pos': 0.0, 'rew_kill': 0.0, 'rew_penalty': 0.0}

        rew = 0.0

        # =================================================================
        # 1. EVENT REWARDS (KILLS) - CHECK FIRST
        # =================================================================
        # We check this before death logic. If agent dies this step, but
        # managed to kill someone simultaneously, they deserve the points.
        for ev in self.core.events:
            if ev['type'] == 'kill' and ev['killer'] == agent_id:
                rew += 10.0
                breakdown['rew_kill'] += 10.0
                stats['kills'] = 1

        # =================================================================
        # 2. DEATH / TERMINAL CHECK
        # =================================================================
        if agent_id not in self.core.entities:
            # Prevent double counting if already processed as dead
            if agent_id in self.dead_agent_ids:
                return 0.0, True, "dead", stats, breakdown

            self.dead_agent_ids.add(agent_id)

            # Determine cause of death
            ev = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            is_crash = (not ev) or (ev.get('type') in ['crash', 'floor'])

            # PENALTIES
            # Crash/Floor: -10.0 (Incompetence)
            # Shot by Enemy: -5.0  (Tactical Failure, but better than crashing)
            penalty = -10.0 if is_crash else -5.0

            rew += penalty
            breakdown['rew_penalty'] += penalty

            reason = "crash" if is_crash else "shot"

            # Return immediately. The episode for this agent is over.
            return rew, True, reason, stats, breakdown

        # =================================================================
        # 3. LIVE AGENT REWARDS
        # =================================================================
        agent = self.core.entities[agent_id]

        # A. EXISTENCE REWARD (The Suicide Fix)
        # Tiny positive reward to counteract soft penalties.
        # Makes "struggling to survive" mathematically better than "giving up".
        rew += 0.001
        breakdown['rew_survival'] += 0.001

        # B. PHYSICS SAFETY (Continuous Gradients)

        # Soft Floor: Warn if below 3500m (Hard deck is 2000m)
        # Penalty ramps from 0.0 to -0.1 per step as you get lower
        if agent.alt < 3500.0:
            floor_dist = (3500.0 - agent.alt) / 1500.0
            floor_pen = 0.05 * floor_dist
            rew -= floor_pen
            breakdown['rew_penalty'] -= floor_pen

        # Soft Stall: Warn if below 350 knots (Stall is ~150)
        # Penalty ramps as you get slower. Teaches "Speed is Life".
        if agent.speed < 350.0:
            speed_dist = (350.0 - agent.speed) / 200.0
            stall_pen = 0.05 * speed_dist
            rew -= stall_pen
            breakdown['rew_penalty'] -= stall_pen

        # Hard Deck Safety Net (If physics engine didn't catch it yet)
        if agent.alt < 2000.0:
            self.dead_agent_ids.add(agent_id)
            if agent_id in self.core.entities: del self.core.entities[agent_id]
            rew -= 10.0  # Same as crash
            breakdown['rew_penalty'] -= 10.0
            return rew, True, "floor_violation", stats, breakdown

        # C. SHAPING (PBRS)
        # Potential Based Reward Shaping.
        # removed 'scale' (guidance decay) to ensure stationarity for the Critic.
        cur_phi = self._get_current_potential(agent_id)
        prev_phi = self.prev_potentials.get(agent_id, cur_phi)

        # Magnitude set to 1.0. Total accumulation over episode is small (< 2.0).
        shaping = (0.99 * cur_phi - prev_phi) * 1.0
        rew += shaping
        breakdown['rew_pos'] += shaping
        self.prev_potentials[agent_id] = cur_phi

        # D. WEAPONS LOGIC
        # Identify target
        nearest_uid = None
        min_dist = float('inf')
        for rid in self.red_ids:
            if rid not in self.core.entities: continue
            d = self.core.get_relative_data(agent_id, rid)[0]
            if d < min_dist: min_dist = d; nearest_uid = rid

        if nearest_uid:
            # Check Lock (Logging only, no reward for staring)
            _, is_locking = self.core.get_sensor_state(agent_id, nearest_uid)
            if is_locking:
                stats['locked'] = 1

            # Fire Discipline
            curr_ammo = agent.ammo
            prev_ammo = self.last_ammo.get(agent_id, curr_ammo)

            if curr_ammo < prev_ammo:  # Missile fired
                stats['missiles_fired'] = 1

                # Assess Shot Quality
                data = self.core.get_relative_data(agent_id, nearest_uid)
                dist_km = data[0] / 1000.0
                ata_cos = data[3]

                # Reward good shots, punish spam
                # Range < 60km, Nose within ~18 degrees (cos > 0.95)
                if dist_km < self.cfg.MISSILE_RANGE_KM and ata_cos > 0.95:
                    rew += 1.0  # Good shot bonus
                    breakdown['rew_kill'] += 1.0
                else:
                    rew -= 2.0  # Wasted ammo penalty
                    breakdown['rew_penalty'] -= 2.0

            self.last_ammo[agent_id] = curr_ammo

        # =================================================================
        # 4. WIN CONDITION (CLEANUP)
        # =================================================================
        if win_condition:
            if stats['kills'] > 0:
                # Active Win: We shot them down.
                rew += 5.0
                breakdown['rew_kill'] += 5.0
                return rew, False, "win", stats, breakdown
            else:
                # Passive Win: They crashed / ran out of fuel.
                # Smaller reward.
                rew += 2.0
                breakdown['rew_survival'] += 2.0
                return rew, False, "win_passive", stats, breakdown

        return rew, False, "none", stats, breakdown

    def _get_all_blue_obs(self):
        return np.stack([self._get_obs(uid) for uid in self.blue_ids]).astype(np.float32)

    def _get_all_red_obs(self):
        if not self.red_ids: return np.zeros((1, self.cfg.OBS_DIM), dtype=np.float32)
        obs_list = []
        for uid in self.red_ids:
            if uid in self.core.entities:
                obs_list.append(self._get_obs(uid))
            else:
                obs_list.append(np.zeros(self.cfg.OBS_DIM, dtype=np.float32))
        return np.stack(obs_list).astype(np.float32)

    # --- UNIFIED FEATURE EXTRACTION ---

    def _get_obs(self, ego_id):
        """Actor Observation: [Ego_Node || Edge_1 || Edge_2 ... ]"""
        ego_ent = self.core.entities.get(ego_id)
        if not ego_ent: return np.zeros(self.cfg.OBS_DIM, dtype=np.float32)

        # 1. Ego (Unified Node)
        ego_vec = self._get_node_features(ego_ent)

        # 2. Tracks (Unified Edge)
        track_vecs = []
        for uid, ent in self.core.entities.items():
            if uid == ego_id: continue

            is_visible = False
            if ent.team == ego_ent.team:
                is_visible = True
            else:
                vis, _ = self.core.get_sensor_state(ego_id, uid)
                is_maws = (ent.type == "missile" and ent.target_id == ego_id)
                if vis or is_maws: is_visible = True

            if is_visible:
                edge = self._get_edge_features(ego_id, uid, visible_flag=1.0)
                track_vecs.append(edge)

        # Sort by Distance (Index 0 of Edge)
        track_vecs.sort(key=lambda x: x[0])

        max_tracks = self.cfg.MAX_ENTITIES - 1
        if len(track_vecs) > max_tracks:
            track_vecs = track_vecs[:max_tracks]

        padding = max_tracks - len(track_vecs)
        if padding > 0:
            track_vecs.extend([np.zeros(self.cfg.EDGE_DIM, dtype=np.float32)] * padding)

        flat_tracks = np.concatenate(track_vecs)
        return np.concatenate([ego_vec, flat_tracks]).astype(np.float32)

    def _get_graph_state(self):
        """Critic Observation: Full Graph (All Nodes + All Edges)"""
        active_uids = list(self.core.entities.keys())
        if not active_uids: return None

        node_feats = []
        for uid in active_uids:
            ent = self.core.entities[uid]
            node_feats.append(self._get_node_features(ent))

        edge_index = []
        edge_attr = []
        n = len(active_uids)

        for i in range(n):
            for j in range(n):
                if i == j: continue
                uid_a = active_uids[i]
                uid_b = active_uids[j]

                ent_a = self.core.entities[uid_a]
                ent_b = self.core.entities[uid_b]

                # Critic Visibility Flag: "Does A see B?"
                is_vis = 0.0
                if ent_a.team == ent_b.team:
                    is_vis = 1.0
                else:
                    vis, _ = self.core.get_sensor_state(uid_a, uid_b)
                    if vis or (ent_b.type == "missile" and ent_b.target_id == uid_a):
                        is_vis = 1.0

                edge_vec = self._get_edge_features(uid_a, uid_b, visible_flag=is_vis)
                if edge_vec is not None:
                    edge_index.append([i, j])
                    edge_attr.append(edge_vec)

        if not edge_index:
            return {
                "x": np.array(node_feats, dtype=np.float32),
                "edge_index": np.zeros((2, 0), dtype=np.int64),
                "edge_attr": np.zeros((0, self.cfg.EDGE_DIM), dtype=np.float32)
            }

        return {
            "x": np.array(node_feats, dtype=np.float32),
            "edge_index": np.array(edge_index, dtype=np.int64).T,
            "edge_attr": np.array(edge_attr, dtype=np.float32)
        }

    def _get_node_features(self, e):
        """
        Unified Node (16D): Private Absolute State
        [Exist, Team, Type, X, Y, Alt, CosH, SinH, SinP, SinR, Spd, G, Fuel, Ammo, Chaff, CM]
        """
        xn, yn = self.map_limits.relative_position(e.x, e.y)
        return np.array([
            1.0,
            1.0 if e.team == "blue" else -1.0,
            1.0 if e.type == "plane" else -1.0,
            xn, yn, e.alt / 15000.0,
            math.cos(e.heading), math.sin(e.heading),
            math.sin(e.pitch), math.sin(e.roll),
                    e.speed / 1000.0, e.g_load / 9.0,
            e.fuel, e.ammo / 4.0, e.chaff / 20.0,
            1.0 if e.cm_active else 0.0
        ], dtype=np.float32)

    def _get_edge_features(self, uid_a, uid_b, visible_flag=1.0):
        """
        Unified Edge (12D): Public Relative/Sensor State
        [Dist, LX, LY, LZ, ATA, AA, Align, Close, TgtSpd, TgtType, TeamRel, Vis]
        """
        data = self.core.get_relative_data(uid_a, uid_b)
        if data is None:
            return np.zeros(self.cfg.EDGE_DIM, dtype=np.float32)

        dist, rel_pos, rel_vel, ata, aa, local_pos = data
        target = self.core.entities[uid_b]
        observer = self.core.entities[uid_a]

        closure = 0.0
        if dist > 0:
            closure = np.clip(-np.dot(rel_vel, rel_pos / dist) / 2000.0, -1, 1)

        align = math.cos(observer.heading - target.heading)
        team_rel = 1.0 if observer.team == target.team else -1.0

        return np.array([
            dist / 60000.0,
            local_pos[0] / 60000.0,
            local_pos[1] / 60000.0,
            local_pos[2] / 10000.0,
            ata, aa, align, closure,
            target.speed / 1000.0,
            1.0 if target.type == "plane" else -1.0,
            team_rel,
            visible_flag
        ], dtype=np.float32)