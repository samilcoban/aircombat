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
            # Phase 1: Tail Chase (Drone Chase)
            # Blue spawns 2km behind Red. Red is slow.
            sep = 2000.0

            # Red Position
            rx, ry = cx, cy
            r_heading_deg = axis_deg
            r_speed = 300.0  # Slow drone

            # Blue Position (Behind Red)
            # -Cos means behind
            bx = cx - sep * math.cos(math.radians(axis_deg))
            by = cy - sep * math.sin(math.radians(axis_deg))

            b_heading_deg = axis_deg  # Pointing at Red
            b_speed = 500.0  # Faster than Red (Closing)

        elif self.phase == 2:
            # Phase 2: Dogfight (Neutral Merge)
            # Closer range (10km), Head-on
            sep = 10000.0

            bx = cx + (sep / 2) * math.cos(math.radians(axis_deg + 180))
            by = cy + (sep / 2) * math.sin(math.radians(axis_deg + 180))
            b_heading_deg = axis_deg
            b_speed = 600.0

            rx = cx + (sep / 2) * math.cos(math.radians(axis_deg))
            ry = cy + (sep / 2) * math.sin(math.radians(axis_deg))
            r_heading_deg = (axis_deg + 180) % 360.0  # Head on
            r_speed = 600.0

        else:
            # Phase 3+: BVR (Long Range)
            sep = rng.uniform(30000.0, 50000.0)
            bx = cx + (sep / 2) * math.cos(math.radians(axis_deg + 180))
            by = cy + (sep / 2) * math.sin(math.radians(axis_deg + 180))
            b_heading_deg = axis_deg
            b_speed = 900.0

            rx = cx + (sep / 2) * math.cos(math.radians(axis_deg))
            ry = cy + (sep / 2) * math.sin(math.radians(axis_deg))
            r_heading_deg = (axis_deg + 180) % 360.0
            r_speed = 900.0

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

        self._compute_frame_data()

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
        self._compute_frame_data()

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
        """
        Calculates Potential-Based Reward Shaping (PBRS) value.
        Formula: Phi = 0.5 * DistFactor + 0.5 * (AlignFactor * EnergyFactor)

        Goals:
        1. Encorage closing distance (DistFactor).
        2. Encourage aiming at enemy (AlignFactor).
        3. BUT ONLY if we have speed (EnergyFactor).
           This prevents the "Turret Hack" where agents stall to aim.
        """
        if agent_id not in self.core.entities: return 0.0

        agent = self.core.entities[agent_id]

        # Find best target (nearest enemy)
        best_target = None
        min_dist = float('inf')

        # We can use the core's cached matrices if we want, but simple loop is fine here for O(N)
        # using the helper get_relative_data which uses the cache.
        for rid in self.red_ids:
            if rid not in self.core.entities: continue
            data = self.core.get_relative_data(agent_id, rid)
            if data and data[0] < min_dist:
                min_dist = data[0]
                best_target = data  # (Dist, RelPos, RelVel, ATA, AA, LocalPos)

        # If no enemies alive, potential is max (mission accomplished state)
        if best_target is None: return 1.0

        dist, _, _, ata_cos, _, _ = best_target

        # 1. Distance Factor (Normalized 0.0 to 1.0)
        # Max effective combat range ~40km (though sensors see further)
        # 0 at >40km, 1 at 0km.
        dist_norm = np.clip(1.0 - (dist / 40000.0), 0.0, 1.0)

        # 2. Alignment Factor (Normalized 0.0 to 1.0)
        # 1.0 if aiming directly at target, 0.0 if aiming away
        align_norm = 0.5 * (1.0 + ata_cos)

        # 3. Energy Factor (Normalized 0.0 to 1.0)
        # Penalty for being below Corner Speed (~400 knots)
        # At 400+ kts, factor is 1.0. At 0 kts, factor is 0.0.
        # This kills the reward for pointing the nose if you have no energy.
        energy_norm = np.clip(agent.speed / 400.0, 0.0, 1.0)

        # Combined Potential
        # We weigh Position (Geometry) and Kinetic (Energy)
        return 0.5 * dist_norm + 0.5 * (align_norm * energy_norm)

    def _calculate_reward(self, agent_id, win_condition, timeout, action):
        """
        Balanced Reward Function (Scaled [-5, 5]).

        Updates:
        1. Missile Attribution: Checks owner_id to credit kills correctly.
        2. Hard Deck: 0m (Death).
        3. Soft Deck: Cubic curve starting at 3000m.
        """
        stats = {'missiles_fired': 0, 'cannons_fired': 0, 'kills': 0, 'locked': 0}
        breakdown = {'rew_survival': 0.0, 'rew_pos': 0.0, 'rew_kill': 0.0, 'rew_penalty': 0.0}

        rew = 0.0

        # =================================================================
        # 1. EVENT REWARDS (OBJECTIVE)
        # =================================================================
        for ev in self.core.events:
            if ev['type'] == 'kill':
                # --- MODIFIED: CHECK OWNER ID ---
                # Check if I am the killer OR if I own the missile that killed
                is_killer = (ev['killer'] == agent_id)
                is_owner = (ev.get('owner_id') == agent_id)

                if is_killer or is_owner:
                    rew += 4.0
                    breakdown['rew_kill'] += 4.0
                    stats['kills'] = 1
                # --------------------------------

        # =================================================================
        # 2. TERMINAL PENALTIES (DEATH)
        # =================================================================
        if agent_id not in self.core.entities:
            if agent_id in self.dead_agent_ids:
                return 0.0, True, "dead", stats, breakdown

            self.dead_agent_ids.add(agent_id)

            ev = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            is_crash = (not ev) or (ev.get('type') in ['crash', 'floor', 'floor_violation'])

            if is_crash:
                # Crash is the worst outcome (-5.0)
                rew -= 5.0
                breakdown['rew_penalty'] -= 5.0
                reason = "crash"
            else:
                # Getting shot is tactical failure (-2.5)
                rew -= 2.5
                breakdown['rew_penalty'] -= 2.5
                reason = "shot"

            return rew, True, reason, stats, breakdown

        agent = self.core.entities[agent_id]

        # =================================================================
        # 3. HARD DECK (0m)
        # =================================================================
        # Explicit check for ground collision
        if agent.alt <= 1.0:
            self.dead_agent_ids.add(agent_id)
            if agent_id in self.core.entities: del self.core.entities[agent_id]

            rew -= 5.0
            breakdown['rew_penalty'] -= 5.0
            return rew, True, "crash", stats, breakdown

        # =================================================================
        # 4. SOFT DECK (EXPONENTIAL 3000m -> 0m)
        # =================================================================
        SOFT_DECK = 3000.0

        if agent.alt < SOFT_DECK:
            # 1. Calculate proximity factor (0.0 at 3000m, 1.0 at 0m)
            proximity = (SOFT_DECK - agent.alt) / SOFT_DECK

            # 2. Apply Cubic Curve (Power of 3)
            # 0.1^3 = 0.001 (Very small start)
            # 0.9^3 = 0.729 (Very steep end)
            curve = proximity ** 3

            # 3. Scale
            penalty = 0.5 * curve

            # 4. Attitude Check
            # If nose is pointing DOWN while in the danger zone, double the pain.
            if agent.pitch < -0.1:
                penalty *= 2.0

            penalty = min(penalty, 1.0)

            rew -= penalty
            breakdown['rew_penalty'] -= penalty

        # Soft Stall Penalty (Unchanged)
        SOFT_STALL = 200.0
        if agent.speed < SOFT_STALL:
            severity = (SOFT_STALL - agent.speed) / SOFT_STALL
            penalty = 0.05 * severity
            rew -= penalty
            breakdown['rew_penalty'] -= penalty

        # =================================================================
        # 5. EXISTENCE REWARD
        # =================================================================
        rew += 0.005
        breakdown['rew_survival'] += 0.005

        # =================================================================
        # 6. POTENTIAL BASED REWARD SHAPING (PBRS)
        # =================================================================
        cur_phi = self._get_current_potential(agent_id)
        prev_phi = self.prev_potentials.get(agent_id, cur_phi)
        shaping = (self.cfg.GAMMA * cur_phi - prev_phi)

        # Clamp shaping
        shaping = np.clip(shaping, -0.1, 0.1)

        rew += shaping
        breakdown['rew_pos'] += shaping
        self.prev_potentials[agent_id] = cur_phi

        # =================================================================
        # 7. WEAPONS COST
        # =================================================================
        nearest_uid = None
        min_dist = float('inf')
        for rid in self.red_ids:
            if rid not in self.core.entities: continue
            d = self.core.get_relative_data(agent_id, rid)[0]
            if d < min_dist: min_dist = d; nearest_uid = rid

        if nearest_uid:
            _, is_locking = self.core.get_sensor_state(agent_id, nearest_uid)
            if is_locking: stats['locked'] = 1

            curr_ammo = agent.ammo
            prev_ammo = self.last_ammo.get(agent_id, curr_ammo)

            if curr_ammo < prev_ammo:
                stats['missiles_fired'] = 1
                rew -= 0.4
                breakdown['rew_penalty'] -= 0.4

            self.last_ammo[agent_id] = curr_ammo

        # =================================================================
        # 8. GLOBAL OUTCOME
        # =================================================================
        if win_condition:
            if stats['kills'] == 0:
                rew += 2.0  # Passive win
                breakdown['rew_survival'] += 2.0
                return rew, False, "win_passive", stats, breakdown
            else:
                rew += 3.0  # Active win bonus
                return rew, False, "win", stats, breakdown

        if timeout:
            rew -= 1.0
            breakdown['rew_penalty'] -= 1.0
            return rew, False, "timeout", stats, breakdown

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

    def _compute_frame_data(self):
        """
        Centralized Vectorized Calculation.
        1. Reads Entity Objects -> Node Matrix (O(N) Extraction)
        2. Reads Core Matrices -> Edge Matrix (O(N^2) Vectorized Calculation)

        This replaces the inline logic previously found in _get_graph_state.
        """
        self.frame_active_uids = list(self.core.entities.keys())
        n = len(self.frame_active_uids)

        # Map UID to Index for O(1) slicing later
        self.frame_uid_map = {uid: i for i, uid in enumerate(self.frame_active_uids)}

        if n == 0:
            self.frame_node_feats = np.zeros((0, self.cfg.NODE_DIM), dtype=np.float32)
            self.frame_edge_matrix = np.zeros((0, 0, self.cfg.EDGE_DIM), dtype=np.float32)
            return

        # 1. Node Features (Extraction)
        # O(N) Loop to read object state
        self.frame_node_feats = np.array(
            [self._get_node_features(self.core.entities[uid]) for uid in self.frame_active_uids],
            dtype=np.float32
        )

        # 2. Edge Features (Fully Vectorized Matrix Ops)
        # O(N^2) Matrix calculations using Core cache

        indices = [self.core.uid_to_index[uid] for uid in self.frame_active_uids]
        mesh_idx = np.ix_(indices, indices)

        # Slice Core Matrices
        dists = self.core.dist_matrix[mesh_idx]  # (N, N)
        ata = self.core.ata_cos_matrix[mesh_idx]  # (N, N)
        aa = self.core.aa_cos_matrix[mesh_idx]  # (N, N)
        local_pos = self.core.local_pos_matrix[mesh_idx]  # (N, N, 3)
        rel_vel = self.core.rel_vel_matrix[mesh_idx]  # (N, N, 3)
        rel_pos = self.core.rel_pos_matrix[mesh_idx]  # (N, N, 3)

        # Derived Features
        headings = np.array([self.core.entities[uid].heading for uid in self.frame_active_uids])
        h_diff = headings[:, None] - headings[None, :]
        align = np.cos(h_diff)

        # Closure
        dot_vp = np.einsum('ijk,ijk->ij', rel_vel, rel_pos)
        safe_dist = dists + 1e-6
        closure = np.clip(-dot_vp / safe_dist / 2000.0, -1.0, 1.0)

        # Broadcast Properties
        speeds = np.array([self.core.entities[uid].speed / 1000.0 for uid in self.frame_active_uids])
        tgt_spd_mat = np.tile(speeds, (n, 1))

        types = np.array([1.0 if self.core.entities[uid].type == "plane" else -1.0 for uid in self.frame_active_uids])
        tgt_type_mat = np.tile(types, (n, 1))

        teams = np.array([1.0 if self.core.entities[uid].team == "blue" else -1.0 for uid in self.frame_active_uids])
        team_rel_mat = teams[:, None] * teams[None, :]

        # Visibility Logic
        is_notched = np.abs(closure) < 0.01
        in_radar_range = dists < (self.cfg.RADAR_RANGE_KM * 1000.0)
        in_radar_fov = ata > math.cos(math.radians(self.cfg.RADAR_FOV_DEG / 2.0))
        is_radar = in_radar_range & in_radar_fov & (~is_notched)
        is_visual = dists < 5000.0

        # MAWS Logic
        tgt_ids = np.array([self.core.entities[uid].target_id if self.core.entities[uid].target_id else -1 for uid in
                            self.frame_active_uids])
        uids_arr = np.array(self.frame_active_uids)
        maws_mat = (tgt_ids[None, :] == uids_arr[:, None]) & (tgt_type_mat == -1.0)

        vis_mask = (team_rel_mat > 0) | is_visual | is_radar | maws_mat
        vis_feat = vis_mask.astype(np.float32)

        # Stack into Cube (N, N, 12)
        self.frame_edge_matrix = np.stack([
            dists / 60000.0,
            local_pos[:, :, 0] / 60000.0,
            local_pos[:, :, 1] / 60000.0,
            local_pos[:, :, 2] / 10000.0,
            ata,
            aa,
            align,
            closure,
            tgt_spd_mat,
            tgt_type_mat,
            team_rel_mat,
            vis_feat
        ], axis=2)

    def _get_obs(self, ego_id):
        """
        Actor Obs: Slices the pre-computed frame data. O(N).
        """
        if ego_id not in self.frame_uid_map:
            return np.zeros(self.cfg.OBS_DIM, dtype=np.float32)

        idx = self.frame_uid_map[ego_id]

        # 1. Get Node
        ego_vec = self.frame_node_feats[idx]

        # 2. Get Edges (Slice from Matrix)
        row_edges = self.frame_edge_matrix[idx]  # Shape (N, 12)

        # 3. Filter
        valid_mask = np.ones(len(row_edges), dtype=bool)
        valid_mask[idx] = False  # No self-loop
        vis_mask = row_edges[:, 11] > 0.5  # Check visibility channel

        visible_edges = row_edges[valid_mask & vis_mask]

        # 4. Sort
        if len(visible_edges) > 0:
            sort_order = np.argsort(visible_edges[:, 0])
            sorted_edges = visible_edges[sort_order]
        else:
            sorted_edges = visible_edges

        # 5. Pad/Truncate
        max_tracks = self.cfg.MAX_ENTITIES - 1
        if len(sorted_edges) > max_tracks:
            sorted_edges = sorted_edges[:max_tracks]

        flat_tracks = sorted_edges.flatten()
        needed = (max_tracks * self.cfg.EDGE_DIM) - len(flat_tracks)
        if needed > 0:
            flat_tracks = np.pad(flat_tracks, (0, needed), 'constant')

        return np.concatenate([ego_vec, flat_tracks])

    def _get_graph_state(self):
        """
        Critic Obs: Wraps pre-computed frame data into PyG Data.
        See _compute_frame_data() for the actual math.
        """
        if len(self.frame_active_uids) == 0: return None

        n = len(self.frame_active_uids)

        # Filter self-loops (i != j)
        mask = ~np.eye(n, dtype=bool)

        # Flatten Matrix -> Edges list
        edge_attr = self.frame_edge_matrix[mask]

        rows, cols = np.indices((n, n))
        edge_index = np.stack([rows[mask], cols[mask]], axis=0)

        if edge_attr.shape[0] == 0:
            return {
                "x": self.frame_node_feats,
                "edge_index": np.zeros((2, 0), dtype=np.int64),
                "edge_attr": np.zeros((0, self.cfg.EDGE_DIM), dtype=np.float32)
            }

        return {
            "x": self.frame_node_feats,
            "edge_index": edge_index.astype(np.int64),
            "edge_attr": edge_attr
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