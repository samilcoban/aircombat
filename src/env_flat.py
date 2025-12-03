# ================================================
# FILE: src/env_flat.py
# ================================================
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import torch

from config import Config
from src.core_flat import AirCombatCore, dist_2d, bearing_deg
from src.utils.map_limits_flat import MapLimits


class AirCombatEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.cfg = Config
        self.core = None
        self.map_limits = MapLimits(*self.cfg.MAP_LIMITS)

        self.n_agents = self.cfg.N_AGENTS

        # Action: [Roll, G-Pull, Throttle, Fire, Countermeasures]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_agents, self.cfg.ACTION_DIM),
            dtype=np.float32
        )

        # Observation: Flattened vector of all entities
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_agents, self.cfg.OBS_DIM),
            dtype=np.float32
        )

        self.blue_ids = []
        self.red_ids = []
        self.phase = 1
        self.kappa = 0.0

        # Training Progress
        self.global_step = 0
        self.total_steps = self.cfg.TOTAL_TIMESTEPS

        self.last_actions = {}
        self.last_ammo = {}
        self.dead_agent_ids = set()
        self.active_locks = set()
        self.prev_dist = {}

    def set_global_step(self, step):
        self.global_step = step

    def _get_guidance_scale(self):
        decay_horizon = 3_000_000
        progress = min(1.0, self.global_step / decay_horizon)
        return 1.0 - progress

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.core = AirCombatCore()
        self.blue_ids = []
        self.red_ids = []
        self.last_actions = {}
        self.last_ammo = {}
        self.dead_agent_ids = set()
        self.active_locks = set()
        self.prev_dist = {}

        cx_rel, cy_rel = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        cx, cy = self.map_limits.absolute_position(cx_rel, cy_rel)
        axis = rng.uniform(0.0, 360.0)

        spawn_alt = 5000.0

        if self.phase == 1:
            sep = rng.uniform(8000.0, 12000.0)
            rx, ry = cx, cy
            r_heading = axis
            r_speed = 400.0

            bx = cx - sep * math.cos(math.radians(axis))
            by = cy - sep * math.sin(math.radians(axis))

            heading_error = rng.uniform(-20.0, 20.0)
            b_heading = (axis + heading_error) % 360.0
            b_speed = 600.0
        else:
            sep = rng.uniform(30000.0, 50000.0)
            bx = cx + (sep / 2) * math.cos(math.radians(axis + 180))
            by = cy + (sep / 2) * math.sin(math.radians(axis + 180))
            b_heading = axis
            b_speed = 900.0

            rx = cx + (sep / 2) * math.cos(math.radians(axis))
            ry = cy + (sep / 2) * math.sin(math.radians(axis))
            r_heading = (axis + 180) % 360
            r_speed = 600.0 if self.phase == 2 else 900.0

        perp_rad = math.radians(b_heading + 90)
        off_x_unit = math.cos(perp_rad)
        off_y_unit = math.sin(perp_rad)

        for i in range(self.n_agents):
            offset_dist = (i - (self.n_agents - 1) / 2.0) * 500.0
            spawn_x = bx + off_x_unit * offset_dist
            spawn_y = by + off_y_unit * offset_dist

            # FIX 1.1: Pass spawn_alt to spawn
            bid = self.core.spawn(spawn_x, spawn_y, spawn_alt, b_heading, b_speed, "blue", "plane")
            self.blue_ids.append(bid)
            self.last_ammo[bid] = 4

        n_red = 1
        if self.phase > 3: n_red = rng.integers(1, self.cfg.N_ENEMIES_MAX + 1)

        r_perp_rad = math.radians(r_heading + 90)
        r_off_x = math.cos(r_perp_rad)
        r_off_y = math.sin(r_perp_rad)

        for i in range(n_red):
            offset_dist = (i - (n_red - 1) / 2.0) * 500.0
            spawn_rx = rx + r_off_x * offset_dist
            spawn_ry = ry + r_off_y * offset_dist

            # FIX 1.1: Pass spawn_alt
            rid = self.core.spawn(spawn_rx, spawn_ry, spawn_alt, r_heading, r_speed, "red", "plane")
            self.red_ids.append(rid)

        # Initialize spatial cache for first frame
        self.core.update_spatial_cache()

        if self.blue_ids and self.red_ids:
            # Use 3D distance from core cache or manual if cache fail
            dist, _, _ = self.core.get_relative_data(self.blue_ids[0], self.red_ids[0])
            if dist is not None:
                self.prev_dist[self.blue_ids[0]] = dist
            else:
                self.prev_dist[self.blue_ids[0]] = 10000.0

        info = {
            "red_obs": self._get_all_red_obs(),
            "graph_data": self._get_graph_state(),
            "termination_reason": "none",
            "stat_missiles_fired": 0,
            "stat_cannons_fired": 0,
            "stat_kills": 0,
            "stat_locked": 0,
        }
        return self._get_all_blue_obs(), info

    def set_phase(self, phase_id, progress=0.0):
        self.phase = phase_id

    def set_kappa(self, k):
        self.kappa = k

    def step(self, action, red_actions=None):
        actions_dict = {}
        for i, agent_id in enumerate(self.blue_ids):
            if i < len(action): actions_dict[agent_id] = action[i]

        if red_actions is not None:
            if isinstance(red_actions, (np.ndarray, list)):
                for i, agent_id in enumerate(self.red_ids):
                    if i < len(red_actions): actions_dict[agent_id] = red_actions[i]
            elif isinstance(red_actions, dict):
                actions_dict.update(red_actions)

        # Phase 1 Logic
        if self.phase == 1:
            for agent_id, act in actions_dict.items():
                if agent_id in self.blue_ids:
                    act[2] = 1.0
                    act[1] = np.clip(act[1], -0.3, 0.3)
                    actions_dict[agent_id] = act

        self.core.step(actions_dict, self.kappa)
        self.core.update_spatial_cache()

        rewards, dones = [], []

        # New: Track total stats for info
        episode_stats = {'missiles_fired': 0, 'cannons_fired': 0, 'kills': 0, 'locked': 0}

        # New: Aggregate breakdown for the step (sum of all agents)
        step_breakdown = {'rew_survival': 0.0, 'rew_pos': 0.0, 'rew_kill': 0.0, 'rew_penalty': 0.0}

        reds_alive = sum(1 for uid in self.red_ids if uid in self.core.entities)
        blues_alive = sum(1 for uid in self.blue_ids if uid in self.core.entities)

        all_enemies_dead = (reds_alive == 0)
        defeat = (blues_alive == 0)
        timeout = (self.core.time >= self.cfg.MAX_DURATION_SEC)

        global_term = all_enemies_dead or defeat
        global_trunc = timeout

        stall_ratio, g_load = 0.0, 0.0

        for i, agent_id in enumerate(self.blue_ids):
            act = actions_dict.get(agent_id, np.zeros(5))

            # CHANGED: Unpack 5 values now
            rew, term, reason, stats, breakdown = self._calculate_reward(agent_id, all_enemies_dead, timeout, act)

            rewards.append(rew)
            dones.append(term or global_term or global_trunc)

            # Aggregate stats
            for k in episode_stats: episode_stats[k] += stats[k]
            for k in step_breakdown: step_breakdown[k] += breakdown[k]

            if agent_id in self.core.entities:
                agent = self.core.entities[agent_id]
                if agent.speed < 150.0:
                    stall_ratio = np.clip((150.0 - agent.speed) / 50.0, 0.0, 1.0)
                else:
                    stall_ratio = 0.0
                g_load = agent.g_load

        term_reason = "none"
        if all_enemies_dead:
            term_reason = "win" if episode_stats['kills'] > 0 else "win_passive"
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

            # Stats passed to train.py
            "stat_missiles_fired": int(episode_stats['missiles_fired']),
            "stat_cannons_fired": int(episode_stats['cannons_fired']),
            "stat_kills": int(episode_stats['kills']),
            "stat_locked": int(episode_stats['locked']),

            # New: Reward Breakdown passed to train.py
            "reward_breakdown": step_breakdown
        }

        return self._get_all_blue_obs(), np.array(rewards, dtype=np.float32), global_term, global_trunc, info

    def _get_graph_state(self):
        active_uids = []
        node_feats = []
        for uid, e in self.core.entities.items():
            active_uids.append(uid)
            xn, yn = self.map_limits.relative_position(e.x, e.y)
            zn = e.alt / 15000.0
            vn = e.speed / 1000.0
            hn = math.radians(e.heading)
            is_missile = 1.0 if e.type == "missile" else 0.0
            is_blue = 1.0 if e.team == "blue" else 0.0

            feat = [xn, yn, zn, math.cos(hn), math.sin(hn), 0.0, vn, e.fuel, e.ammo / 4.0, is_missile, is_blue,
                    e.g_load / 9.0]
            node_feats.append(feat)

        if not node_feats: return None

        edge_index = []
        edge_attr = []
        num_nodes = len(active_uids)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue

                uid_i = active_uids[i]
                uid_j = active_uids[j]

                data = self.core.get_relative_data(uid_i, uid_j)
                if data is None: continue
                dist, rel_pos, rel_vel = data

                ent_i = self.core.entities[uid_i]
                ent_j = self.core.entities[uid_j]

                def get_h_vec(e):
                    h = math.radians(e.heading)
                    p = e.pitch
                    return np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), math.sin(p)])

                vec_i = get_h_vec(ent_i)
                vec_j = get_h_vec(ent_j)
                safe_dist = dist + 1e-5
                vec_i_to_j = rel_pos / safe_dist

                ata_dot = np.clip(np.dot(vec_i, vec_i_to_j), -1, 1)
                ata_deg = math.degrees(math.acos(ata_dot))
                aa_dot = np.clip(np.dot(vec_j, -vec_i_to_j), -1, 1)
                aa_deg = math.degrees(math.acos(aa_dot))
                closing_ms = -np.dot(rel_vel, vec_i_to_j)
                heading_alignment = np.dot(vec_i, vec_j)

                attr = [
                    dist / 50000.0,
                    ata_deg / 180.0,
                    aa_deg / 180.0,
                    heading_alignment,
                    closing_ms / 2000.0,
                    1.0 if ent_i.team == ent_j.team else 0.0
                ]
                edge_index.append([i, j])
                edge_attr.append(attr)

        # --- FINAL CLEAN RETURN ---
        # Handle 0-edge case correctly to preserve (2, N) shape
        if len(edge_index) == 0:
            edge_index_np = np.zeros((2, 0), dtype=np.int64)
            edge_attr_np = np.zeros((0, 6), dtype=np.float32)
        else:
            edge_index_np = np.array(edge_index, dtype=np.int64).T
            edge_attr_np = np.array(edge_attr, dtype=np.float32)

        return {"x": np.array(node_feats, dtype=np.float32),
                "edge_index": edge_index_np,
                "edge_attr": edge_attr_np}

    def _calculate_reward(self, agent_id, win_condition, timeout_condition, action):
        stats = {'missiles_fired': 0, 'cannons_fired': 0, 'kills': 0, 'locked': 0}

        # New: Component tracker
        breakdown = {'rew_survival': 0.0, 'rew_pos': 0.0, 'rew_kill': 0.0, 'rew_penalty': 0.0}

        if agent_id not in self.core.entities:
            if agent_id in self.dead_agent_ids:
                return 0.0, True, "dead", stats, breakdown

            self.dead_agent_ids.add(agent_id)
            ev = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            reason = "shot" if ev and ev['type'] == 'kill' else "crash"

            # Penalty logic
            penalty = -5.0
            breakdown['rew_penalty'] += penalty
            return penalty, True, reason, stats, breakdown

        agent = self.core.entities[agent_id]
        rew = 0.0
        scale = self._get_guidance_scale()

        # 1. Flight Safety (Survival/Penalty)
        if agent.speed > 400.0:
            r = 0.005
            rew += r;
            breakdown['rew_survival'] += r
        elif agent.speed < 200.0:
            r = -0.1
            rew += r;
            breakdown['rew_penalty'] += r

        if 4000 < agent.alt < 8000:
            r = 0.05
            rew += r;
            breakdown['rew_pos'] += r  # Good altitude positioning

        if agent.alt < 2000:
            self.dead_agent_ids.add(agent_id)
            penalty = -10.0
            breakdown['rew_penalty'] += penalty
            return penalty, True, "floor_violation", stats, breakdown

        if agent.alt < 4000:
            r = -0.004
            rew += r;
            breakdown['rew_penalty'] += r

        if agent.pitch < -0.17 and agent.alt < 5000:
            r = -0.01
            rew += r;
            breakdown['rew_penalty'] += r

        # 2. Combat (Positioning & Killing)
        nearest = None
        min_dist_3d = float('inf')

        # Use Spatial Cache
        for rid in self.red_ids:
            if rid in self.core.entities:
                data = self.core.get_relative_data(agent_id, rid)
                if data:
                    d = data[0]
                    if d < min_dist_3d:
                        min_dist_3d = d
                        nearest = self.core.entities[rid]

        if nearest:
            dist_km = min_dist_3d / 1000.0

            # Recalculate vectors for angles
            h_rad = math.radians(agent.heading)
            p_rad = agent.pitch
            ego_vec = np.array([
                math.cos(p_rad) * math.cos(h_rad),
                math.cos(p_rad) * math.sin(h_rad),
                math.sin(p_rad)
            ])

            dx = nearest.x - agent.x
            dy = nearest.y - agent.y
            dz = nearest.alt - agent.alt
            vec_to_tgt = np.array([dx, dy, dz]) / (min_dist_3d + 1e-5)

            dot_prod = np.clip(np.dot(ego_vec, vec_to_tgt), -1.0, 1.0)
            ata_deg = math.degrees(math.acos(dot_prod))

            # Approach Reward (Positioning)
            if agent_id in self.prev_dist:
                delta_km = (self.prev_dist[agent_id] / 1000.0) - dist_km
                if ata_deg < 90.0:
                    r = delta_km * 0.1 * scale
                    rew += r;
                    breakdown['rew_pos'] += r
            self.prev_dist[agent_id] = min_dist_3d

            # Bore Sight (Positioning)
            if ata_deg < 60.0:
                r_bore = (1.0 - (ata_deg / 60.0)) * 0.01 * scale
                rew += r_bore;
                breakdown['rew_pos'] += r_bore

            # Lock (Positioning/Combat)
            is_locking = False
            if dist_km < self.cfg.MISSILE_RANGE_KM:
                _, is_locking = self.core.get_sensor_state(agent_id, nearest.uid)
                if is_locking:
                    r = 0.05 * scale
                    rew += r;
                    breakdown['rew_pos'] += r
                    stats['locked'] = 1

            # Weapons (Combat)
            curr_ammo = agent.ammo
            prev_ammo = self.last_ammo.get(agent_id, curr_ammo)

            if curr_ammo < prev_ammo:
                stats['missiles_fired'] = 1
                if dist_km < self.cfg.MISSILE_RANGE_KM and ata_deg < 25.0 and is_locking:
                    r = 2.0
                    rew += r;
                    breakdown['rew_kill'] += r
                else:
                    r = -0.1
                    rew += r;
                    breakdown['rew_penalty'] += r

            elif action[3] > 0.0:
                if dist_km < self.cfg.CANNON_RANGE_KM:
                    stats['cannons_fired'] = 1
                    if ata_deg < 5.0:
                        r = 0.2
                        rew += r;
                        breakdown['rew_kill'] += r
                else:
                    r = -0.005
                    rew += r;
                    breakdown['rew_penalty'] += r

            self.last_ammo[agent_id] = curr_ammo

        # Kill Logic
        for ev in self.core.events:
            if ev['type'] == 'kill' and ev['killer'] == agent_id:
                r = 5.0
                rew += r;
                breakdown['rew_kill'] += r
                stats['kills'] = 1

        if win_condition and stats['kills'] > 0:
            r = 10.0
            rew += r;
            breakdown['rew_kill'] += r
            return rew, False, "win", stats, breakdown

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

    def _get_obs(self, ego_id):
        vecs = []
        if ego_id in self.core.entities:
            vecs.append(self._vectorize(self.core.entities[ego_id], ego_id, True))
        else:
            vecs.append(np.zeros(self.cfg.FEAT_DIM, dtype=np.float32))

        # --- FIX: OPTIMIZED SORTING USING CACHE ---
        others = []

        for uid, ent in self.core.entities.items():
            if uid == ego_id: continue

            dist_val = 1e9  # Default huge distance

            # Use the spatial cache O(1) lookup
            if ego_id in self.core.entities:
                data = self.core.get_relative_data(ego_id, uid)
                if data is not None:  # <--- Explicit check
                    dist_val = float(data[0])  # Ensure float

            # Fallback (Safety net)
            if dist_val >= 1e9 and ego_id in self.core.entities:
                ego_ent = self.core.entities[ego_id]
                dist_val = math.sqrt((ent.x - ego_ent.x) ** 2 + (ent.y - ego_ent.y) ** 2 + (ent.alt - ego_ent.alt) ** 2)

            others.append((dist_val, uid, ent))

        # Sort: Closest first
        others.sort(key=lambda x: x[0])

        for _, uid, ent in others:
            visible = True

            if ego_id in self.core.entities and ent.team != "blue":
                visible, _ = self.core.get_sensor_state(ego_id, uid)

            # ALLOW INVISIBLE MISSILES IF TARGETING ME (MAWS SIMULATION)
            is_missile_targeting_me = (ent.type == "missile" and ent.target_id == ego_id)

            if visible or is_missile_targeting_me:
                vecs.append(self._vectorize(ent, ego_id, False))
            else:
                vecs.append(np.zeros(self.cfg.FEAT_DIM, dtype=np.float32))

        # Flatten
        flat = []
        for v in vecs: flat.extend(v)

        if len(flat) > self.cfg.OBS_DIM:
            flat = flat[:self.cfg.OBS_DIM]
        if len(flat) < self.cfg.OBS_DIM:
            flat.extend([0.0] * (self.cfg.OBS_DIM - len(flat)))

        return np.array(flat, dtype=np.float32)

    def _get_relative_angles(self, ego, global_rel_vec):
        """
        Rotates global vector into ego's body frame to get Azimuth/Elevation.
        """
        h_rad = math.radians(ego.heading)
        p_rad = ego.pitch

        dx, dy, dz = global_rel_vec

        # 1. Rotate Yaw (Global -> Heading Frame)
        # Global: X=North, Y=East
        # Rotated: Forward=(Cos, Sin), Right=(Sin, -Cos)?
        # Let's use standard rotation matrix
        # North(x), East(y).
        # x_h = x cos(-h) - y sin(-h)
        # y_h = x sin(-h) + y cos(-h)
        # Since h=0 is +X (North), we rotate by -h.

        cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)

        # Forward in horizontal plane
        forward_h = dx * cos_h + dy * sin_h
        # Right in horizontal plane
        right_h = dy * cos_h - dx * sin_h
        up_h = dz

        # 2. Rotate Pitch (Heading Frame -> Body Frame)
        # Rotate around Right axis by -pitch
        cos_p, sin_p = math.cos(p_rad), math.sin(p_rad)

        body_forward = forward_h * cos_p + up_h * sin_p
        body_right = right_h
        body_up = up_h * cos_p - forward_h * sin_p

        # 3. Extract Angles
        azimuth = math.atan2(body_right, body_forward)
        horiz_dist = math.sqrt(body_forward ** 2 + body_right ** 2)
        elevation = math.atan2(body_up, horiz_dist)

        return azimuth, elevation

    def _vectorize(self, e, ego_id, is_ego):
        """
        Creates the Relative/Egocentric feature vector (29 dims).
        """
        # Basic Absolute Features
        xn, yn = self.map_limits.relative_position(e.x, e.y)
        hr = math.radians(e.heading)

        # One-hot ID
        agent_id_oh = [0.0] * self.cfg.MAX_TEAM_SIZE
        if e.team == "blue" and e.uid in self.blue_ids:
            try:
                idx = self.blue_ids.index(e.uid)
                if idx < self.cfg.MAX_TEAM_SIZE:
                    agent_id_oh[idx] = 1.0
            except:
                pass

        # RELATIVE METRICS
        range_norm = 0.0
        az_cos, az_sin = 1.0, 0.0
        el_sin = 0.0
        asp_cos, asp_sin = 1.0, 0.0
        closure_norm = 0.0
        rwr = 0.0
        maws = 0.0

        # --- FIX: ROBUST SENSOR EXTRACTION ---
        # Calculate MAWS/RWR even if spatial cache is missing (Safety)
        if not is_ego:
            # MAWS: Absolute check, no geometry needed
            if e.type == "missile" and e.target_id == ego_id:
                maws = 1.0

            # RWR: Geometric check (re-calcs geometry if needed)
            if ego_id in self.core.entities:
                _, is_locking_me = self.core.get_sensor_state(e.uid, ego_id)
                if is_locking_me:
                    rwr = 1.0

        if not is_ego and ego_id in self.core.entities:
            ego = self.core.entities[ego_id]

            # Use Spatial Cache data for heavy vector math
            data = self.core.get_relative_data(ego_id, e.uid)

            if data is not None:
                dist, rel_pos, rel_vel = data

                # 1. Range
                range_norm = dist / 60000.0  # 60km max

                # 2. Azimuth / Elevation (Body Frame)
                az, el = self._get_relative_angles(ego, rel_pos)
                az_cos = math.cos(az)
                az_sin = math.sin(az)
                el_sin = math.sin(el)

                # 3. Aspect Angle (Relative Heading)
                safe_dist = dist + 1e-5
                u_los = rel_pos / safe_dist

                # Target Heading Vector
                tgt_h = math.radians(e.heading)
                tgt_p = e.pitch
                tgt_vec = np.array([
                    math.cos(tgt_p) * math.cos(tgt_h),
                    math.cos(tgt_p) * math.sin(tgt_h),
                    math.sin(tgt_p)
                ])

                dot_asp = np.clip(np.dot(tgt_vec, -u_los), -1.0, 1.0)
                asp_cos = dot_asp
                asp_sin = math.sqrt(1.0 - asp_cos ** 2)

                # 4. Closure
                closure_val = -np.dot(rel_vel, u_los)
                closure_norm = np.clip(closure_val / 2000.0, -1.0, 1.0)

        return [
            range_norm,  # 0
            az_cos,  # 1
            az_sin,  # 2
            el_sin,  # 3
            asp_cos,  # 4
            asp_sin,  # 5
            closure_norm,  # 6
            e.speed / 1000.0,  # 7
            e.alt / 15000.0,  # 8
            np.cos(hr),  # 9
            np.sin(hr),  # 10
            np.cos(e.pitch),  # 11
            np.sin(e.pitch),  # 12
            np.cos(e.roll),  # 13
            np.sin(e.roll),  # 14
            e.fuel,  # 15
            e.ammo / 4.0,  # 16
            1.0 if e.team == "blue" else -1.0,  # 17
            1.0 if e.type == "missile" else 0.0,  # 18
            1.0 if is_ego else 0.0,  # 19
            rwr,  # 20
            maws,  # 21
            xn,  # 22
            yn,  # 23
            *agent_id_oh  # 24-28
        ]