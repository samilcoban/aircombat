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
        # Linearly decays guidance rewards from 1.0 to 0.0 over 3M steps
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

        # --- SPAWNING LOGIC ---
        cx_rel, cy_rel = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        cx, cy = self.map_limits.absolute_position(cx_rel, cy_rel)
        axis = rng.uniform(0.0, 360.0)

        # Default Altitudes
        spawn_alt = 5000.0

        if self.phase == 1:
            # PHASE 1: "The Intercept"
            # Spawn CLOSER (8-12km) so Lock is immediate.
            sep = rng.uniform(8000.0, 12000.0)

            rx, ry = cx, cy
            r_heading = axis
            r_speed = 400.0

            bx = cx - sep * math.cos(math.radians(axis))
            by = cy - sep * math.sin(math.radians(axis))

            # Reduce heading error to make it easier to find the target
            heading_error = rng.uniform(-20.0, 20.0)
            b_heading = (axis + heading_error) % 360.0
            b_speed = 600.0

        else:
            # PHASE 2+: "The Merge"
            sep = rng.uniform(30000.0, 50000.0)
            bx = cx + (sep / 2) * math.cos(math.radians(axis + 180))
            by = cy + (sep / 2) * math.sin(math.radians(axis + 180))
            b_heading = axis
            b_speed = 900.0

            rx = cx + (sep / 2) * math.cos(math.radians(axis))
            ry = cy + (sep / 2) * math.sin(math.radians(axis))
            r_heading = (axis + 180) % 360
            r_speed = 600.0 if self.phase == 2 else 900.0

        # Offset calculation for wingmen
        perp_rad = math.radians(b_heading + 90)
        off_x_unit = math.cos(perp_rad)
        off_y_unit = math.sin(perp_rad)

        for i in range(self.n_agents):
            offset_dist = (i - (self.n_agents - 1) / 2.0) * 500.0
            spawn_x = bx + off_x_unit * offset_dist
            spawn_y = by + off_y_unit * offset_dist

            bid = self.core.spawn(spawn_x, spawn_y, b_heading, b_speed, "blue", "plane")
            self.core.entities[bid].alt = spawn_alt
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

            rid = self.core.spawn(spawn_rx, spawn_ry, r_heading, r_speed, "red", "plane")
            self.core.entities[rid].alt = spawn_alt
            self.red_ids.append(rid)

        # Initialize Previous Distance (Using 3D Distance to avoid reward spikes)
        if self.blue_ids and self.red_ids:
            b = self.core.entities[self.blue_ids[0]]
            r = self.core.entities[self.red_ids[0]]
            dx = b.x - r.x
            dy = b.y - r.y
            dz = b.alt - r.alt
            self.prev_dist[self.blue_ids[0]] = math.sqrt(dx * dx + dy * dy + dz * dz)

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

        # AI Logic for Red Team (if not provided by Self-Play)
        if self.phase <= 3 and self.red_ids:
            for rid in self.red_ids:
                if rid not in actions_dict:
                    turn = 0.0
                    if self.phase == 3 and np.random.rand() < 0.05: turn = np.random.uniform(-0.5, 0.5)
                    actions_dict[rid] = np.array([turn, 0.0, 0.6, 0.0, 0.0])

        self.core.step(actions_dict, self.kappa)

        rewards, dones = [], []
        total_missiles, total_cannons, total_kills, total_locks = 0, 0, 0, 0

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
            rew, term, reason, stats = self._calculate_reward(agent_id, all_enemies_dead, timeout, act)
            rewards.append(rew)
            dones.append(term or global_term or global_trunc)

            total_missiles += stats['missiles_fired']
            total_cannons += stats['cannons_fired']
            total_kills += stats['kills']
            total_locks += stats['locked']

            if agent_id in self.core.entities:
                agent = self.core.entities[agent_id]
                # Check stall condition for logging
                if agent.speed < 150.0:
                    stall_ratio = np.clip((150.0 - agent.speed) / 50.0, 0.0, 1.0)
                else:
                    stall_ratio = 0.0
                g_load = agent.g_load

        term_reason = "none"
        if all_enemies_dead:
            term_reason = "win" if total_kills > 0 else "win_passive"
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
            "stat_missiles_fired": int(total_missiles),
            "stat_cannons_fired": int(total_cannons),
            "stat_kills": int(total_kills),
            "stat_locked": int(total_locks),
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
                ent_i = self.core.entities[uid_i]
                ent_j = self.core.entities[uid_j]

                # --- 3D EDGE FEATURES ---

                # 1. 3D Distance
                dx = ent_i.x - ent_j.x
                dy = ent_i.y - ent_j.y
                dz = ent_i.alt - ent_j.alt
                dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

                # 2. 3D Unit Vectors
                vec_i_to_j = np.array([ent_j.x - ent_i.x, ent_j.y - ent_i.y, ent_j.alt - ent_i.alt])
                vec_i_to_j /= (dist_3d + 1e-5)

                # Heading Vectors (X=North, Y=East, Z=Up)
                def get_h_vec(e):
                    h = math.radians(e.heading)
                    p = e.pitch
                    return np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), math.sin(p)])

                vec_i = get_h_vec(ent_i)
                vec_j = get_h_vec(ent_j)

                # 3. 3D Angles
                # ATA (Angle i looking at j)
                ata_dot = np.clip(np.dot(vec_i, vec_i_to_j), -1, 1)
                ata_deg = math.degrees(math.acos(ata_dot))

                # AA (Angle j looking at i)
                # vec_j dot (vec_j_to_i) -> vec_j dot (-vec_i_to_j)
                aa_dot = np.clip(np.dot(vec_j, -vec_i_to_j), -1, 1)
                aa_deg = math.degrees(math.acos(aa_dot))

                # 4. Closing Speed (Projected)
                k2ms = 0.514444
                vel_i = vec_i * (ent_i.speed * k2ms)
                vel_j = vec_j * (ent_j.speed * k2ms)
                rel_vel = vel_i - vel_j
                closing_ms = np.dot(rel_vel, vec_i_to_j)

                # 5. Heading Diff (Scalar)
                # Dot product of heading vectors gives orientation difference
                # 1.0 = Same direction, -1.0 = Head on
                heading_alignment = np.dot(vec_i, vec_j)

                # Construct Attribute Vector (6 dims)
                # [Dist, ATA, AA, Alignment, Closure, Team]
                attr = [
                    dist_3d / 50000.0,
                    ata_deg / 180.0,
                    aa_deg / 180.0,
                    heading_alignment,
                    closing_ms / 2000.0,
                    1.0 if ent_i.team == ent_j.team else 0.0
                ]

                edge_index.append([i, j])
                edge_attr.append(attr)

        return {"x": np.array(node_feats, dtype=np.float32),
                "edge_index": np.array(edge_index, dtype=np.int64).T,
                "edge_attr": np.array(edge_attr, dtype=np.float32)}

    def _calculate_reward(self, agent_id, win_condition, timeout_condition, action):
        stats = {'missiles_fired': 0, 'cannons_fired': 0, 'kills': 0, 'locked': 0}

        # 1. Death Penalty
        if agent_id not in self.core.entities:
            if agent_id in self.dead_agent_ids:
                return 0.0, True, "dead", stats

            self.dead_agent_ids.add(agent_id)
            ev = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            reason = "shot" if ev and ev['type'] == 'kill' else "crash"
            return -5.0, True, reason, stats

        agent = self.core.entities[agent_id]
        rew = 0.0
        scale = self._get_guidance_scale()

        # 2. Flight Safety Penalties
        # ---------------------------------------------------------
        # Stall Penalty: If speed drops below 150 kts
        if agent.speed < 150.0:
            rew -= 0.05 * (150.0 - agent.speed) / 50.0

        # Hard Deck (Immediate Death)
        if agent.alt < 2000:
            self.dead_agent_ids.add(agent_id)
            return -10.0, True, "floor_violation", stats

        # Soft Deck (Warning Zone) - Encourage flying above 4km
        if agent.alt < 4000:
            rew -= 0.005

        # Diving Penalty: Prevent lawn-darting
        # If diving steeper than -10 deg (-0.17 rad) while low
        if agent.pitch < -0.17 and agent.alt < 5000:
            rew -= 0.01

        # 3. Combat Logic (3D)
        # ---------------------------------------------------------
        nearest = None
        min_dist_3d = float('inf')

        # Find nearest enemy using True 3D Distance
        for rid in self.red_ids:
            if rid in self.core.entities:
                enemy = self.core.entities[rid]
                dx = enemy.x - agent.x
                dy = enemy.y - agent.y
                dz = enemy.alt - agent.alt
                d_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

                if d_3d < min_dist_3d:
                    min_dist_3d = d_3d
                    nearest = enemy

        if nearest:
            dist_km = min_dist_3d / 1000.0

            # --- 3D Vector Math for Angle-off-Boresight (ATA) ---

            # A. Calculate Ego Boresight Vector (Where nose is pointing)
            # Convention: X=North, Y=East, Z=Up
            h_rad = math.radians(agent.heading)
            p_rad = agent.pitch
            ego_vec = np.array([
                math.cos(p_rad) * math.cos(h_rad),
                math.cos(p_rad) * math.sin(h_rad),
                math.sin(p_rad)
            ])

            # B. Calculate Vector to Target
            vec_to_tgt = np.array([
                nearest.x - agent.x,
                nearest.y - agent.y,
                nearest.alt - agent.alt
            ])

            # Normalize vector to target
            vec_to_tgt = vec_to_tgt / (min_dist_3d + 1e-5)

            # C. Calculate Angle (Dot Product)
            # Dot = |a||b|cos(theta). Since vectors are normalized, Dot = cos(theta)
            dot_prod = np.clip(np.dot(ego_vec, vec_to_tgt), -1.0, 1.0)
            ata_deg = math.degrees(math.acos(dot_prod))

            # --- Rewards ---

            # Approach Reward (Shaping)
            if agent_id in self.prev_dist:
                delta_km = (self.prev_dist[agent_id] / 1000.0) - dist_km
                # Only reward approach if we are generally facing the target (<90 deg)
                if ata_deg < 90.0:
                    rew += delta_km * 0.1 * scale
            self.prev_dist[agent_id] = min_dist_3d

            # Bore Sight Reward (Shaping)
            # Reward keeping enemy in HUD cone (60 deg), peaking at 0 deg
            if ata_deg < 60.0:
                r_bore = (1.0 - (ata_deg / 60.0)) * 0.01 * scale
                rew += r_bore

            # Lock Maintenance Reward
            # Relies on updated 3D sensor logic in Core
            is_locking = False
            if dist_km < self.cfg.MISSILE_RANGE_KM:
                _, is_locking = self.core.get_sensor_state(agent_id, nearest.uid)
                if is_locking:
                    rew += 0.05 * scale
                    stats['locked'] = 1

            # Weapon Usage
            curr_ammo = agent.ammo
            prev_ammo = self.last_ammo.get(agent_id, curr_ammo)

            if curr_ammo < prev_ammo:
                stats['missiles_fired'] = 1
                # Jackpot: Good Shot Geometry
                # 3D Distance < Range, 3D ATA < 25 deg, Locked
                if dist_km < self.cfg.MISSILE_RANGE_KM and ata_deg < 25.0 and is_locking:
                    rew += 2.0
                else:
                    rew -= 0.1  # Waste

            elif action[3] > 0.0:
                # Cannon usage
                if dist_km < self.cfg.CANNON_RANGE_KM:
                    stats['cannons_fired'] = 1
                    # Cannon requires high precision (5 deg cone)
                    if ata_deg < 5.0:
                        rew += 0.2
                else:
                    rew -= 0.005  # Spray and pray penalty

            self.last_ammo[agent_id] = curr_ammo

        # 4. Kills & Wins
        for ev in self.core.events:
            if ev['type'] == 'kill' and ev['killer'] == agent_id:
                rew += 5.0
                stats['kills'] = 1

        if win_condition and stats['kills'] > 0:
            rew += 10.0
            return rew, False, "win", stats

        return rew, False, "none", stats

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

        for uid, ent in self.core.entities.items():
            if uid == ego_id: continue

            # Simple Sensor Check: "Fog of War"
            # Blue sees Red only if sensors detect them
            # Red sees Blue only if sensors detect them
            visible = True
            if ego_id in self.core.entities and ent.team != "blue":
                visible, _ = self.core.get_sensor_state(ego_id, uid)

            if visible:
                vecs.append(self._vectorize(ent, ego_id, False))
            else:
                vecs.append(np.zeros(self.cfg.FEAT_DIM, dtype=np.float32))

        flat = []
        for v in vecs: flat.extend(v)
        if len(flat) < self.cfg.OBS_DIM: flat.extend([0.0] * (self.cfg.OBS_DIM - len(flat)))
        return np.array(flat[:self.cfg.OBS_DIM], dtype=np.float32)

    def _vectorize(self, e, ego_id, is_ego):
        """
        Creates the feature vector for an entity.
        CRITICAL: Uses 3D Vector Math to ensure the agent understands verticality.
        """
        xn, yn = self.map_limits.relative_position(e.x, e.y)
        hr = math.radians(e.heading)

        # One-hot ID logic
        agent_id_oh = [0.0] * self.cfg.MAX_TEAM_SIZE
        if e.team == "blue" and e.uid in self.blue_ids:
            try:
                idx = self.blue_ids.index(e.uid)
                agent_id_oh[idx] = 1.0 if idx < self.cfg.MAX_TEAM_SIZE else 0.0
            except:
                pass

        ata_norm, aa_norm, closure_norm = 0.0, 0.0, 0.0

        if not is_ego and ego_id in self.core.entities:
            ego = self.core.entities[ego_id]

            # --- 3D MATH START ---

            # 1. Position Vectors
            pos_ego = np.array([ego.x, ego.y, ego.alt])
            pos_tgt = np.array([e.x, e.y, e.alt])
            vec_to_tgt = pos_tgt - pos_ego
            dist_3d = np.linalg.norm(vec_to_tgt) + 1e-5
            unit_to_tgt = vec_to_tgt / dist_3d

            # 2. Ego Heading Vector (X=North, Y=East, Z=Up)
            ego_h = math.radians(ego.heading)
            ego_p = ego.pitch
            ego_vec = np.array([
                math.cos(ego_p) * math.cos(ego_h),
                math.cos(ego_p) * math.sin(ego_h),
                math.sin(ego_p)
            ])

            # 3. Target Heading Vector
            tgt_h = math.radians(e.heading)
            tgt_p = e.pitch
            tgt_vec = np.array([
                math.cos(tgt_p) * math.cos(tgt_h),
                math.cos(tgt_p) * math.sin(tgt_h),
                math.sin(tgt_p)
            ])

            # 4. Calculate ATA (Angle Off Boresight for Ego) - 3D
            # Cos(theta) = Dot(A, B)
            dot_ata = np.clip(np.dot(ego_vec, unit_to_tgt), -1.0, 1.0)
            ata_deg = math.degrees(math.acos(dot_ata))
            ata_norm = ata_deg / 180.0

            # 5. Calculate AA (Aspect Angle for Target) - 3D
            # Angle between Target's Tail and vector to Ego
            unit_to_ego = -unit_to_tgt
            dot_aa = np.clip(np.dot(tgt_vec, unit_to_ego), -1.0, 1.0)
            aa_deg = math.degrees(math.acos(dot_aa))
            aa_norm = aa_deg / 180.0

            # 6. Calculate Closure Rate (3D)
            # Project 3D velocity vectors onto the 3D Line of Sight
            k2ms = 0.514444
            v_ego_3d = ego_vec * (ego.speed * k2ms)
            v_tgt_3d = tgt_vec * (e.speed * k2ms)

            rel_vel = v_ego_3d - v_tgt_3d
            closure_ms = np.dot(rel_vel, unit_to_tgt)  # Positive = closing
            closure_norm = np.clip(closure_ms / 1400.0, -1.0, 1.0)

            # --- 3D MATH END ---

        is_locked_by_me = 0.0
        if not is_ego and ego_id in self.core.entities:
            _, is_locking = self.core.get_sensor_state(ego_id, e.uid)
            if is_locking: is_locked_by_me = 1.0

        return [
            xn, yn,
            np.cos(hr), np.sin(hr),
            e.speed / 1000.0,
            1.0 if e.team == "blue" else -1.0,
            1.0 if e.type == "missile" else 0.0,
            1.0 if is_ego else 0.0,
            np.cos(e.roll), np.sin(e.roll),
            np.cos(e.pitch), np.sin(e.pitch),
            0.0, 0.0,  # Placeholders for sensor signals if needed later
            e.alt / 15000.0,
            e.fuel,
            e.ammo / 4.0,
            ata_norm,
            aa_norm,
            closure_norm,
            is_locked_by_me,
            *agent_id_oh
        ]