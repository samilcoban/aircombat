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

        # Training Progress
        self.global_step = 0
        self.total_steps = self.cfg.TOTAL_TIMESTEPS

        self.last_actions = {}
        self.last_ammo = {}
        self.dead_agent_ids = set()
        self.active_locks = set()

    def set_global_step(self, step):
        self.global_step = step

    def _get_guidance_scale(self):
        """
        Linearly decays from 1.0 to 0.0 over the first 3M steps.
        """
        decay_horizon = 3_000_000
        progress = min(1.0, self.global_step / decay_horizon)
        # Decay to 0.0
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

        # --- SPAWNING LOGIC ---
        cx_rel, cy_rel = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        cx, cy = self.map_limits.absolute_position(cx_rel, cy_rel)
        axis = rng.uniform(0.0, 360.0)

        if self.phase == 1:
            # PHASE 1: "The Setup" (Blue Behind Red)
            sep = rng.uniform(2000, 4000)
            bx, by = cx, cy
            b_heading = axis
            b_speed = 600.0

            rx = cx + sep * math.cos(math.radians(axis))
            ry = cy + sep * math.sin(math.radians(axis))
            r_heading = axis
            r_speed = 300.0

        else:
            # PHASE 2+: "The Merge" (Head On)
            sep = rng.uniform(30000.0, 50000.0)
            bx = cx + (sep / 2) * math.cos(math.radians(axis + 180))
            by = cy + (sep / 2) * math.sin(math.radians(axis + 180))
            b_heading = axis
            b_speed = 900.0

            rx = cx + (sep / 2) * math.cos(math.radians(axis))
            ry = cy + (sep / 2) * math.sin(math.radians(axis))
            r_heading = (axis + 180) % 360
            r_speed = 600.0 if self.phase == 2 else 900.0

        for i in range(self.n_agents):
            offset = (i - (self.n_agents - 1) / 2) * 500.0
            bid = self.core.spawn(bx, by, b_heading, b_speed, "blue", "plane")
            self.core.entities[bid].alt = 10000.0
            self.blue_ids.append(bid)
            self.last_ammo[bid] = 4

        n_red = 1
        if self.phase > 3: n_red = rng.integers(1, self.cfg.N_ENEMIES_MAX + 1)

        for i in range(n_red):
            rid = self.core.spawn(rx, ry, r_heading, r_speed, "red", "plane")
            self.core.entities[rid].alt = 10000.0
            self.red_ids.append(rid)

        info = {
            "red_obs": self._get_all_red_obs(),
            "graph_data": self._get_graph_state(),
            "termination_reason": "none",
            "stat_missiles_fired": 0,
            "stat_cannons_fired": 0,  # Added stat
            "stat_kills": 0,
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

        if self.phase <= 3 and self.red_ids:
            for rid in self.red_ids:
                if rid not in actions_dict:
                    turn = 0.0
                    if self.phase == 3 and np.random.rand() < 0.05: turn = np.random.uniform(-0.5, 0.5)
                    actions_dict[rid] = np.array([turn, 0.0, 0.6, 0.0, 0.0])

        self.core.step(actions_dict, self.kappa)

        rewards, dones = [], []
        total_missiles, total_cannons, total_kills = 0, 0, 0

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

            if agent_id in self.core.entities:
                agent = self.core.entities[agent_id]
                stall_ratio = np.clip((150.0 - agent.speed) / 50.0, 0.0, 1.0) if agent.speed < 150.0 else 0.0
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
        }

        return self._get_all_blue_obs(), np.array(rewards, dtype=np.float32), global_term, global_trunc, info

    def _get_graph_state(self):
        active_uids = []
        node_feats = []
        for uid, e in self.core.entities.items():
            active_uids.append(uid)
            xn, yn = self.map_limits.relative_position(e.x, e.y)
            zn = e.alt / 15000.0;
            vn = e.speed / 1000.0;
            hn = math.radians(e.heading)
            is_missile = 1.0 if e.type == "missile" else 0.0
            is_blue = 1.0 if e.team == "blue" else 0.0
            feat = [xn, yn, zn, math.cos(hn), math.sin(hn), 0.0, vn, e.fuel, e.ammo / 4.0, is_missile, is_blue,
                    e.g_load / 9.0]
            node_feats.append(feat)
        if not node_feats: return None
        edge_index = [];
        edge_attr = []
        num_nodes = len(active_uids)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue
                uid_i = active_uids[i];
                uid_j = active_uids[j]
                ent_i = self.core.entities[uid_i];
                ent_j = self.core.entities[uid_j]
                dist = dist_2d(ent_i.x, ent_i.y, ent_j.x, ent_j.y)
                bearing = bearing_deg(ent_i.x, ent_i.y, ent_j.x, ent_j.y)
                ata = abs((bearing - ent_i.heading + 180) % 360 - 180)
                bearing_j_to_i = (bearing + 180) % 360
                aa = abs((bearing_j_to_i - ent_j.heading + 180) % 360 - 180)
                v_closing = (ent_i.speed * math.cos(math.radians(ata)) + ent_j.speed * math.cos(math.radians(aa)))
                attr = [dist / 50000.0, ata / 180.0, aa / 180.0, (ent_i.heading - ent_j.heading) % 360 / 180.0,
                        v_closing / 2000.0, 1.0 if ent_i.team == ent_j.team else 0.0]
                edge_index.append([i, j]);
                edge_attr.append(attr)
        return {"x": np.array(node_feats, dtype=np.float32), "edge_index": np.array(edge_index, dtype=np.int64).T,
                "edge_attr": np.array(edge_attr, dtype=np.float32)}

    def _calculate_reward(self, agent_id, win_condition, timeout_condition, action):
        stats = {'missiles_fired': 0, 'cannons_fired': 0, 'kills': 0}

        # 1. Death Penalty
        if agent_id not in self.core.entities:
            if agent_id in self.dead_agent_ids: return 0.0, True, "dead", stats
            self.dead_agent_ids.add(agent_id)
            ev = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            reason = "shot" if ev and ev['type'] == 'kill' else "crash"
            return -50.0, True, reason, stats

        agent = self.core.entities[agent_id]
        rew = 0.0

        # 2. Linear Scaling for Guidance
        scale = self._get_guidance_scale()

        # 3. Energy Penalty (No Existence Reward)
        if agent.speed < 200.0:
            rew -= 0.05 * (200.0 - agent.speed) / 100.0

        if agent.alt < 2000:
            self.dead_agent_ids.add(agent_id);
            return -100.0, True, "floor", stats

        # 4. Combat Logic
        nearest = None;
        min_dist = float('inf')
        for rid in self.red_ids:
            if rid in self.core.entities:
                d = dist_2d(agent.x, agent.y, self.core.entities[rid].x, self.core.entities[rid].y)
                if d < min_dist: min_dist = d; nearest = self.core.entities[rid]

        if nearest:
            bearing = bearing_deg(agent.x, agent.y, nearest.x, nearest.y)
            ata = abs((bearing - agent.heading + 180) % 360 - 180)
            dist_km = min_dist / 1000.0

            # Guidance (Decaying)
            if ata < 30.0:
                r_bore = (1.0 - (ata / 30.0)) * 0.1 * scale
                rew += r_bore

            if dist_km < self.cfg.MISSILE_RANGE_KM:
                _, is_locking = self.core.get_sensor_state(agent_id, nearest.uid)
                if is_locking:
                    rew += 0.05 * scale

                    # Weapon Usage
            curr_ammo = agent.ammo
            prev_ammo = self.last_ammo.get(agent_id, curr_ammo)

            if curr_ammo < prev_ammo:
                stats['missiles_fired'] = 1
                if dist_km < self.cfg.MISSILE_RANGE_KM and ata < 20.0 and is_locking:
                    rew += 5.0  # Valid Missile
                else:
                    rew -= 0.5  # Waste

            elif action[3] > 0.0:
                # Cannon Logic
                if dist_km < self.cfg.CANNON_RANGE_KM:
                    stats['cannons_fired'] = 1
                    if ata < 5.0: rew += 2.0  # Valid Burst
                else:
                    rew -= 0.05  # Trigger Discipline

            self.last_ammo[agent_id] = curr_ammo

        # 5. Kills
        for ev in self.core.events:
            if ev['type'] == 'kill' and ev['killer'] == agent_id:
                rew += 50.0;
                stats['kills'] = 1

        # 6. Win
        if win_condition and stats['kills'] > 0:
            rew += 100.0
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
            visible, _ = True, False
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
        xn, yn = self.map_limits.relative_position(e.x, e.y)
        hr = math.radians(e.heading)
        agent_id_oh = [0.0] * self.cfg.MAX_TEAM_SIZE
        if e.team == "blue" and e.uid in self.blue_ids:
            try:
                idx = self.blue_ids.index(e.uid); agent_id_oh[idx] = 1.0 if idx < self.cfg.MAX_TEAM_SIZE else 0.0
            except:
                pass
        ata_norm, aa_norm, closure = 0.0, 0.0, 0.0
        if not is_ego and ego_id in self.core.entities:
            ego = self.core.entities[ego_id]
            bearing = bearing_deg(ego.x, ego.y, e.x, e.y)
            ata = abs((bearing - ego.heading + 180) % 360 - 180);
            ata_norm = ata / 180.0
            aa = abs(((bearing + 180) % 360 - e.heading + 180) % 360 - 180);
            aa_norm = aa / 180.0
            v_ego = ego.speed * math.cos(math.radians(ata));
            v_tgt = e.speed * math.cos(math.radians(aa))
            closure = np.clip((v_ego + v_tgt) / 2000.0, -1.0, 1.0)
        is_locked_by_me = 0.0
        if not is_ego and ego_id in self.core.entities:
            _, is_locking = self.core.get_sensor_state(ego_id, e.uid)
            if is_locking: is_locked_by_me = 1.0
        return [xn, yn, np.cos(hr), np.sin(hr), e.speed / 1000.0, 1.0 if e.team == "blue" else -1.0,
                1.0 if e.type == "missile" else 0.0, 1.0 if is_ego else 0.0, np.cos(e.roll), np.sin(e.roll),
                np.cos(e.pitch), np.sin(e.pitch), 0.0, 0.0, e.alt / 10000.0, e.fuel, e.ammo / 4.0, ata_norm, aa_norm,
                closure, is_locked_by_me, *agent_id_oh]