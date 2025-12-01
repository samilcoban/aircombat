# ================================================
# FILE: src/env_flat.py
# ================================================
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

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
        self.last_actions = {}
        self.last_ammo = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.core = AirCombatCore()
        self.blue_ids = []
        self.red_ids = []
        self.last_actions = {}
        self.last_ammo = {}

        cx_rel, cy_rel = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        cx, cy = self.map_limits.absolute_position(cx_rel, cy_rel)
        axis = rng.uniform(0.0, 360.0)

        if self.phase <= 2:
            sep = rng.uniform(5000, 8000)
        else:
            sep = rng.uniform(40000.0, 60000.0)

        for i in range(self.n_agents):
            offset = (i - (self.n_agents - 1) / 2) * 500.0
            bx = cx + (sep / 2) * math.cos(math.radians(axis + 180)) + offset * math.sin(math.radians(axis))
            by = cy + (sep / 2) * math.sin(math.radians(axis + 180)) - offset * math.cos(math.radians(axis))
            spd = 600.0 if self.phase <= 2 else 900.0
            bid = self.core.spawn(bx, by, axis, spd, "blue", "plane")
            self.core.entities[bid].alt = 10000.0
            self.blue_ids.append(bid)
            self.last_actions[bid] = np.zeros(self.cfg.ACTION_DIM)
            self.last_ammo[bid] = 4

        n_red = 1 if self.phase <= 2 else self.cfg.N_ENEMIES
        for i in range(n_red):
            offset = (i - (n_red - 1) / 2) * 500.0
            rx = cx + (sep / 2) * math.cos(math.radians(axis)) + offset * math.sin(math.radians(axis + 180))
            ry = cy + (sep / 2) * math.sin(math.radians(axis)) - offset * math.cos(math.radians(axis + 180))
            spd = 300.0 if self.phase == 1 else (600.0 if self.phase == 2 else 900.0)
            rid = self.core.spawn(rx, ry, (axis + 180) % 360, spd, "red", "plane")
            self.core.entities[rid].alt = 10000.0 if self.phase > 1 else 5000.0
            self.red_ids.append(rid)

        info = {
            "red_obs": self._get_all_red_obs(),
            "global_state": self._get_global_state(),
            "termination_reason": "none",
            "agent_dones": np.zeros(self.n_agents, dtype=bool),
            "physics_stall_ratio": 0.0,
            "physics_g": 1.0,
            "stat_missiles_fired": 0,
            "stat_kills": 0,
            "rew_existence": 0.0
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

        if self.phase <= 2 and self.red_ids:
            for rid in self.red_ids:
                if rid not in actions_dict: actions_dict[rid] = np.array([0.0, 0.0, 0.6, 0.0, 0.0])

        self.core.step(actions_dict, self.kappa)

        rewards, dones = [], []
        total_fired, total_kills = 0, 0

        reds_alive = sum(1 for uid in self.red_ids if uid in self.core.entities)
        blues_alive = sum(1 for uid in self.blue_ids if uid in self.core.entities)

        win = bool(reds_alive == 0)
        defeat = bool(blues_alive == 0)
        timeout = bool(self.core.time >= self.cfg.MAX_DURATION_SEC)
        global_term = win or defeat
        global_trunc = timeout

        stall_ratio, g_load = 0.0, 0.0
        comps = {'existence': 0.0}

        for i, agent_id in enumerate(self.blue_ids):
            rew, term, reason, comps, stats = self._calculate_reward(agent_id, win, timeout)
            rewards.append(rew)
            dones.append(term or global_term or global_trunc)
            total_fired += stats['fired']
            total_kills += stats['kills']

            if agent_id in self.core.entities:
                agent = self.core.entities[agent_id]
                stall_ratio = np.clip((agent.speed - 100.0) / 50.0, 0.0, 1.0)
                g_load = agent.g_load

        term_reason = "win" if win else ("timeout" if timeout else ("crash" if defeat else "none"))

        info = {
            "termination_reason": term_reason,
            "red_obs": self._get_all_red_obs(),
            "global_state": self._get_global_state(),
            "agent_dones": np.array(dones, dtype=bool),
            "physics_stall_ratio": float(stall_ratio),
            "physics_g": float(g_load),
            "stat_missiles_fired": int(total_fired),
            "stat_kills": int(total_kills),
            "rew_existence": float(comps['existence']),
        }

        return self._get_all_blue_obs(), np.array(rewards, dtype=np.float32), global_term, global_trunc, info

    def _get_all_blue_obs(self):
        return np.stack([self._get_obs(uid) for uid in self.blue_ids]).astype(np.float32)

    def _get_all_red_obs(self):
        if not self.red_ids: return np.zeros((1, self.cfg.OBS_DIM), dtype=np.float32)
        return np.stack([self._get_obs(uid) if uid in self.core.entities else np.zeros(self.cfg.OBS_DIM) for uid in
                         self.red_ids]).astype(np.float32)

    def _calculate_reward(self, agent_id, win_condition, timeout_condition):
        comps = {'existence': 0, 'instructor': 0, 'penalty': 0, 'guidance': 0, 'combat': 0}
        stats = {'fired': 0, 'kills': 0}

        if agent_id not in self.core.entities:
            ev = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            reason = "shot" if ev and ev['type'] == 'kill' else "crash"
            return -50.0, True, reason, comps, stats

        agent = self.core.entities[agent_id]
        rew = 0.0

        if self.phase in [1, 2]:
            rew += 0.02;
            comps['existence'] += 0.02
            alt_score = math.exp(-((agent.alt - 6000.0) ** 2) / (2 * 1000 ** 2))
            spd_score = math.exp(-((agent.speed - 600.0) ** 2) / (2 * 100 ** 2))
            r_inst = (alt_score + spd_score) * 0.05
            rew += r_inst;
            comps['instructor'] += r_inst
        else:
            rew -= 0.005;
            comps['existence'] -= 0.005

        r_pen = 0
        if agent.speed < 250.0: r_pen -= (250.0 - agent.speed) * 0.001

        # FIX: Raised G Threshold to 6.0
        if agent.g_load > 6.0: r_pen -= (0.005 * (agent.g_load ** 2))

        if agent.alt < 2000: return -50.0, True, "floor", comps, stats

        rew += r_pen;
        comps['penalty'] += r_pen

        nearest = None
        min_dist = float('inf')
        for rid in self.red_ids:
            if rid in self.core.entities:
                d = dist_2d(agent.x, agent.y, self.core.entities[rid].x, self.core.entities[rid].y)
                if d < min_dist: min_dist = d; nearest = self.core.entities[rid]

        if nearest:
            bearing = bearing_deg(agent.x, agent.y, nearest.x, nearest.y)
            ata = abs((bearing - agent.heading + 180) % 360 - 180)
            dist_km = min_dist / 1000.0

            if ata < 60.0:
                r_guide = (1.0 - (ata / 60.0)) * 0.05
                rew += r_guide;
                comps['guidance'] += r_guide
                if dist_km < self.cfg.MISSILE_RANGE_KM:
                    _, is_locking = self.core.get_sensor_state(agent_id, nearest.uid)
                    if is_locking:
                        rew += 0.1;
                        comps['guidance'] += 0.1

            curr_ammo = agent.ammo
            prev_ammo = self.last_ammo.get(agent_id, curr_ammo)
            if curr_ammo < prev_ammo:
                stats['fired'] = 1
                if dist_km < self.cfg.MISSILE_RANGE_KM and ata < 20.0:
                    rew += 2.0;
                    comps['combat'] += 2.0
                else:
                    rew -= 0.5;
                    comps['combat'] -= 0.5
            self.last_ammo[agent_id] = curr_ammo

        for ev in self.core.events:
            if ev['type'] == 'kill' and ev['killer'] == agent_id:
                rew += 50.0;
                comps['combat'] += 50.0;
                stats['kills'] = 1

        if win_condition:
            rew += 50.0;
            comps['combat'] += 50.0
            return rew, False, "win", comps, stats

        return rew, False, "none", comps, stats

    def _vectorize(self, e, ego_id, is_ego):
        xn, yn = self.map_limits.relative_position(e.x, e.y)
        hr = math.radians(e.heading)

        agent_id_oh = [0.0] * self.cfg.MAX_TEAM_SIZE
        if e.team == "blue" and e.uid in self.blue_ids:
            try:
                idx = self.blue_ids.index(e.uid)
                if idx < self.cfg.MAX_TEAM_SIZE: agent_id_oh[idx] = 1.0
            except:
                pass

        ata_norm, aa_norm, closure = 0.0, 0.0, 0.0
        if not is_ego and ego_id in self.core.entities:
            ego = self.core.entities[ego_id]
            bearing = bearing_deg(ego.x, ego.y, e.x, e.y)
            ata = abs((bearing - ego.heading + 180) % 360 - 180)
            ata_norm = ata / 180.0
            aa = abs(((bearing + 180) % 360 - e.heading + 180) % 360 - 180)
            aa_norm = aa / 180.0
            v_ego = ego.speed * math.cos(math.radians(ata))
            v_tgt = e.speed * math.cos(math.radians(aa))
            closure = np.clip((v_ego + v_tgt) / 2000.0, -1.0, 1.0)

        return [
            xn, yn, np.cos(hr), np.sin(hr), e.speed / 1000.0,
            1.0 if e.team == "blue" else -1.0,
            1.0 if e.type == "missile" else 0.0,
            1.0 if is_ego else 0.0,
            np.cos(e.roll), np.sin(e.roll),
            np.cos(e.pitch), np.sin(e.pitch),
            0.0, 0.0,
                                            e.alt / 10000.0, e.fuel, e.ammo / 4.0,
            ata_norm, aa_norm, closure,
            *agent_id_oh
        ]

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

    def _get_global_state(self):
        flat = []
        for e in self.core.entities.values():
            flat.extend(self._vectorize(e, None, False))
        if len(flat) < self.cfg.OBS_DIM: flat.extend([0.0] * (self.cfg.OBS_DIM - len(flat)))
        return np.array(flat[:self.cfg.OBS_DIM], dtype=np.float32)