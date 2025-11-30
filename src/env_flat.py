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

        center_x = (self.map_limits.min_x + self.map_limits.max_x) / 2.0
        center_y = (self.map_limits.min_y + self.map_limits.max_y) / 2.0
        zoom = 15000.0
        self.render_limits = MapLimits(center_x - zoom, center_x + zoom, center_y - zoom, center_y + zoom)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.OBS_DIM,), dtype=np.float32)

        self.blue_ids = []
        self.red_ids = []
        self.phase = 1
        self.kappa = 0.0
        self.last_action = np.zeros(self.cfg.ACTION_DIM)
        self.last_ammo = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.core = AirCombatCore()
        self.blue_ids = []
        self.red_ids = []
        self.last_action = np.zeros(self.cfg.ACTION_DIM)
        self.last_ammo = {}

        # === SPAWNING LOGIC ===
        sitting_duck = (self.phase in [2, 3]) and (rng.random() < 0.2)

        if self.phase in [1, 2] or sitting_duck:
            # Flight School / Pursuit
            cx_rel, cy_rel = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
            cx, cy = self.map_limits.absolute_position(cx_rel, cy_rel)
            head = rng.uniform(0.0, 360.0)

            if sitting_duck:
                r_dist, r_spd, b_spd = 3000.0, 300.0, 600.0
            else:
                r_dist = rng.uniform(5000, 8000)
                r_spd = 300.0 if self.phase == 1 else 600.0
                b_spd = 600.0

            rx = cx + r_dist * math.cos(math.radians(head))
            ry = cy + r_dist * math.sin(math.radians(head))

            rid = self.core.spawn(rx, ry, head, r_spd, "red", "plane")
            self.core.entities[rid].alt = 5000.0
            self.red_ids.append(rid)

            bid = self.core.spawn(cx, cy, head, b_spd, "blue", "plane")
            self.core.entities[bid].alt = 5000.0
            self.blue_ids.append(bid)
        else:
            # Combat
            cx_rel, cy_rel = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
            cx, cy = self.map_limits.absolute_position(cx_rel, cy_rel)
            sep = rng.uniform(40000.0, 60000.0)
            axis = rng.uniform(0.0, 360.0)

            bx = cx + (sep / 2) * math.cos(math.radians(axis + 180))
            by = cy + (sep / 2) * math.sin(math.radians(axis + 180))
            rx = cx + (sep / 2) * math.cos(math.radians(axis))
            ry = cy + (sep / 2) * math.sin(math.radians(axis))

            bid = self.core.spawn(bx, by, axis, 900.0, "blue", "plane")
            self.core.entities[bid].alt = 10000.0
            self.blue_ids.append(bid)

            rid = self.core.spawn(rx, ry, (axis + 180) % 360, 900.0, "red", "plane")
            self.core.entities[rid].alt = 10000.0
            self.red_ids.append(rid)

        for uid in self.blue_ids:
            if uid in self.core.entities:
                self.last_ammo[uid] = self.core.entities[uid].ammo

        info = {
            "red_obs": self._get_obs(self.red_ids[0]) if self.red_ids else np.zeros(self.cfg.OBS_DIM),
            "global_state": self._get_global_state()
        }
        return self._get_obs(self.blue_ids[0]), info

    def set_phase(self, phase_id, progress=0.0):
        self.phase = phase_id

    def set_kappa(self, k):
        self.kappa = k

    def step(self, action, red_actions=None):
        actions = {}
        agent_id = self.blue_ids[0]

        if len(action.shape) > 0 and action.shape[0] == 2 * self.cfg.ACTION_DIM:
            actions[agent_id] = action[:self.cfg.ACTION_DIM]
            if self.red_ids: actions[self.red_ids[0]] = action[self.cfg.ACTION_DIM:]
        else:
            actions[agent_id] = action
            if red_actions is not None and self.red_ids:
                if isinstance(red_actions, (np.ndarray, list)):
                    actions[self.red_ids[0]] = red_actions
                elif isinstance(red_actions, dict):
                    actions.update(red_actions)

        if self.phase == 1 and self.red_ids and self.red_ids[0] not in actions:
            actions[self.red_ids[0]] = np.array([0.0, 0.0, 0.6, 0.0, 0.0])

        self.core.step(actions, self.kappa)

        agent = self.core.entities.get(agent_id)
        if agent:
            if agent.alt < 2000.0:
                return self._get_result(agent_id, reward=-10.0, term=True, reason="floor_violation")

        reward, term, trunc, reason, components = self._calculate_reward(agent_id, actions)

        info = {
            "termination_reason": reason,
            "red_obs": self._get_obs(self.red_ids[0]) if self.red_ids and self.red_ids[
                0] in self.core.entities else np.zeros(self.cfg.OBS_DIM),
            "global_state": self._get_global_state(),
            "physics_alt": agent.alt if agent else 0.0,
            "physics_speed": agent.speed if agent else 0.0,
            "physics_fuel": agent.fuel if agent else 0.0,
            # Pass components flattened for logging
            "rew_existence": components['existence'],
            "rew_instructor": components['instructor'],
            "rew_penalty": components['penalty'],
            "rew_guidance": components['guidance'],
            "rew_combat": components['combat']
        }

        return self._get_obs(agent_id), reward, term, trunc, info

    def _calculate_reward(self, agent_id, actions):
        """
        Calculates rewards broken down by component for Tensorboard logging.
        """
        # Initialize components
        r_exist = 0.0
        r_inst = 0.0
        r_pen = 0.0
        r_guide = 0.0
        r_combat = 0.0

        term = False
        trunc = False
        reason = "none"

        # === DEATH CHECK ===
        if agent_id not in self.core.entities:
            ev = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            if ev and ev['type'] == 'kill':
                return -50.0, True, False, "shot", {'existence': 0, 'instructor': 0, 'penalty': 0, 'guidance': 0,
                                                    'combat': -50}
            else:
                return -50.0, True, False, "crash", {'existence': 0, 'instructor': 0, 'penalty': 0, 'guidance': 0,
                                                     'combat': -50}

        agent = self.core.entities[agent_id]

        # 1. EXISTENCE / TIME
        if self.phase in [1, 2]:
            r_exist += 0.02
            # Instructor (Alt/Speed)
            alt_score = math.exp(-((agent.alt - 6000.0) ** 2) / (2 * 1000 ** 2))
            spd_score = math.exp(-((agent.speed - 600.0) ** 2) / (2 * 100 ** 2))
            r_inst += (alt_score + spd_score) * 0.05
        else:
            r_exist -= 0.005  # Time pressure in combat

        # 2. PENALTIES
        if agent.speed < 250.0:
            r_pen -= (250.0 - agent.speed) * 0.001

        penalty_factor = max(0.0, 1.0 - (self.phase - 1) * 0.5)
        if agent.g_load > 2.5:
            r_pen -= (0.005 * (agent.g_load ** 2)) * penalty_factor
        if self.phase == 1:
            r_pen -= abs(agent.roll) * 0.01

        if agent_id in actions:
            delta = np.sum(np.abs(actions[agent_id][:2] - self.last_action[:2]))
            r_pen -= min(delta * 0.005, 0.01)
            self.last_action = actions[agent_id].copy()

        # 3. GUIDANCE (Align/Lock) & ACTION (Shoot)
        nearest = None
        min_dist = float('inf')
        for e in self.core.entities.values():
            if e.team == "red":
                d = dist_2d(agent.x, agent.y, e.x, e.y)
                if d < min_dist: min_dist = d; nearest = e

        if nearest:
            bearing = bearing_deg(agent.x, agent.y, nearest.x, nearest.y)
            ata = abs((bearing - agent.heading + 180) % 360 - 180)
            dist_km = min_dist / 1000.0

            if ata < 60.0:
                r_guide += (1.0 - (ata / 60.0)) * 0.05
                if dist_km < self.cfg.MISSILE_RANGE_KM:
                    _, is_locking = self.core.get_sensor_state(agent_id, nearest.uid)
                    if is_locking:
                        r_guide += 0.1

                        # Shot Bonus
            curr_ammo = agent.ammo
            prev_ammo = self.last_ammo.get(agent_id, curr_ammo)
            if curr_ammo < prev_ammo:
                if dist_km < self.cfg.MISSILE_RANGE_KM and ata < 20.0:
                    r_combat += 2.0  # Good shot
                else:
                    r_combat -= 0.5  # Bad shot
            self.last_ammo[agent_id] = curr_ammo

        # 4. EVENTS (Kill/Win)
        for ev in self.core.events:
            if ev['type'] == 'kill':
                if ev['killer'] == agent_id:
                    r_combat += 50.0
                elif ev['victim'] == agent_id:
                    r_combat -= 50.0

        reds_alive = sum(1 for e in self.core.entities.values() if e.team == "red")
        if reds_alive == 0:
            r_combat += 50.0
            r_combat += agent.ammo * 5.0
            term = True;
            reason = "win"

        if self.core.time >= self.cfg.MAX_DURATION_SEC:
            trunc = True;
            reason = "timeout"

        total_reward = r_exist + r_inst + r_pen + r_guide + r_combat

        components = {
            "existence": r_exist,
            "instructor": r_inst,
            "penalty": r_pen,
            "guidance": r_guide,
            "combat": r_combat
        }

        return total_reward, term, trunc, reason, components

    def _get_result(self, agent_id, reward, term, reason):
        # Return empty components for fail states
        comps = {'existence': 0, 'instructor': 0, 'penalty': 0, 'guidance': 0, 'combat': 0}
        # Assign reward to penalty component for logging visibility
        comps['penalty'] = reward

        info = {
            "termination_reason": reason,
            "red_obs": np.zeros(self.cfg.OBS_DIM),
            "global_state": self._get_global_state(),
            "physics_alt": 0.0, "physics_speed": 0.0, "physics_fuel": 0.0,
            "rew_existence": comps['existence'],
            "rew_instructor": comps['instructor'],
            "rew_penalty": comps['penalty'],
            "rew_guidance": comps['guidance'],
            "rew_combat": comps['combat']
        }
        return self._get_obs(agent_id), reward, term, False, info

    # ... [Keep _get_obs, _get_global_state, _vectorize unchanged] ...
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

    def _vectorize(self, e, ego_id, is_ego):
        xn, yn = self.map_limits.relative_position(e.x, e.y)
        hr = math.radians(e.heading)

        agent_id_oh = [0.0] * self.cfg.MAX_TEAM_SIZE
        if e.team == "blue" and e.uid in self.blue_ids:
            try:
                agent_id_oh[self.blue_ids.index(e.uid)] = 1.0
            except:
                pass

        ata_norm, aa_norm, closure = 0.0, 0.0, 0.0
        if not is_ego and ego_id in self.core.entities:
            ego = self.core.entities[ego_id]
            bearing = bearing_deg(ego.x, ego.y, e.x, e.y)
            ata = abs((bearing - ego.heading + 180) % 360 - 180)
            ata_norm = ata / 180.0
            b_to_ego = (bearing + 180) % 360
            aa = abs((b_to_ego - e.heading + 180) % 360 - 180)
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