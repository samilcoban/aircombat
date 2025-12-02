# ================================================
# FILE: src/env_flat.py
# ================================================
"""
REINFORCEMENT LEARNING ENVIRONMENT FOR AIR COMBAT

This module wraps the AirCombatCore physics engine in a Gymnasium-compatible
RL environment. It handles observation encoding, reward shaping, curriculum
learning, and multi-agent coordination.

OBSERVATION SPACE:
- Ego-centric view: Each agent sees the world from its own perspective
- Entity features: Position, velocity, attitude, team, type, tactical geometry
- Sensor simulation: Radar visibility affects what enemies are observable
- Normalization: All values scaled to roughly [-1, 1] for neural network training

REWARD SHAPING PHILOSOPHY:
1. **Sparse Terminal Rewards**: +50 for kills, -200 for death
2. **Dense Guidance Rewards**: Small bonuses for pointing at enemy, achieving lock
3. **Instructor Rewards**: Teach specific behaviors (trigger discipline, altitude management)
4. **Curriculum Scaling**: Reduce shaping rewards in later phases to avoid exploitation

CURRICULUM LEARNING (5 Phases):
- Phase 1: Survival training against stationary dummy (learn to fly)
- Phase 2: Intercept training against slow-moving drone (learn to approach and shoot)
- Phase 3+: Combat training against increasingly skilled opponents (learn to fight)

KEY DESIGN DECISIONS:
- **Boresight Bonus**: Massive reward for keeping enemy in crosshairs (ATA < 10°)
- **Trigger Training**: Reward pressing fire button when locked (teaches weapon employment)
- **Trigger Discipline**: Punish random firing without lock (prevents spam)
- **Strict Win Condition**: Only count as win if agent actively scored kills (no passive wins)
- **Altitude Floor**: Hard deck at 2000m to prevent ground crashes

TACTICAL GEOMETRY:
- ATA (Angle-To-Attack): How far off-nose the target is from my heading
- AA (Aspect Angle): Target's angle relative to me (head-on vs tail-on)
- Closure Rate: Combined velocity toward each other
"""
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
        self.dead_agent_ids = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.core = AirCombatCore()
        self.blue_ids = []
        self.red_ids = []
        self.last_actions = {}
        self.last_ammo = {}
        self.dead_agent_ids = set()

        cx_rel, cy_rel = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        cx, cy = self.map_limits.absolute_position(cx_rel, cy_rel)
        axis = rng.uniform(0.0, 360.0)

        if self.phase <= 2:
            sep = rng.uniform(5000, 8000)
        else:
            sep = rng.uniform(40000.0, 60000.0)

        # Spawn Blue Agents
        for i in range(self.n_agents):
            offset = (i - (self.n_agents - 1) / 2) * 500.0
            bx = cx + (sep / 2) * math.cos(math.radians(axis + 180)) + offset * math.sin(math.radians(axis))
            by = cy + (sep / 2) * math.sin(math.radians(axis + 180)) - offset * math.cos(math.radians(axis))
            spd = 600.0 if self.phase <= 2 else 900.0
            bid = self.core.spawn(bx, by, axis, spd, "blue", "plane")

            if self.phase <= 2:
                self.core.entities[bid].alt = 6000.0
            else:
                self.core.entities[bid].alt = 10000.0

            self.blue_ids.append(bid)
            self.last_actions[bid] = np.zeros(self.cfg.ACTION_DIM)
            self.last_ammo[bid] = 4

        # Spawn Red Agents
        n_red = 1
        if self.phase > 2:
            n_red = rng.integers(1, self.cfg.N_ENEMIES_MAX + 1)

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
            # Pass action for Trigger Discipline check
            act = actions_dict.get(agent_id, np.zeros(5))
            rew, term, reason, comps, stats = self._calculate_reward(agent_id, win, timeout, act)
            rewards.append(rew)
            dones.append(term or global_term or global_trunc)
            total_fired += stats['fired']
            total_kills += stats['kills']

            if agent_id in self.core.entities:
                agent = self.core.entities[agent_id]
                stall_ratio = np.clip((150.0 - agent.speed) / 50.0, 0.0, 1.0) if agent.speed < 150.0 else 0.0
                g_load = agent.g_load

        # FIX: Strict Win Condition (Thesis 4)
        term_reason = "none"
        if win:
            if total_kills > 0:
                term_reason = "win"  # Active Win
            else:
                term_reason = "win_passive"  # Passive (Draw)
        elif defeat:
            term_reason = "crash"
        elif timeout:
            term_reason = "timeout"

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

        obs_list = []
        for uid in self.red_ids:
            if uid in self.core.entities:
                obs_list.append(self._get_obs(uid))
            else:
                obs_list.append(np.zeros(self.cfg.OBS_DIM, dtype=np.float32))
        return np.stack(obs_list).astype(np.float32)

    def _calculate_reward(self, agent_id, win_condition, timeout_condition, action):
        """
        Calculate reward for a single agent based on its actions and outcomes.
        
        REWARD DESIGN PHILOSOPHY:
        The reward function is the most critical part of RL training. It must:
        1. Encourage desired behaviors (pointing at enemy, firing when locked)
        2. Discourage dangerous behaviors (stalling, flying too low, random firing)
        3. Provide dense feedback for learning (not just sparse terminal rewards)
        4. Scale appropriately across curriculum phases
        
        REWARD COMPONENTS:
        
        1. TERMINAL REWARDS (Sparse, Large):
           - Death: -200 (crash or shot down)
           - Kill: +50 (destroyed an enemy)
           - Win: +50 (all enemies destroyed AND agent scored kills)
        
        2. EXISTENCE REWARDS (Time Pressure):
           - Phase 1: +0.01/step (encourage survival)
           - Phase 2: 0.0 (neutral - focus on mission)
           - Phase 3+: -0.005/step (pressure to end fight quickly)
        
        3. FLIGHT SAFETY PENALTIES:
           - Low speed: -0.001 × (250 - speed) if speed < 250 kts
           - High G: -0.005 × G² if G > 6.0
           - Low altitude: Soft penalty below 4000m, death below 2000m
        
        4. GUIDANCE REWARDS (Dense Shaping):
           - Boresight bonus: +0.1 if ATA < 10° (enemy in crosshairs)
           - General guidance: +0.05 × (1 - ATA/60°) if ATA < 60°
           - Lock reward: +0.1 if locked and in missile range
        
        5. INSTRUCTOR REWARDS (Behavior Shaping):
           - Trigger training: +0.5 for pressing fire when locked
           - Altitude recovery: +0.05 for pitching up when low
        
        6. TRIGGER DISCIPLINE:
           - Penalty: -0.05 for firing without lock (ATA > 60°)
           - Prevents agents from spamming fire button randomly
        
        7. COMBAT REWARDS:
           - Valid launch: +2.0 for firing missile when locked and in range
           - Kill: +50.0 for destroying an enemy
        
        CURRICULUM SCALING:
        - Phases 1-2: Full shaping rewards (shaping_scale = 1.0)
        - Phase 3+: Reduced shaping (shaping_scale = 0.1)
        - This prevents exploitation of dense rewards in combat scenarios
        
        DESIGN RATIONALE:
        - Boresight bonus is intentionally large to teach "point at enemy" behavior
        - Trigger training explicitly rewards button press to overcome exploration barrier
        - Strict win condition (must score kills) prevents passive strategies
        - Altitude floor prevents agents from learning to dive into ground
        """
        comps = {'existence': 0, 'instructor': 0, 'penalty': 0, 'guidance': 0, 'combat': 0}
        stats = {'fired': 0, 'kills': 0}

        # --- DEATH ---
        if agent_id not in self.core.entities:
            if agent_id in self.dead_agent_ids:
                return 0.0, True, "dead", comps, stats
            self.dead_agent_ids.add(agent_id)
            ev = next((e for e in self.core.events if e.get('victim') == agent_id), None)
            reason = "shot" if ev and ev['type'] == 'kill' else "crash"
            return -200.0, True, reason, comps, stats

        agent = self.core.entities[agent_id]
        rew = 0.0

        # --- REWARD SCALING ---
        shaping_scale = 1.0
        if self.phase >= 3:
            shaping_scale = 0.1  # Starve the drone in combat

        # --- EXISTENCE / TIME PRESSURE ---
        if self.phase == 1:
            # Phase 1: Just survive
            rew += 0.01
            comps['existence'] += 0.01
        elif self.phase == 2:
            # Phase 2: Intercept Training. NO free existence reward.
            # Only reward doing the job.
            pass
        else:
            # Phase 3+: Combat Pressure
            rew -= 0.005
            comps['existence'] -= 0.005

        # --- FLIGHT SAFETY ---
        r_pen = 0
        if agent.speed < 250.0: r_pen -= (250.0 - agent.speed) * 0.001
        if agent.g_load > 6.0: r_pen -= (0.005 * (agent.g_load ** 2))

        if agent.alt < 2000:
            self.dead_agent_ids.add(agent_id)
            return -200.0, True, "floor", comps, stats

        if agent.alt < 4000:
            r_soft = -1.0 * (4000.0 - agent.alt) / 2000.0
            rew += r_soft
            comps['penalty'] += r_soft
            if agent.pitch > 0:
                rew += 0.05
                comps['instructor'] += 0.05

        rew += r_pen
        comps['penalty'] += r_pen

        # --- COMBAT / GUIDANCE ---
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

            # Boresight Bonus (Thesis 1)
            # Massive reward for keeping enemy in crosshairs (ATA < 10)
            if ata < 10.0:
                r_bore = 0.1 * shaping_scale
                rew += r_bore
                comps['guidance'] += r_bore

            # General Guidance
            if ata < 60.0:
                r_guide = (1.0 - (ata / 60.0)) * 0.05 * shaping_scale
                rew += r_guide
                comps['guidance'] += r_guide

                # Lock Reward
                if dist_km < self.cfg.MISSILE_RANGE_KM:
                    _, is_locking = self.core.get_sensor_state(agent_id, nearest.uid)
                    if is_locking:
                        rew += 0.1 * shaping_scale
                        comps['guidance'] += 0.1 * shaping_scale

                        # TRIGGER TRAINING (Thesis 2)
                        # If Locked AND Pressing Fire -> Reward the ATTEMPT
                        # action[3] is Fire. > 0 means pressed.
                        if action[3] > 0.0:
                            rew += 0.5  # "Click" Reward
                            comps['instructor'] += 0.5

            # TRIGGER DISCIPLINE (Thesis 3)
            # If pressing fire randomly (no lock, or wide angle), punish spam
            if action[3] > 0.0 and ata > 60.0:
                rew -= 0.05
                comps['penalty'] -= 0.05

            # Distance Penalty (Close the gap!)
            if dist_km > 10.0 and self.phase == 2:
                rew -= 0.001

            # Actual Firing Event
            curr_ammo = agent.ammo
            prev_ammo = self.last_ammo.get(agent_id, curr_ammo)
            if curr_ammo < prev_ammo:
                stats['fired'] = 1
                if dist_km < self.cfg.MISSILE_RANGE_KM and ata < 20.0:
                    rew += 2.0  # Big reward for valid launch
                    comps['combat'] += 2.0
                else:
                    # Remove penalty for missing (Thesis 4) - Let them learn to shoot first
                    pass
            self.last_ammo[agent_id] = curr_ammo

        # Kills
        for ev in self.core.events:
            if ev['type'] == 'kill' and ev['killer'] == agent_id:
                rew += 50.0
                comps['combat'] += 50.0
                stats['kills'] = 1

        # --- WIN CONDITION ---
        if win_condition and stats['kills'] > 0:
            rew += 50.0
            comps['combat'] += 50.0
            return rew, False, "win", comps, stats
        elif win_condition:
            # Passive Win -> No Reward, No Win Status
            return rew, False, "win_passive", comps, stats

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

        is_locked_by_me = 0.0
        if not is_ego and ego_id in self.core.entities:
            _, is_locking = self.core.get_sensor_state(ego_id, e.uid)
            if is_locking: is_locked_by_me = 1.0

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
            is_locked_by_me,
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