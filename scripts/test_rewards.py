#!/usr/bin/env python3
import sys
import os
import numpy as np
import math
import torch

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from src.env import AirCombatEnv
from src.bot import HardcodedAce


def set_scenario(env, scenario_type="tail_chase", dist_m=5000):
    """
    God Mode: Manually positions entities to create specific testing scenarios.
    """
    # 1. Clear existing entities to be safe, keeping IDs
    if not env.blue_ids or not env.red_ids:
        env.reset()

    bid = env.blue_ids[0]
    rid = env.red_ids[0]

    blue = env.core.entities[bid]
    red = env.core.entities[rid]

    # Reset stats
    blue.speed = 600.0  # Knots
    red.speed = 600.0
    blue.alt = 5000.0
    red.alt = 5000.0
    blue.pitch = 0.0
    red.pitch = 0.0
    blue.roll = 0.0
    red.roll = 0.0
    blue.ammo = 4  # Ensure ammo

    # Center of map
    cx, cy = 0.0, 0.0

    if scenario_type == "tail_chase":
        # Blue behind Red, both heading North (0 rad)
        # Red at (0, dist/2), Blue at (0, -dist/2)
        red.x = cx + dist_m / 2.0
        red.y = cy
        red.heading = 0.0  # North (+X)

        blue.x = cx - dist_m / 2.0
        blue.y = cy
        blue.heading = 0.0  # North (+X)

    elif scenario_type == "head_on":
        # Blue heading North, Red heading South
        red.x = cx + dist_m / 2.0
        red.y = cy
        red.heading = math.pi  # South (-X)

        blue.x = cx - dist_m / 2.0
        blue.y = cy
        blue.heading = 0.0  # North (+X)

    elif scenario_type == "beam":
        # Blue heading North, Red heading East (Notching)
        # Red is in front of Blue
        red.x = cx + dist_m
        red.y = cy
        red.heading = math.pi / 2.0  # East (+Y)

        blue.x = cx
        blue.y = cy
        blue.heading = 0.0  # North

    # CRITICAL: Update physics cache immediately so observations are correct
    env.core.update_spatial_cache()

    # Re-init potentials for PBRS so we don't get a massive jump from the teleport
    env.prev_potentials[bid] = env._get_current_potential(bid)

    print(f"--- SCENARIO SET: {scenario_type.upper()} | Dist: {dist_m}m ---")


def run_test(env, bot, steps=50, auto_fire=True):
    print(
        f"{'Step':<5} | {'Action (Roll/G/Fire)':<20} | {'Dist(km)':<8} | {'Rew_Tot':<8} | {'Rew_Pos':<8} | {'Rew_Kill':<8} | {'Event'}")
    print("-" * 100)

    total_shaping = 0.0

    for i in range(steps):
        # 1. Get Obs
        obs = env._get_all_blue_obs()  # (N, Dim)

        # 2. Get Bot Action
        # Flatten obs for bot
        act = bot.get_action(obs[0])

        # Override Fire if needed
        if auto_fire and i > 5:
            # Force fire if locked to test kill reward
            # Check lock internally via env core to be sure
            dist_val = env.core.get_relative_data(env.blue_ids[0], env.red_ids[0])[0]
            if dist_val < 20000:
                act[3] = 1.0  # Pull trigger

        # 3. Step
        # Add batch dim for step
        act_batch = np.expand_dims(act, 0)

        # Dummy Red Action (Fly straight)
        red_act = np.zeros((1, 5), dtype=np.float32)
        red_act[0, 2] = 0.8  # Throttle

        next_obs, rewards, term, trunc, info = env.step(act_batch, red_actions=red_act)

        # 4. Log
        bd = info['reward_breakdown']

        # Get actual distance for logging
        dist_str = "DEAD"
        if env.blue_ids[0] in env.core.entities and env.red_ids[0] in env.core.entities:
            d = env.core.get_relative_data(env.blue_ids[0], env.red_ids[0])[0]
            dist_str = f"{d / 1000.0:.1f}"

        event_str = ""
        if info['stat_kills'] > 0: event_str = "KILL!"
        if term: event_str += " TERM"

        # Format Action string
        act_str = f"{act[0]:.1f} / {act[1]:.1f} / {act[3]:.0f}"

        print(
            f"{i:<5} | {act_str:<20} | {dist_str:<8} | {rewards[0]:.4f}   | {bd['rew_pos']:.4f}   | {bd['rew_kill']:.4f}   | {event_str}")

        total_shaping += bd['rew_pos']

        if term:
            break

    print("-" * 100)
    print(f"Total Shaping Reward: {total_shaping:.4f}")
    if total_shaping < 0:
        print("⚠️ WARNING: Net Shaping is NEGATIVE. Check if PBRS is punishing approach.")
    else:
        print("✅ Shaping is Positive.")
    print("\n")


def main():
    # Setup
    env = AirCombatEnv()
    bot = HardcodedAce()

    # 1. Test Tail Chase (Should be easy positive shaping)
    env.reset()
    set_scenario(env, "tail_chase", dist_m=10000)
    run_test(env, bot, steps=20, auto_fire=False)

    # 2. Test Head On (High closure, dangerous)
    env.reset()
    set_scenario(env, "head_on", dist_m=20000)
    run_test(env, bot, steps=20, auto_fire=False)

    # 3. Test THE KILL BUG (Close range, force kill)
    # We want to see if Rew_Pos drops negatively when enemy dies
    env.reset()
    set_scenario(env, "tail_chase", dist_m=2000)  # Very close
    print("TESTING KILL STABILITY...")
    run_test(env, bot, steps=50, auto_fire=True)


if __name__ == "__main__":
    main()