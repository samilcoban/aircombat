#!/usr/bin/env python3
"""
DEBUG ARCHITECTURE SCRIPT (FIXED)
---------------------------------
Verifies:
1. Config Consistency (Dimensions)
2. Core Spatial Cache (Vectorization)
3. 3D Physics Logic (Pitch/Heading projection)
4. Observation Construction (Relative features)
5. Model Forward Pass (Tensor shapes)
"""

import sys
import os
import math
import numpy as np
import torch

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from src.core_flat import AirCombatCore
from src.env_flat import AirCombatEnv
from src.model import HybridActorCritic


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"TEST: {title}")
    print(f"{'=' * 60}")


def test_config():
    print_section("Configuration & Dimensions")
    print(f"FEAT_DIM:       {Config.FEAT_DIM} (Should be 29)")
    print(f"OBS_DIM:        {Config.OBS_DIM}")
    print(f"MAX_ENTITIES:   {Config.MAX_ENTITIES}")

    expected_feat = 24 + Config.MAX_TEAM_SIZE
    if Config.FEAT_DIM != expected_feat:
        print(f"❌ WARNING: FEAT_DIM {Config.FEAT_DIM} does not match expected {expected_feat}")
    else:
        print("✅ Config Dimensions look consistent.")


def test_spatial_cache():
    print_section("Core Spatial Cache & Vectorization")
    core = AirCombatCore()

    # Spawn Setup: 3-4-5 Triangle
    ego_id = core.spawn(x=0, y=0, alt=5000, heading=0, speed=100, team="blue", etype="plane")
    tgt_id = core.spawn(x=3000, y=4000, alt=5000, heading=180, speed=100, team="red", etype="plane")

    core.update_spatial_cache()

    data = core.get_relative_data(ego_id, tgt_id)
    if data is None:
        print("❌ get_relative_data returned None!")
        return

    dist, rel_pos, rel_vel = data

    print(f"Cache Distance: {dist:.2f} m")

    if abs(dist - 5000.0) < 1.0:
        print("✅ Distance Matrix is accurate.")
    else:
        print("❌ Distance Matrix calculation failed!")


def test_physics_hyper_speed():
    print_section("Fix 1.3: 3D Movement (Hyper-Speed Check)")
    core = AirCombatCore()

    # Spawn plane pointing straight UP
    # NOTE: Core limits pitch to +/- 1.4 rad (~80 deg)
    uid = core.spawn(x=0, y=0, alt=5000, heading=0, speed=600, team="blue", etype="plane")
    ent = core.entities[uid]

    # Set to max allowed pitch
    ent.pitch = 1.4

    # Step Physics
    dummy_action = np.array([0, 0, 1.0, 0, 0])
    core._update_plane_physics(ent, dummy_action)

    horizontal_move = math.sqrt(ent.x ** 2 + ent.y ** 2)
    vertical_move = ent.alt - 5000.0

    print(f"Horizontal Move: {horizontal_move:.2f} m")
    print(f"Vertical Move:   {vertical_move:.2f} m")

    # At 80 deg pitch (1.4 rad), cos(1.4) = 0.17
    # Speed 308 m/s * 0.04s = 12.3m total dist
    # Horiz = 12.3 * 0.17 = 2.1m. Vert = 12.3 * 0.98 = 12.0m

    if horizontal_move < 5.0 and vertical_move > 10.0:
        print("✅ Physics Vector Projection is correct (Matches Pitch Limit).")
    else:
        print("❌ Physics logic flaw detected!")


def test_observation_construction():
    print_section("Relative Observation Encoding")
    env = AirCombatEnv()
    obs, info = env.reset()

    ego_obs = obs[0]

    # Find the Enemy in the observation list
    found_enemy = False

    print("Scanning observation slots for RED agent...")
    for i in range(1, Config.MAX_ENTITIES):
        start = i * Config.FEAT_DIM
        vec = ego_obs[start: start + Config.FEAT_DIM]

        # Check Team ID (Idx 17). Blue=1, Red=-1
        team_flag = vec[17]
        range_val = vec[0]

        if team_flag == -1.0 and range_val > 0.0:
            print(f"-> Found ENEMY at slot {i}")
            print(f"   Team: {team_flag}")
            print(f"   Range: {range_val:.4f}")
            found_enemy = True
            break

    if found_enemy:
        print("✅ Enemy correctly encoded in observation.")
    else:
        print("❌ Could not find enemy (-1.0 team) in observation!")

    # MAWS Test
    print("\n[Testing MAWS Logic]")
    if env.blue_ids:
        bid = env.blue_ids[0]
        blue_ent = env.core.entities[bid]

        # Spawn missile close by
        mid = env.core.spawn(blue_ent.x + 1000, blue_ent.y, blue_ent.alt, 0, 2000, "red", "missile")
        env.core.entities[mid].target_id = bid

        # KEY FIX: Force cache invalidation because time didn't advance
        env.core.cached_step = -1
        env.core.update_spatial_cache()

        # Get obs
        new_obs = env._get_obs(bid)

        found_missile = False
        for i in range(1, Config.MAX_ENTITIES):
            start = i * Config.FEAT_DIM
            vec = new_obs[start: start + Config.FEAT_DIM]

            # Check type (Idx 18 is Missile, 1.0)
            if vec[18] > 0.5:
                print(f"-> Found Missile at slot {i}")
                print(f"   MAWS (Idx 21): {vec[21]}")
                if vec[21] == 1.0:
                    found_missile = True

        if found_missile:
            print("✅ MAWS correctly triggered.")
        else:
            print("❌ MAWS failed to trigger.")


def test_model_forward():
    print_section("Model Forward Pass")
    try:
        model = HybridActorCritic().to(Config.DEVICE)
        dummy_obs = torch.randn(2, Config.OBS_DIM).to(Config.DEVICE)
        action, _, _, _, _ = model.get_action_and_value(dummy_obs)

        if action.shape == (2, 5):
            print("✅ Model Forward Pass Successful.")
        else:
            print("❌ Model output shape mismatch.")

    except Exception as e:
        print(f"❌ CRASH: {e}")


if __name__ == "__main__":
    print("🐞 STARTING DEBUG SUITE (V2) 🐞")
    test_config()
    test_spatial_cache()
    test_physics_hyper_speed()
    test_observation_construction()
    test_model_forward()
    print("\nDebug complete.")