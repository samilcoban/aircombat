# ================================================
# FILE: scripts/debug_architecture.py
# ================================================
# !/usr/bin/env python3
"""
DEBUG ARCHITECTURE SCRIPT (PHASE 5 VERIFICATION)
------------------------------------------------
Verifies:
1. Config Consistency (Unified Dimensions)
2. Core Spatial Cache
3. 3D Physics Logic
4. Observation Construction (Unified Node/Edge)
5. Model Forward Pass
"""

import sys
import os
import math
import numpy as np
import torch

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
    print(f"NODE_DIM:       {Config.NODE_DIM} (Should be 16)")
    print(f"EDGE_DIM:       {Config.EDGE_DIM} (Should be 12)")
    print(f"OBS_DIM:        {Config.OBS_DIM}")
    print(f"MAX_ENTITIES:   {Config.MAX_ENTITIES}")

    # Calculation Check
    expected_obs = Config.NODE_DIM + ((Config.MAX_ENTITIES - 1) * Config.EDGE_DIM)
    if Config.OBS_DIM != expected_obs:
        print(f"❌ WARNING: OBS_DIM {Config.OBS_DIM} does not match expected {expected_obs}")
    else:
        print("✅ Config Dimensions look consistent.")


def test_spatial_cache():
    print_section("Core Spatial Cache & Vectorization")
    core = AirCombatCore()
    ego_id = core.spawn(x=0, y=0, alt=5000, heading=0.0, speed=100, team="blue", etype="plane")
    tgt_id = core.spawn(x=100, y=100, alt=5000, heading=math.pi / 2, speed=100, team="red", etype="plane")

    core.update_spatial_cache()
    data = core.get_relative_data(ego_id, tgt_id)

    if data is None:
        print("❌ get_relative_data returned None!")
        return

    dist, rel_pos, rel_vel, ata_cos, aa_cos, local_pos = data
    print(f"Cache Distance: {dist:.2f} m")

    if abs(dist - 141.42) < 1.0:
        print("✅ Distance Matrix is accurate.")
    else:
        print(f"❌ Distance Error! Expected 141.42, got {dist}")


def test_observation_construction():
    print_section("Observation Encoding (Unified Node/Edge)")
    env = AirCombatEnv()
    obs, info = env.reset()

    # Ego Obs (Blue 0)
    ego_obs = obs[0]

    # 1. Check Dimensions
    if len(ego_obs) != Config.OBS_DIM:
        print(f"❌ Obs Dim mismatch! Got {len(ego_obs)}, Expected {Config.OBS_DIM}")
        return

    # 2. Check Ego Features (Unified Node)
    # Index 0 is 'Exists' (1.0)
    if ego_obs[0] != 1.0:
        print("❌ Ego Existence Flag missing!")
    else:
        print("✅ Ego Node Block seems valid.")

    # 3. Check Track Features (Unified Edge)
    tracks_flat = ego_obs[Config.NODE_DIM:]
    num_tracks = len(tracks_flat) // Config.EDGE_DIM

    print(f"Scanning {num_tracks} potential tracks...")
    found_enemy = False

    for i in range(num_tracks):
        start = i * Config.EDGE_DIM
        vec = tracks_flat[start: start + Config.EDGE_DIM]

        # Check Range (Index 0). If > 0, it's a valid track.
        if vec[0] > 0.0:
            print(f"-> Found Track {i}: Range Norm {vec[0]:.4f}")
            # Check Team Relation (Index 10). 1.0=Friend, -1.0=Enemy
            if vec[10] < -0.5:
                print("   Type: Enemy")
                found_enemy = True
            elif vec[10] > 0.5:
                print("   Type: Friend")
            break

    if found_enemy:
        print("✅ Enemy correctly encoded in unified edge observation.")
    else:
        print("⚠️ No Enemy found in obs (Might be out of range or dead).")


def test_model_forward():
    print_section("Model Forward Pass (Hybrid)")
    try:
        model = HybridActorCritic().to(Config.DEVICE)
        dummy_obs = torch.randn(2, Config.OBS_DIM).to(Config.DEVICE)

        from torch_geometric.data import Data, Batch
        # Use new dimensions
        g1 = Data(x=torch.randn(3, Config.NODE_DIM), edge_index=torch.zeros(2, 6, dtype=torch.long),
                  edge_attr=torch.randn(6, Config.EDGE_DIM))
        g2 = Data(x=torch.randn(2, Config.NODE_DIM), edge_index=torch.zeros(2, 2, dtype=torch.long),
                  edge_attr=torch.randn(2, Config.EDGE_DIM))
        batch = Batch.from_data_list([g1, g2]).to(Config.DEVICE)

        action, _, _, val, _ = model.get_action_and_value(dummy_obs, graph_data=batch)

        if action.shape == (2, 5) and val.shape == (2, 1):
            print("✅ Forward Pass Successful.")
        else:
            print(f"❌ Output Shape Mismatch: Act={action.shape}, Val={val.shape}")

    except Exception as e:
        print(f"❌ CRASH in Model: {e}")


if __name__ == "__main__":
    test_config()
    test_spatial_cache()
    test_observation_construction()
    test_model_forward()