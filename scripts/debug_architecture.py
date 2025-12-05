#!/usr/bin/env python3
"""
DEBUG ARCHITECTURE SCRIPT (PHASE 5 VERIFICATION)
------------------------------------------------
Verifies:
1. Config Consistency (Dimensions)
2. Core Spatial Cache (Vectorization, Radians, Body Frame)
3. 3D Physics Logic (Gravity, Infinite Glide fix)
4. Observation Construction (Cos/Sin features)
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
    print(f"FEAT_DIM_EGO:   {Config.FEAT_DIM_EGO} (Should be 18)")
    print(f"FEAT_DIM_EDGE:  {Config.FEAT_DIM_EDGE} (Should be 14)")
    print(f"GNN_EDGE_DIM:   {Config.GNN_EDGE_DIM} (Should be 8)")
    print(f"OBS_DIM:        {Config.OBS_DIM}")
    print(f"MAX_ENTITIES:   {Config.MAX_ENTITIES}")

    # Calculation Check
    expected_obs = Config.FEAT_DIM_EGO + ((Config.MAX_ENTITIES - 1) * Config.FEAT_DIM_EDGE)
    if Config.OBS_DIM != expected_obs:
        print(f"❌ WARNING: OBS_DIM {Config.OBS_DIM} does not match expected {expected_obs}")
    else:
        print("✅ Config Dimensions look consistent.")


def test_spatial_cache():
    print_section("Core Spatial Cache & Vectorization")
    core = AirCombatCore()

    # Spawn Setup:
    # Ego: At (0,0,5000), Heading North (0 rad), Level
    # Tgt: At (100, 100, 5000), Heading West (PI/2 rad), Level

    # NOTE: Inputs to spawn are now RADIANS per Phase 1 refactor
    ego_id = core.spawn(x=0, y=0, alt=5000, heading=0.0, speed=100, team="blue", etype="plane")
    tgt_id = core.spawn(x=100, y=100, alt=5000, heading=math.pi / 2, speed=100, team="red", etype="plane")

    core.update_spatial_cache()

    # Get Data: Ego -> Tgt
    # (Dist, RelPos, RelVel, ATA_Cos, AA_Cos, LocalPos)
    data = core.get_relative_data(ego_id, tgt_id)
    if data is None:
        print("❌ get_relative_data returned None!")
        return

    dist, rel_pos, rel_vel, ata_cos, aa_cos, local_pos = data

    print(f"Cache Distance: {dist:.2f} m")
    print(f"Local Pos (Body Frame): {local_pos}")
    print(f"ATA Cos: {ata_cos:.4f}")

    # Validation
    # Distance should be sqrt(100^2 + 100^2) = 141.42
    if abs(dist - 141.42) < 1.0:
        print("✅ Distance Matrix is accurate.")
    else:
        print(f"❌ Distance Error! Expected 141.42, got {dist}")

    # Local Pos Check (Ego facing North/X)
    # Tgt is at (100, 100). So Local X (Fwd) = 100, Local Y (Right) = -100 or +100?
    # Wait, Y is East. 0 rad is North (+X).
    # If Heading=0 (North), then X=North, Y=East.
    # Target at (100, 100) relative (North 100, East 100).
    # Local Pos should be [100, 100, 0].
    if abs(local_pos[0] - 100) < 1.0 and abs(local_pos[1] - 100) < 1.0:
        print("✅ Body Frame Transformation (Local Pos) is correct.")
    else:
        print(f"❌ Body Frame Error! Expected [100, 100, 0], got {local_pos}")


def test_physics_logic():
    print_section("Physics Logic (Radians & Gravity)")
    core = AirCombatCore()

    # Spawn plane pointing East (+Y)
    # Heading = PI/2
    uid = core.spawn(x=0, y=0, alt=5000, heading=math.pi / 2, speed=600, team="blue", etype="plane")
    ent = core.entities[uid]

    # Step Physics
    # Action: [Roll=0, G=0 (1.0 net), Throttle=1.0, Fire=0, CM=0]
    dummy_action = np.array([0.0, 0.0, 1.0, 0, 0])

    # Run 1 second of physics (5 steps of 0.2s, 25 sub-steps total)
    # Actually core.step takes main step. So call it 5 times.
    for _ in range(5):
        core.step({uid: dummy_action})

    # Analysis
    # Speed 600 knots ~= 308 m/s
    # Time 1.0s
    # Distance ~= 308m
    # Direction East (+Y) -> X should be ~0, Y should be ~308

    print(f"Final Pos: X={ent.x:.1f}, Y={ent.y:.1f}, Z={ent.alt:.1f}")

    if ent.y > 200.0 and abs(ent.x) < 50.0:
        print("✅ Physics Movement Direction (Radians) is correct (Moved East).")
    else:
        print("❌ Physics Direction Error! Plane did not move East.")

    # Gravity Check (Infinite Glide Fix)
    # Pitch is 0. Lift is 1G (Commanded).
    # If Fly-By-Wire logic works, Alt should be roughly constant.
    # If "Infinite Glide" bug exists (and no FBW), it might drift.
    # In our refactor, we implemented FBW-style "Command G".
    # 1.0 G commanded - Gravity = 0 Vertical Accel.
    # So altitude should be stable.
    if abs(ent.alt - 5000.0) < 50.0:
        print("✅ Altitude Hold logic (Fly-By-Wire) is working.")
    else:
        print(f"⚠️ Altitude drifted significantly: {ent.alt}")


def test_observation_construction():
    print_section("Observation Encoding")
    env = AirCombatEnv()
    obs, info = env.reset()

    # Ego Obs (Blue 0)
    ego_obs = obs[0]

    # 1. Check Dimensions
    if len(ego_obs) != Config.OBS_DIM:
        print(f"❌ Obs Dim mismatch! Got {len(ego_obs)}, Expected {Config.OBS_DIM}")
        return

    # 2. Check Ego Features (0-17)
    # Index 0 is 'Exists' (Should be 1.0)
    if ego_obs[0] != 1.0:
        print("❌ Ego Existence Flag missing!")
    else:
        print("✅ Ego Feature Block seems valid.")

    # 3. Check Track Features (18+)
    # We should have at least one track (Red agent)
    # Find the non-zero track
    found_enemy = False

    # Iterate tracks
    # Skip Ego (18)
    tracks_flat = ego_obs[Config.FEAT_DIM_EGO:]
    num_tracks = len(tracks_flat) // Config.FEAT_DIM_EDGE

    print(f"Scanning {num_tracks} potential tracks...")

    for i in range(num_tracks):
        start = i * Config.FEAT_DIM_EDGE
        vec = tracks_flat[start: start + Config.FEAT_DIM_EDGE]

        # Check Range (Index 0). If > 0, it's a valid track.
        if vec[0] > 0.0:
            print(f"-> Found Track {i}: Range Norm {vec[0]:.4f}")
            # Check Team (Index 7). 1.0=Friend, -1.0=Enemy
            if vec[7] < -0.5:
                print("   Type: Enemy")
                found_enemy = True
            elif vec[7] > 0.5:
                print("   Type: Friend")
            break  # Just check first valid one

    if found_enemy:
        print("✅ Enemy correctly encoded in observation.")
    else:
        print("⚠️ No Enemy found in obs (Might be out of range or dead).")


def test_model_forward():
    print_section("Model Forward Pass (Hybrid)")
    try:
        model = HybridActorCritic().to(Config.DEVICE)

        # Dummy Obs: (2 Agents, OBS_DIM)
        dummy_obs = torch.randn(2, Config.OBS_DIM).to(Config.DEVICE)

        # Dummy Graph: List of 2 Data objects
        from torch_geometric.data import Data, Batch
        # New Edge Dim is 8
        g1 = Data(x=torch.randn(3, 12), edge_index=torch.zeros(2, 6, dtype=torch.long), edge_attr=torch.randn(6, 8))
        g2 = Data(x=torch.randn(2, 12), edge_index=torch.zeros(2, 2, dtype=torch.long), edge_attr=torch.randn(2, 8))
        batch = Batch.from_data_list([g1, g2]).to(Config.DEVICE)

        # Forward
        action, _, _, val, _ = model.get_action_and_value(dummy_obs, graph_data=batch)

        if action.shape == (2, 5):
            print("✅ Actor Output Shape Correct (2, 5).")
        else:
            print(f"❌ Actor Output Mismatch: {action.shape}")

        if val.shape == (2, 1):
            print("✅ Critic Output Shape Correct (2, 1).")
        else:
            print(f"❌ Critic Output Mismatch: {val.shape}")

    except Exception as e:
        print(f"❌ CRASH in Model: {e}")


if __name__ == "__main__":
    print("🐞 STARTING ARCHITECTURE VERIFICATION (PHASE 5) 🐞")
    test_config()
    test_spatial_cache()
    test_physics_logic()
    test_observation_construction()
    test_model_forward()
    print("\nDebug complete.")