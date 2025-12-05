#!/usr/bin/env python3
"""
DEBUG GNN SCRIPT (PHASE 5 VERIFICATION)
---------------------------------------
Verifies:
1. Environment Graph Generation (_get_graph_state)
2. Graph Feature Dimensions (Node=12, Edge=8)
3. Model Critic Forward Pass (GNN Layers)
4. Batching Logic (PyG Batching)
"""

import sys
import os
import torch
import numpy as np
from torch_geometric.data import Data, Batch

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from src.env_flat import AirCombatEnv
from src.model import HybridActorCritic


def test_gnn_pipeline():
    print(f"Testing GNN Pipeline with GNN_EDGE_DIM={Config.GNN_EDGE_DIM}...")

    # 1. Setup Env and Model
    env = AirCombatEnv()
    obs, info = env.reset()

    model = HybridActorCritic().to(Config.DEVICE)
    model.eval()

    # 2. Inspect Graph Data from Reset
    print("\n[1] Inspecting Initial Graph State...")
    if "graph_data" not in info:
        print("❌ Graph data missing from info dict!")
        return

    gd = info["graph_data"]
    x = gd['x']
    edge_index = gd['edge_index']
    edge_attr = gd['edge_attr']

    print(f"   Node Features (x): {x.shape} (Expected N, 12)")
    print(f"   Edge Index:        {edge_index.shape} (Expected 2, E)")
    print(f"   Edge Attr:         {edge_attr.shape} (Expected E, {Config.GNN_EDGE_DIM})")

    # Validation: Check Dimensions
    if x.shape[1] != 12:
        print(f"❌ Node Feature Dim Error! Got {x.shape[1]}, Expected 12")

    if edge_attr.shape[1] != Config.GNN_EDGE_DIM:
        print(f"❌ Edge Feature Dim Error! Got {edge_attr.shape[1]}, Expected {Config.GNN_EDGE_DIM}")
        print("   Did you update config.py and src/env_flat.py?")

    # 3. Simulate a Step to check updates
    print("\n[2] Stepping Environment...")
    # Create dummy action
    action = np.zeros((env.n_agents, Config.ACTION_DIM), dtype=np.float32)
    obs, rew, term, trunc, info = env.step(action)

    gd_next = info["graph_data"]
    # Check if Pitch (Index 5) is non-zero (implies it's working)
    # Pitch feature is sin(pitch).
    # If planes are level (0.0), sin(0) is 0.
    # Let's verify it exists at least.
    print(f"   Sample Node Feat (Index 0): {gd_next['x'][0]}")

    # 4. Model Forward Pass
    print("\n[3] Testing Critic Forward Pass...")

    # Convert to Tensor (Mimic train.py)
    x_t = torch.tensor(gd_next['x'], dtype=torch.float32)
    edge_index_t = torch.tensor(gd_next['edge_index'], dtype=torch.long)
    edge_attr_t = torch.tensor(gd_next['edge_attr'], dtype=torch.float32)

    # Create PyG Data
    data = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t)

    # Create a Batch (Simulate 2 environments)
    # We duplicate the graph to make a batch of 2
    batch = Batch.from_data_list([data, data]).to(Config.DEVICE)

    print(f"   Batch Created: {batch}")
    print(f"   Batch Nodes: {batch.num_nodes}, Batch Edges: {batch.num_edges}")

    try:
        # We need a dummy observation to pass to get_value (Hybrid Architecture)
        # 2 Environments * N_Agents
        total_agents = 2 * Config.N_AGENTS
        dummy_obs = torch.randn(total_agents, Config.OBS_DIM).to(Config.DEVICE)

        # Critic Pass
        val = model.get_value(batch, dummy_obs)

        print(f"✅ Success! Critic Value Output Shape: {val.shape}")
        print(f"   Values: {val.view(-1).detach().cpu().numpy()}")

    except RuntimeError as e:
        print(f"❌ CRASH in Model Forward: {e}")
        print("   Hint: Check if src/model.py EdgeGCNConv init matches GNN_EDGE_DIM")


if __name__ == "__main__":
    test_gnn_pipeline()