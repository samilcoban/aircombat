# ================================================
# FILE: scripts/debug_gnn.py
# ================================================
#!/usr/bin/env python3
"""
DEBUG GNN SCRIPT (PHASE 5 VERIFICATION)
"""

import sys
import os
import torch
import numpy as np
from torch_geometric.data import Data, Batch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from src.env_flat import AirCombatEnv
from src.model import HybridActorCritic


def test_gnn_pipeline():
    print(f"Testing GNN Pipeline with EDGE_DIM={Config.EDGE_DIM}...")

    env = AirCombatEnv()
    obs, info = env.reset()
    model = HybridActorCritic().to(Config.DEVICE)
    model.eval()

    if "graph_data" not in info:
        print("❌ Graph data missing!")
        return

    gd = info["graph_data"]
    x = gd['x']
    edge_attr = gd['edge_attr']

    print(f"   Node Features (x): {x.shape} (Expected N, {Config.NODE_DIM})")
    print(f"   Edge Attr:         {edge_attr.shape} (Expected E, {Config.EDGE_DIM})")

    if x.shape[1] != Config.NODE_DIM:
        print(f"❌ Node Feature Dim Error! Got {x.shape[1]}, Expected {Config.NODE_DIM}")

    if edge_attr.shape[1] != Config.EDGE_DIM:
        print(f"❌ Edge Feature Dim Error! Got {edge_attr.shape[1]}, Expected {Config.EDGE_DIM}")

    # Test Forward Pass
    try:
        x_t = torch.tensor(gd['x'], dtype=torch.float32)
        edge_index_t = torch.tensor(gd['edge_index'], dtype=torch.long)
        edge_attr_t = torch.tensor(gd['edge_attr'], dtype=torch.float32)
        data = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t)
        batch = Batch.from_data_list([data, data]).to(Config.DEVICE)

        total_agents = 2 * Config.N_AGENTS
        dummy_obs = torch.randn(total_agents, Config.OBS_DIM).to(Config.DEVICE)

        val = model.get_value(batch, dummy_obs)
        print(f"✅ Critic Value Output Shape: {val.shape}")

    except RuntimeError as e:
        print(f"❌ CRASH in Model Forward: {e}")


if __name__ == "__main__":
    test_gnn_pipeline()