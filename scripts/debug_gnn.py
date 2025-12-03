#!/usr/bin/env python3
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
    print("Testing Env -> Graph -> Model Pipeline...")

    # 1. Setup Env and Model
    env = AirCombatEnv()
    env.reset()
    model = HybridActorCritic().to(Config.DEVICE)
    model.eval()

    # 2. Force Empty Edges Case (Single Agent)
    # Kill the enemy to remove edges
    if env.red_ids:
        del env.core.entities[env.red_ids[0]]
        env.red_ids = []
    env.core.update_spatial_cache()

    # 3. Get Graph State
    gd = env._get_graph_state()
    print(f"Graph Data Keys: {gd.keys()}")
    print(f"X Shape: {gd['x'].shape}")
    print(f"Edge Index Shape: {gd['edge_index'].shape}")

    # 4. Convert to Tensor (Mimic train.py)
    x_t = torch.tensor(gd['x'], dtype=torch.float32)
    edge_index_t = torch.tensor(gd['edge_index'], dtype=torch.long)
    edge_attr_t = torch.tensor(gd['edge_attr'], dtype=torch.float32)

    # 5. Create Batch
    data = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t)
    batch = Batch.from_data_list([data, data]).to(Config.DEVICE)  # Batch of 2

    print("Running Model.get_value() with GNN...")
    try:
        # Create dummy obs for the actor part
        dummy_obs = torch.randn(2, Config.OBS_DIM).to(Config.DEVICE)

        val = model.get_value(batch, dummy_obs)
        print(f"✅ Success! Value output shape: {val.shape}")
    except RuntimeError as e:
        print(f"❌ CRASH: {e}")


if __name__ == "__main__":
    test_gnn_pipeline()