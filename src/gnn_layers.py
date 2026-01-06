# ================================================
# FILE: src/gnn_layers.py
# ================================================
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


# Dimensions based on the logic below
# Node: [Lat, Lon, Alt, SinH, CosH, Speed, Fuel, Ammo, Team, Type] (Approx 10)
# Edge: [Dist, AA, ATA, HeadingDiff, ClosingSpeed, IsSameTeam] (6)

class EdgeGCNConv(MessagePassing):
    def __init__(self, node_channels, edge_channels, out_channels):
        # Use both 'max' and 'mean' aggregations
        super().__init__(aggr=['max', 'mean'])

        self.message_mlp = nn.Sequential(
            nn.Linear(node_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

        # The aggregation produces 2 * out_channels (Max + Mean)
        # So input is: Node_Feats + (Max_Aggr) + (Mean_Aggr)
        self.update_mlp = nn.Sequential(
            nn.Linear(node_channels + (2 * out_channels), out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        inputs = torch.cat([x_j, edge_attr], dim=1)
        return self.message_mlp(inputs)

    def update(self, aggr_out, x):
        # aggr_out shape is [N, 2 * out_channels] because of ['max', 'mean']
        inputs = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(inputs)