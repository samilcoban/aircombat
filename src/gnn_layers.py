import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


# Dimensions based on the logic below
# Node: [Lat, Lon, Alt, SinH, CosH, Speed, Fuel, Ammo, Team, Type] (Approx 10)
# Edge: [Dist, AA, ATA, HeadingDiff, ClosingSpeed, IsSameTeam] (6)

class EdgeGCNConv(MessagePassing):
    """
    A custom message passing layer that incorporates edge features into the message
    before aggregation. Inspired by hhmarl_2d.
    """

    def __init__(self, node_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')

        # MLP processing source node + edge features
        self.message_mlp = nn.Sequential(
            nn.Linear(node_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

        # MLP updating target node with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(node_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: Source node features [E, node_dim]
        # edge_attr: Edge features [E, edge_dim]
        inputs = torch.cat([x_j, edge_attr], dim=1)
        return self.message_mlp(inputs)

    def update(self, aggr_out, x):
        # x: Target node features [N, node_dim]
        # aggr_out: Aggregated messages [N, out_dim]
        inputs = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(inputs)