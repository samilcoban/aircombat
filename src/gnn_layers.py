# ================================================
# FILE: src/gnn_layers.py
# ================================================
"""
Graph Neural Network layers for relational reasoning.

This module implements custom GNN layers based on PyTorch Geometric's
MessagePassing framework. These layers process the entity graph to
enable the agent to reason about spatial relationships between
aircraft and missiles.

The graph structure:
- Nodes: Entities (aircraft, missiles) with state features
- Edges: Relationships between entities with relational features

The EdgeGCNConv layer uses edge features during message passing,
allowing the network to reason about distance, angles, and other
pairwise relationships.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


# Dimensions based on the unified observation format:
# Node: [Exist, Team, Type, X, Y, Alt, CosH, SinH, SinP, SinR, Spd, G, Fuel, Ammo, Chaff, CM, d_Head, d_Pitch, d_Roll, d_Speed] (20D)
# Edge: [Dist, LX, LY, LZ, ATA, AA, Align, Close, TgtSpd, TgtType, TeamRel, Vis, Tgt_dH, Tgt_dP, Tgt_dR, Tgt_dS] (16D)

class EdgeGCNConv(MessagePassing):
    """
    Graph Convolutional layer with edge feature conditioning.
    
    This layer extends standard GCN by incorporating edge features
    into the message computation. It uses dual aggregation (max + mean)
    to capture both salient features and average neighborhood information.
    
    Message Passing Steps:
    1. Message: Combine source node features with edge features via MLP
    2. Aggregate: Apply both max and mean pooling over neighbors
    3. Update: Combine original node features with aggregated messages
    
    Args:
        node_channels: Input node feature dimension.
        edge_channels: Edge feature dimension.
        out_channels: Output node feature dimension.
    """
    
    def __init__(self, node_channels, edge_channels, out_channels):
        # Use both 'max' and 'mean' aggregations for robust feature extraction.
        # Max captures salient/extreme features, mean captures average context.
        super().__init__(aggr=['max', 'mean'])

        # MLP for computing messages from source node + edge features.
        self.message_mlp = nn.Sequential(
            nn.Linear(node_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

        # MLP for updating node features after aggregation.
        # The aggregation produces 2 * out_channels (Max + Mean concatenated).
        # So input is: Node_Feats + (Max_Aggr) + (Mean_Aggr).
        self.update_mlp = nn.Sequential(
            nn.Linear(node_channels + (2 * out_channels), out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the layer.
        
        Args:
            x: Node features [num_nodes, node_channels].
            edge_index: Edge connectivity [2, num_edges].
            edge_attr: Edge features [num_edges, edge_channels].
            
        Returns:
            Updated node features [num_nodes, out_channels].
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        """
        Compute messages from source nodes to target nodes.
        
        Args:
            x_j: Source node features for each edge.
            edge_attr: Edge features.
            
        Returns:
            Messages [num_edges, out_channels].
        """
        # Concatenate source node features with edge features.
        inputs = torch.cat([x_j, edge_attr], dim=1)
        return self.message_mlp(inputs)

    def update(self, aggr_out, x):
        """
        Update node features using aggregated messages.
        
        Args:
            aggr_out: Aggregated messages [num_nodes, 2*out_channels].
                     Contains concatenated max and mean aggregations.
            x: Original node features.
            
        Returns:
            Updated node features [num_nodes, out_channels].
        """
        # aggr_out shape is [N, 2 * out_channels] because of ['max', 'mean']
        inputs = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(inputs)