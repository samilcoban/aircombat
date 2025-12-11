# ================================================
# FILE: src/model.py
# ================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config
from src.gnn_layers import EdgeGCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HybridActorCritic(nn.Module):
    """
    Decoupled Commander Architecture with Attention-Based Linking.

    - Actor: Standard Transformer+GRU (Tactical).
    - Critic: GNN (Strategic).
      It uses a Shared Encoder to match the Agent's Local Identity (Query)
      to the Graph's Node Identities (Keys) to extract the Context-Aware State (Values).
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # =================================================================
        # 1. SHARED COMPONENTS (Identity Matching)
        # =================================================================
        # This small MLP ensures that "Raw Physics" look the same
        # whether they come from the Graph or the Local Observation.
        # Used to generate Keys (from Graph) and Queries (from Obs).
        self.shared_physics_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, 128)),
            nn.ReLU()
        )

        # =================================================================
        # 2. ACTOR (Local Transformer + Memory)
        # =================================================================
        self.ego_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL), nn.ReLU())

        self.edge_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.EDGE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL), nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL, nhead=self.cfg.N_HEADS,
            dim_feedforward=self.cfg.D_MODEL * 4, batch_first=True, norm_first=True
        )
        self.actor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.N_LAYERS)

        self.actor_gru = nn.GRU(self.cfg.D_MODEL, self.cfg.D_MODEL, batch_first=True)

        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, self.cfg.ACTION_DIM), std=0.01)
        )

        # Initialize throttle bias high to prevent stalling at start
        with torch.no_grad(): self.actor_head[-1].bias[2].fill_(1.0)
        self.actor_logstd = nn.Parameter(torch.ones(1, self.cfg.ACTION_DIM) * -0.5)

        # Auxiliary World Model (Attached to Actor)
        self.world_model = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL + self.cfg.ACTION_DIM, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, self.cfg.NODE_DIM + 1), std=1.0)
        )

        # =================================================================
        # 3. CRITIC (Commander with Attention)
        # =================================================================

        # GNN Layers (Process the Shared Embeddings)
        # Input is 128 (from shared_physics_encoder), not NODE_DIM
        self.gnn_conv1 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)
        self.gnn_conv2 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)

        self.attention_scale = 1.0 / np.sqrt(128)

        # Critic Head
        # Inputs: [Subject_GNN_State (128) | Ally_Context (128) | Enemy_Context (128)]
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(128 + 128 + 128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def extract_actor_features(self, x, gru_state=None, done=None):
        """Standard Actor Pipeline: Enc -> Transformer -> GRU"""
        has_seq_dim = (x.ndim == 3)
        if has_seq_dim:
            batch, seq, dim = x.shape; x_flat = x.reshape(-1, dim)
        else:
            batch, seq = x.shape[0], 1; x_flat = x

        ego_raw = x_flat[:, :self.cfg.NODE_DIM]
        track_raw = x_flat[:, self.cfg.NODE_DIM:].reshape(batch * seq, -1, self.cfg.EDGE_DIM)

        ego_emb = self.ego_encoder(ego_raw).unsqueeze(1)
        track_emb = self.edge_encoder(track_raw)

        # Mask padding (distance=0 tracks)
        mask = torch.cat([torch.zeros(batch * seq, 1, dtype=torch.bool, device=x.device), (track_raw[:, :, 0] < 1e-5)],
                         dim=1)

        out = self.actor_transformer(torch.cat([ego_emb, track_emb], dim=1), src_key_padding_mask=mask)

        # Extract Ego Token
        gru_in = out[:, 0].reshape(batch, seq, -1)
        if gru_state is None: gru_state = torch.zeros(1, batch, self.cfg.D_MODEL, device=x.device)

        # GRU Handling
        if has_seq_dim and done is not None:
            outs = []
            for t in range(seq):
                gru_state = gru_state * (1.0 - done[:, t].view(1, -1, 1))
                o, gru_state = self.actor_gru(gru_in[:, t:t + 1], gru_state)
                outs.append(o)
            res = torch.cat(outs, dim=1)
        else:
            if done is not None: gru_state = gru_state * (1.0 - done.view(1, -1, 1))
            res, gru_state = self.actor_gru(gru_in, gru_state)

        return (res.squeeze(1) if not has_seq_dim else res), gru_state

    def get_aux_prediction(self, actor_features, action):
        """World Model Prediction (Next State + Reward)"""
        if actor_features.ndim == 3: actor_features = actor_features.reshape(-1, self.cfg.D_MODEL)
        if action.ndim == 3: action = action.reshape(-1, self.cfg.ACTION_DIM)
        preds = self.world_model(torch.cat([actor_features, action], dim=-1))
        return preds[:, :self.cfg.NODE_DIM], preds[:, -1]

    def get_value(self, graph_batch, obs, gru_state=None, done=None):
        """
        Critic Pass:
        1. Encode Graph Nodes (Keys) and Local Obs (Query) using Shared Encoder.
        2. Run GNN on Graph Nodes to get Context (Values).
        3. Match Query to Keys to find 'Me' in the graph.
        4. Extract 'My' GNN State.
        """
        # A. Encode Graph (Keys & Pre-GNN Values)
        # raw_nodes: [Total_Nodes, 16]
        raw_nodes = graph_batch.x
        node_embeddings = self.shared_physics_encoder(raw_nodes)  # [Total_Nodes, 128]

        # B. Run GNN (Context-Aware Values)
        # Input to GNN is now the 128d embedding
        gnn_out = torch.relu(self.gnn_conv1(node_embeddings, graph_batch.edge_index, graph_batch.edge_attr))
        gnn_out = torch.relu(
            self.gnn_conv2(gnn_out, graph_batch.edge_index, graph_batch.edge_attr))  # [Total_Nodes, 128]

        # C. Global Context Pooling
        is_ally = (graph_batch.x[:, 1] > 0.5)
        is_enemy = (graph_batch.x[:, 1] <= 0.5)

        if graph_batch.batch is None:
            batch_idx = torch.zeros(raw_nodes.shape[0], dtype=torch.long, device=raw_nodes.device)
        else:
            batch_idx = graph_batch.batch

        num_graphs = batch_idx.max().item() + 1

        ally_context = global_mean_pool(gnn_out[is_ally], batch_idx[is_ally], size=num_graphs)
        enemy_context = global_mean_pool(gnn_out[is_enemy], batch_idx[is_enemy], size=num_graphs)

        if is_ally.sum() == 0: ally_context = torch.zeros(num_graphs, 128, device=raw_nodes.device)
        if is_enemy.sum() == 0: enemy_context = torch.zeros(num_graphs, 128, device=raw_nodes.device)

        # D. Entity Linking (Attention)

        # 1. Prepare Query (Local Obs -> Shared Encoder)
        if obs.ndim == 3:
            obs_flat = obs.reshape(-1, self.cfg.OBS_DIM)
        else:
            obs_flat = obs

        # Slice only the 'Self' part (first 16 dims) to match Node features
        ego_raw = obs_flat[:, :self.cfg.NODE_DIM]
        query = self.shared_physics_encoder(ego_raw)  # [Total_Agents, 128]

        # 2. Prepare Keys (Original Embeddings) & Values (GNN Output)
        # Reshape to [Num_Graphs, Max_Nodes, 128]
        dense_keys, mask = to_dense_batch(node_embeddings, batch_idx)
        dense_values, _ = to_dense_batch(gnn_out, batch_idx)

        # 3. Broadcast
        # Match Agents to Graphs (e.g., 2 agents per env means repeating graphs 2x)
        if query.shape[0] >= num_graphs and num_graphs > 0:
            agents_per_env = query.shape[0] // num_graphs
        else:
            agents_per_env = 1  # Fallback for single batch inference

        keys = dense_keys.repeat_interleave(agents_per_env, dim=0)  # [Total_Agents, Max_Nodes, 128]
        vals = dense_values.repeat_interleave(agents_per_env, dim=0)  # [Total_Agents, Max_Nodes, 128]
        mask_expanded = mask.repeat_interleave(agents_per_env, dim=0)

        # 4. Dot Product Attention (Soft Matching)
        # query: [Total_Agents, 128] -> [Total_Agents, 1, 128]
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * self.attention_scale

        # Masking padding nodes
        scores = scores.masked_fill(~mask_expanded.unsqueeze(1), -1e4)
        attn_weights = F.softmax(scores, dim=-1)  # [Total_Agents, 1, Max_Nodes]

        # 5. Extract Subject GNN State
        subject_gnn_emb = torch.bmm(attn_weights, vals).squeeze(1)  # [Total_Agents, 128]

        # E. Fusion
        ally_context = ally_context.repeat_interleave(agents_per_env, dim=0)
        enemy_context = enemy_context.repeat_interleave(agents_per_env, dim=0)

        critic_input = torch.cat([subject_gnn_emb, ally_context, enemy_context], dim=1)
        value = self.critic_head(critic_input)

        if obs.ndim == 3: value = value.reshape(obs.shape[0], obs.shape[1], 1)
        return value

    def get_action_and_value(self, obs, graph_data=None, action=None, gru_state=None, done=None):
        # 1. Actor Pipeline
        actor_features, new_gru_state = self.extract_actor_features(obs, gru_state, done)
        action_mean = self.actor_head(actor_features)

        logstd = torch.clamp(self.actor_logstd, -10.0, 2.0)
        std = torch.exp(logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, std)

        if action is None: action = torch.clamp(probs.sample(), -1.0, 1.0)
        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)

        # 2. Critic Pipeline (Decoupled)
        value = None
        if graph_data is not None:
            value = self.get_value(graph_data, obs)

        return action, log_prob, entropy, value, new_gru_state