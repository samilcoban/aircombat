# ================================================
# FILE: src/model.py
# ================================================
import torch
import torch.nn as nn
import numpy as np
from config import Config
from src.gnn_layers import EdgeGCNConv
from torch_geometric.nn import global_mean_pool


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HybridActorCritic(nn.Module):
    """
    Hybrid Actor-Critic Model.
    UPDATED: Uses Unified Node/Edge Dimensions.
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # --- ACTOR ---
        # Ego Encoder: Consumes Unified Node (Private State)
        self.ego_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, self.cfg.D_MODEL)),
            nn.ReLU()
        )

        # Edge Encoder: Consumes Unified Edge (Public/Sensor State)
        self.edge_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.EDGE_DIM, self.cfg.D_MODEL)),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL,
            nhead=self.cfg.N_HEADS,
            dim_feedforward=512,
            batch_first=True,
            norm_first=True
        )
        self.actor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.N_LAYERS)

        self.actor_gru = nn.GRU(
            input_size=self.cfg.D_MODEL,
            hidden_size=self.cfg.D_MODEL,
            batch_first=True
        )

        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, self.cfg.ACTION_DIM), std=0.01)
        )

        with torch.no_grad():
            self.actor_head[-1].bias[2].fill_(1.0)  # Bias throttle up

        self.actor_logstd = nn.Parameter(torch.ones(1, self.cfg.ACTION_DIM) * -0.5)

        # --- CRITIC (GNN) ---
        # Consumes Nodes and Edges directly
        self.gnn_conv1 = EdgeGCNConv(
            node_channels=self.cfg.NODE_DIM,
            edge_channels=self.cfg.EDGE_DIM,
            out_channels=128
        )
        self.gnn_conv2 = EdgeGCNConv(
            node_channels=128,
            edge_channels=self.cfg.EDGE_DIM,
            out_channels=128
        )

        input_dim = self.cfg.D_MODEL + 128 + 128

        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def _extract_ego_features(self, x, gru_state=None, done=None):
        """
        Extracts features using Transformer + GRU.
        Input x: [Batch, OBS_DIM] or [Batch, Seq, OBS_DIM]
        """
        # 1. Detect Input Shape
        has_seq_dim = (x.ndim == 3)

        if has_seq_dim:
            batch_size, seq_len, obs_dim = x.shape
            # Flatten to (Batch * Seq, Dim)
            x_flat = x.reshape(-1, obs_dim)
        else:
            batch_size = x.shape[0]
            seq_len = 1
            x_flat = x

        # 2. Dual Projection
        # Slicing based on Config Dimensions
        ego_raw = x_flat[:, :self.cfg.NODE_DIM]
        track_raw_flat = x_flat[:, self.cfg.NODE_DIM:]

        num_tracks = self.cfg.MAX_ENTITIES - 1
        # Use reshape to be safe against non-contiguous memory
        track_raw = track_raw_flat.reshape(batch_size * seq_len, num_tracks, self.cfg.EDGE_DIM)

        ego_emb = self.ego_encoder(ego_raw).unsqueeze(1)
        track_emb = self.edge_encoder(track_raw)

        # 3. Transformer
        transformer_input = torch.cat([ego_emb, track_emb], dim=1)

        # Masking: Check distance (index 0) to detect padding
        track_dists = track_raw[:, :, 0]
        track_mask = (track_dists < 1e-5)

        ego_mask = torch.zeros(batch_size * seq_len, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([ego_mask, track_mask], dim=1)

        out = self.actor_transformer(transformer_input, src_key_padding_mask=full_mask)
        ego_out_flat = out[:, 0, :]  # (Batch * Seq, D_Model)

        # 4. GRU
        # Reshape to (Batch, Seq, D_Model)
        ego_gru_in = ego_out_flat.reshape(batch_size, seq_len, self.cfg.D_MODEL)

        if gru_state is None:
            gru_state = torch.zeros(1, batch_size, self.cfg.D_MODEL, device=x.device)

        # Handle resets during ROLLOUT only
        if done is not None and not has_seq_dim:
            gru_state = gru_state * (1.0 - done).view(1, -1, 1)

        gru_out, new_gru_state = self.actor_gru(ego_gru_in, gru_state)

        if not has_seq_dim:
            gru_out = gru_out.squeeze(1)

        return gru_out, new_gru_state

    def _process_critic_graph(self, graph_data):
        x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch
        x = torch.relu(self.gnn_conv1(x, edge_index, edge_attr))
        x = torch.relu(self.gnn_conv2(x, edge_index, edge_attr))

        # Check Team (Index 1 in Unified Node) -> 1.0 is Blue
        is_blue = x[:, 1]
        ally_mask = (is_blue > 0.5)
        enemy_mask = (is_blue <= 0.5)

        # Pooling
        if batch is None:  # Handle case with single graph
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1

        # Handle cases where mask is empty to avoid NaNs
        ally_emb = global_mean_pool(x[ally_mask], batch[ally_mask], size=num_graphs)
        if ally_mask.sum() == 0: ally_emb = torch.zeros(num_graphs, 128, device=x.device)

        enemy_emb = global_mean_pool(x[enemy_mask], batch[enemy_mask], size=num_graphs)
        if enemy_mask.sum() == 0: enemy_emb = torch.zeros(num_graphs, 128, device=x.device)

        return ally_emb, enemy_emb

    def get_value(self, graph_batch, obs, gru_state=None, done=None):
        ally_emb, enemy_emb = self._process_critic_graph(graph_batch)

        # Ego Embeddings
        ego_emb, _ = self._extract_ego_features(obs, gru_state, done)

        # Flatten sequence dimension if present
        if ego_emb.ndim == 3:
            ego_emb = ego_emb.reshape(-1, self.cfg.D_MODEL)

        # Graph batch size check
        num_graphs = ally_emb.shape[0]
        num_egos = ego_emb.shape[0]

        if num_graphs != num_egos and num_graphs > 0:
            agents_per_env = num_egos // num_graphs
            ally_emb = ally_emb.repeat_interleave(agents_per_env, dim=0)
            enemy_emb = enemy_emb.repeat_interleave(agents_per_env, dim=0)

        critic_input = torch.cat([ego_emb, ally_emb, enemy_emb], dim=1)
        return self.critic_head(critic_input)

    def get_action_and_value(self, obs, graph_data=None, action=None, gru_state=None, done=None):
        # 1. Actor
        ego_emb, new_gru_state = self._extract_ego_features(obs, gru_state, done)

        action_mean = self.actor_head(ego_emb)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            action = torch.clamp(action, -1.0, 1.0)  # Explicit Clamp

        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)

        # 2. Critic
        value = None
        if graph_data is not None:
            # Flatten sequence dimension for Critic
            critic_ego = ego_emb.reshape(-1, self.cfg.D_MODEL) if ego_emb.ndim == 3 else ego_emb

            ally_emb, enemy_emb = self._process_critic_graph(graph_data)

            num_graphs = ally_emb.shape[0]
            num_egos = critic_ego.shape[0]

            if num_graphs != num_egos and num_graphs > 0:
                agents_per_env = num_egos // num_graphs
                ally_emb = ally_emb.repeat_interleave(agents_per_env, dim=0)
                enemy_emb = enemy_emb.repeat_interleave(agents_per_env, dim=0)

            critic_input = torch.cat([critic_ego, ally_emb, enemy_emb], dim=1)
            value = self.critic_head(critic_input)

            # Reshape value back to (Batch, Seq, 1) if necessary
            if ego_emb.ndim == 3:
                value = value.reshape(ego_emb.shape[0], ego_emb.shape[1], 1)

        return action, log_prob, entropy, value, new_gru_state