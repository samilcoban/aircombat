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
    def __init__(self):
        super().__init__()
        self.cfg = Config

        # ==========================================
        # 1. ACTOR (Transformer + GRU)
        # ==========================================

        # Embed Entity Features
        self.actor_embed = nn.Sequential(
            layer_init(nn.Linear(self.cfg.FEAT_DIM, self.cfg.D_MODEL)),
            nn.ReLU()
        )

        # Transformer Encoder (spatial context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL,
            nhead=self.cfg.N_HEADS,
            dim_feedforward=512,
            batch_first=True,
            norm_first=True
        )
        self.actor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.N_LAYERS)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg.D_MODEL))

        # GRU (Temporal Memory) - Replaces LSTM
        self.actor_gru = nn.GRU(
            input_size=self.cfg.D_MODEL,
            hidden_size=self.cfg.D_MODEL,
            batch_first=True
        )

        # Action Head
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, self.cfg.ACTION_DIM), std=0.01)
        )

        # Bias throttle high (index 2) to prevent stalling early
        with torch.no_grad():
            self.actor_head[-1].bias[2].fill_(1.0)

        self.actor_logstd = nn.Parameter(torch.zeros(1, self.cfg.ACTION_DIM))

        # ==========================================
        # 2. CRITIC (GNN - Global Graph)
        # ==========================================
        # Node: [x, y, z, vx, vy, vz, speed, fuel, ammo, is_missile, is_blue, g_load] (12)
        # Edge: [dist, ata, aa, hdg_diff, closure, same_team] (6)

        self.gnn_conv1 = EdgeGCNConv(node_channels=12, edge_channels=6, out_channels=128)
        self.gnn_conv2 = EdgeGCNConv(node_channels=128, edge_channels=6, out_channels=128)

        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def get_actor_features(self, x, gru_state=None, done=None):
        # x: (Batch, Entities, Feat_Dim)
        batch_size = x.shape[0]

        # 1. Masking & Embedding
        # Assuming index 5 is 'team' (0=padding)
        entity_teams = x[:, :, 5]
        mask = (entity_teams == 0.0)

        emb = self.actor_embed(x)

        # 2. Add CLS Token
        cls = self.cls_token.expand(batch_size, -1, -1)
        emb = torch.cat([cls, emb], dim=1)

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        # 3. Transformer
        out = self.actor_transformer(emb, src_key_padding_mask=full_mask)

        # Extract Ego Embedding (Index 1, since 0 is CLS)
        ego_emb = out[:, 1, :].unsqueeze(1)  # (Batch, 1, D_Model)

        # 4. GRU Layer
        if gru_state is None:
            gru_state = torch.zeros(1, batch_size, self.cfg.D_MODEL, device=x.device)

        # Handle done masking for GRU state
        if done is not None:
            gru_state = gru_state * (1.0 - done).view(1, -1, 1)

        gru_out, new_gru_state = self.actor_gru(ego_emb, gru_state)

        return gru_out.squeeze(1), new_gru_state

    def get_value(self, graph_batch):
        """
        graph_batch: PyG Batch object containing aggregated graphs from all envs
        """
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch

        # GNN Layers
        x = self.gnn_conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.gnn_conv2(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Global Pooling: Collapses N nodes -> Batch_Size vectors
        graph_emb = global_mean_pool(x, batch)

        return self.critic_head(graph_emb)

    def get_action_and_value(self, obs, graph_data=None, action=None, gru_state=None, done=None):
        # 1. Actor Pass
        # obs shape needs to be (Batch, Max_Ent, Feat)
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], self.cfg.MAX_ENTITIES, self.cfg.FEAT_DIM)

        actor_feat, new_gru_state = self.get_actor_features(obs, gru_state, done)

        action_mean = self.actor_head(actor_feat)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        # 2. Critic Pass (Optional)
        value = None
        if graph_data is not None:
            value = self.get_value(graph_data)

        return action, log_prob, entropy, value, new_gru_state