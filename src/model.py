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
    Hybrid Actor-Critic Model for Air Combat.
    """
    def __init__(self):
        super().__init__()
        self.cfg = Config

        # ==========================================
        # 1. SHARED / ACTOR ENCODER
        # ==========================================
        self.actor_embed = nn.Sequential(
            layer_init(nn.Linear(self.cfg.FEAT_DIM, self.cfg.D_MODEL)),
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg.D_MODEL))

        # GRU (Temporal Memory)
        self.actor_gru = nn.GRU(
            input_size=self.cfg.D_MODEL,
            hidden_size=self.cfg.D_MODEL,
            batch_first=True
        )

        # ==========================================
        # 2. ACTOR HEAD
        # ==========================================
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, self.cfg.ACTION_DIM), std=0.01)
        )

        # Bias throttle high
        with torch.no_grad():
            self.actor_head[-1].bias[2].fill_(1.0)

        self.actor_logstd = nn.Parameter(torch.zeros(1, self.cfg.ACTION_DIM))

        # ==========================================
        # 3. CRITIC (GNN + EGO FUSION)
        # ==========================================
        # Node features (12) + Edge features (6)
        self.gnn_conv1 = EdgeGCNConv(node_channels=12, edge_channels=6, out_channels=128)
        self.gnn_conv2 = EdgeGCNConv(node_channels=128, edge_channels=6, out_channels=128)

        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(128 + self.cfg.D_MODEL, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def _extract_ego_features(self, x, gru_state=None, done=None):
        """
        Runs the Actor's Transformer/GRU.
        """
        batch_size = x.shape[0]

        # Masking (Team=0 is padding)
        # FIX: Updated index from 5 to 17 to match new Relative Obs Space
        entity_teams = x[:, :, 17]
        mask = (entity_teams == 0.0)

        # Embedding
        emb = self.actor_embed(x)

        # CLS Token
        cls = self.cls_token.expand(batch_size, -1, -1)
        emb = torch.cat([cls, emb], dim=1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer
        out = self.actor_transformer(emb, src_key_padding_mask=full_mask)

        # Extract Ego Embedding (Index 1)
        # Index 0 is CLS, Index 1 is Ego
        ego_emb = out[:, 1, :].unsqueeze(1)  # (Batch, 1, D_Model)

        # GRU
        if gru_state is None:
            gru_state = torch.zeros(1, batch_size, self.cfg.D_MODEL, device=x.device)

        if done is not None:
            gru_state = gru_state * (1.0 - done).view(1, -1, 1)

        gru_out, new_gru_state = self.actor_gru(ego_emb, gru_state)

        return gru_out.squeeze(1), new_gru_state

    def get_value(self, graph_batch, obs, gru_state=None, done=None):
        """
        Calculates Value V(s) specific to the agent in 'obs'.
        """
        # 1. Process Global Graph
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch
        x = torch.relu(self.gnn_conv1(x, edge_index, edge_attr))
        x = torch.relu(self.gnn_conv2(x, edge_index, edge_attr))

        # Global Embedding: [N_Envs, 128]
        global_emb = global_mean_pool(x, batch)

        # 2. Expand Global Embedding
        num_envs = global_emb.shape[0]
        total_agents = obs.shape[0]
        agents_per_env = total_agents // num_envs
        global_emb_expanded = global_emb.repeat_interleave(agents_per_env, dim=0)

        # 3. Process Local Ego
        if obs.dim() == 2:
            obs = obs.view(total_agents, self.cfg.MAX_ENTITIES, self.cfg.FEAT_DIM)

        ego_emb, _ = self._extract_ego_features(obs, gru_state, done)

        # 4. Fuse
        critic_input = torch.cat([global_emb_expanded, ego_emb], dim=1)

        return self.critic_head(critic_input)

    def get_action_and_value(self, obs, graph_data=None, action=None, gru_state=None, done=None):
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], self.cfg.MAX_ENTITIES, self.cfg.FEAT_DIM)

        ego_emb, new_gru_state = self._extract_ego_features(obs, gru_state, done)

        action_mean = self.actor_head(ego_emb)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        value = None
        if graph_data is not None:
            x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch
            x = torch.relu(self.gnn_conv1(x, edge_index, edge_attr))
            x = torch.relu(self.gnn_conv2(x, edge_index, edge_attr))
            global_emb = global_mean_pool(x, batch)

            num_envs = global_emb.shape[0]
            total_agents = obs.shape[0]
            agents_per_env = total_agents // num_envs
            global_emb_expanded = global_emb.repeat_interleave(agents_per_env, dim=0)

            critic_input = torch.cat([global_emb_expanded, ego_emb], dim=1)
            value = self.critic_head(critic_input)

        return action, log_prob, entropy, value, new_gru_state