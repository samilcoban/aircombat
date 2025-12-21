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


class RecursiveBlock(nn.Module):
    """
    Tiny Recursive Model (TRM) Block.
    Replaces GRU. Iteratively refines the action draft 'y' and thought 'z'.
    """

    def __init__(self, context_dim, action_dim, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Input: Context(D_MODEL) + Action(Action_Dim) + Latent(Latent_Dim)
        input_total = context_dim + action_dim + latent_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(input_total, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU()
        )

        # Heads
        self.head_z = layer_init(nn.Linear(256, latent_dim), std=0.01)
        self.head_y = layer_init(nn.Linear(256, action_dim), std=0.01)

        # Learnable Initial State
        self.z_init = nn.Parameter(torch.zeros(1, latent_dim))
        self.y_init = nn.Parameter(torch.zeros(1, action_dim))
        with torch.no_grad(): self.y_init[0, 2] = 1.0  # Throttle bias

    def forward(self, context, recursions=3):
        batch_size = context.shape[0]

        # Broadcast initial
        z = self.z_init.expand(batch_size, -1)
        y = self.y_init.expand(batch_size, -1)

        history_y = []

        for _ in range(recursions):
            inp = torch.cat([context, y, z], dim=-1)
            feat = self.net(inp)

            # Delta updates (Residual)
            dz = self.head_z(feat)
            dy = self.head_y(feat)

            z = z + dz
            y = y + dy

            history_y.append(y)

        return y, history_y  # Return final y and history for Deep Supervision


class AirCombatDiscriminator(nn.Module):
    """
    GAIL Discriminator (D(s, a)).
    Distinguishes between Expert (1) and Agent (0) behavior.
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # 1. Shared Physics Encoder
        self.shared_physics_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, 128)),
            nn.ReLU()
        )

        # 2. GNN Layers
        self.gnn_conv1 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)
        self.gnn_conv2 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)

        # 3. Action Encoder
        self.action_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.ACTION_DIM, 64)),
            nn.Tanh()
        )

        # 4. Head
        input_dim = 128 + 128 + 128 + 64
        self.head = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)),
            nn.ReLU(),
            nn.Dropout(0.1),
            layer_init(nn.Linear(128, 1))
        )
        self.attention_scale = 1.0 / np.sqrt(128)

    def forward(self, graph_batch, obs, action):
        # A. Graph
        raw_nodes = graph_batch.x
        node_embeddings = self.shared_physics_encoder(raw_nodes)
        x = torch.relu(self.gnn_conv1(node_embeddings, graph_batch.edge_index, graph_batch.edge_attr))
        gnn_out = torch.relu(self.gnn_conv2(x, graph_batch.edge_index, graph_batch.edge_attr))

        batch_idx = graph_batch.batch if graph_batch.batch is not None else torch.zeros(raw_nodes.size(0),
                                                                                        dtype=torch.long,
                                                                                        device=raw_nodes.device)
        num_graphs = batch_idx.max().item() + 1

        is_ally = (raw_nodes[:, 1] > 0.5)
        is_enemy = (raw_nodes[:, 1] <= 0.5)

        ally_context = global_mean_pool(gnn_out[is_ally], batch_idx[is_ally], size=num_graphs)
        enemy_context = global_mean_pool(gnn_out[is_enemy], batch_idx[is_enemy], size=num_graphs)
        if is_ally.sum() == 0: ally_context = torch.zeros(num_graphs, 128, device=obs.device)
        if is_enemy.sum() == 0: enemy_context = torch.zeros(num_graphs, 128, device=obs.device)

        # B. Attention
        if obs.ndim == 3: obs = obs.reshape(-1, self.cfg.OBS_DIM)
        ego_raw = obs[:, :self.cfg.NODE_DIM]
        query = self.shared_physics_encoder(ego_raw)

        dense_keys, mask = to_dense_batch(node_embeddings, batch_idx)
        dense_vals, _ = to_dense_batch(gnn_out, batch_idx)

        if query.shape[0] >= num_graphs and num_graphs > 0:
            agents_per_env = query.shape[0] // num_graphs
            keys = dense_keys.repeat_interleave(agents_per_env, dim=0)
            vals = dense_vals.repeat_interleave(agents_per_env, dim=0)
            mask_expanded = mask.repeat_interleave(agents_per_env, dim=0)
            ally_context = ally_context.repeat_interleave(agents_per_env, dim=0)
            enemy_context = enemy_context.repeat_interleave(agents_per_env, dim=0)
        else:
            keys, vals, mask_expanded = dense_keys, dense_vals, mask

        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * self.attention_scale
        scores = scores.masked_fill(~mask_expanded.unsqueeze(1), -1e4)
        weights = F.softmax(scores, dim=-1)
        subject_emb = torch.bmm(weights, vals).squeeze(1)

        # C. Fusion
        if action.ndim == 3: action = action.reshape(-1, self.cfg.ACTION_DIM)
        action_emb = self.action_encoder(action)
        fusion = torch.cat([subject_emb, ally_context, enemy_context, action_emb], dim=1)
        return self.head(fusion)


class HybridActorCritic(nn.Module):
    """
    Updated: Uses TRM instead of GRU.
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # Actor Encoders
        self.ego_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL), nn.ReLU(), nn.Dropout(self.cfg.DROPOUT))

        self.edge_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.EDGE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL), nn.ReLU(), nn.Dropout(self.cfg.DROPOUT))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL, nhead=self.cfg.N_HEADS,
            dim_feedforward=self.cfg.D_MODEL * 4, dropout=self.cfg.DROPOUT,
            batch_first=True, norm_first=True
        )
        self.actor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.N_LAYERS)

        # NEW: Recursive Block instead of GRU + Head
        self.trm = RecursiveBlock(self.cfg.D_MODEL, self.cfg.ACTION_DIM, latent_dim=128)

        # Standard Deviation for PPO (Action Sampling)
        self.actor_logstd = nn.Parameter(torch.ones(1, self.cfg.ACTION_DIM) * -0.5)

        # World Model
        self.world_model = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL + self.cfg.ACTION_DIM, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, self.cfg.NODE_DIM + 1), std=1.0)
        )

        # Critic (GNN)
        self.gnn_conv1 = EdgeGCNConv(self.cfg.NODE_DIM, self.cfg.EDGE_DIM, 128)
        self.gnn_conv2 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)
        self.shared_physics_encoder = nn.Sequential(layer_init(nn.Linear(self.cfg.NODE_DIM, 128)), nn.ReLU())
        self.attention_scale = 1.0 / np.sqrt(128)
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL + 256, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def extract_actor_features(self, x, gru_state=None, done=None):
        # NOTE: 'gru_state' is kept in signature for API compatibility but unused by TRM
        has_seq_dim = (x.ndim == 3)
        if has_seq_dim:
            batch, seq, dim = x.shape;
            x_flat = x.reshape(-1, dim)
        else:
            batch, seq = x.shape[0], 1;
            x_flat = x

        ego_raw = x_flat[:, :self.cfg.NODE_DIM]
        track_raw = x_flat[:, self.cfg.NODE_DIM:].reshape(batch * seq, -1, self.cfg.EDGE_DIM)

        ego_emb = self.ego_encoder(ego_raw).unsqueeze(1)
        track_emb = self.edge_encoder(track_raw)

        mask = torch.cat([torch.zeros(batch * seq, 1, dtype=torch.bool, device=x.device), (track_raw[:, :, 0] < 1e-5)],
                         dim=1)
        out = self.actor_transformer(torch.cat([ego_emb, track_emb], dim=1), src_key_padding_mask=mask)

        # Context for TRM
        context = out[:, 0, :]  # (Batch*Seq, D_Model)

        if not has_seq_dim: context = context.squeeze(1)
        return context, None  # No hidden state to carry over

    def get_aux_prediction(self, actor_features, action):
        if actor_features.ndim == 3: actor_features = actor_features.reshape(-1, self.cfg.D_MODEL)
        if action.ndim == 3: action = action.reshape(-1, self.cfg.ACTION_DIM)
        preds = self.world_model(torch.cat([actor_features, action], dim=-1))
        return preds[:, :self.cfg.NODE_DIM], preds[:, -1]

    def _process_critic_graph(self, graph_data):
        x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch
        raw_nodes = x
        node_embeddings = self.shared_physics_encoder(raw_nodes)

        x = torch.relu(self.gnn_conv1(node_embeddings, edge_index, edge_attr))
        x = torch.relu(self.gnn_conv2(x, edge_index, edge_attr))

        is_blue = raw_nodes[:, 1]
        ally_mask = (is_blue > 0.5)
        enemy_mask = (is_blue <= 0.5)

        if batch is None: batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        num_graphs = batch.max().item() + 1

        ally_emb = global_mean_pool(x[ally_mask], batch[ally_mask], size=num_graphs)
        enemy_emb = global_mean_pool(x[enemy_mask], batch[enemy_mask], size=num_graphs)
        if ally_mask.sum() == 0: ally_emb = torch.zeros(num_graphs, 128, device=x.device)
        if enemy_mask.sum() == 0: enemy_emb = torch.zeros(num_graphs, 128, device=x.device)
        return ally_emb, enemy_emb

    def get_value(self, graph_batch, obs, gru_state=None, done=None):
        ally_emb, enemy_emb = self._process_critic_graph(graph_batch)
        actor_features, _ = self.extract_actor_features(obs)
        if actor_features.ndim == 3: actor_features = actor_features.reshape(-1, self.cfg.D_MODEL)

        num_graphs = ally_emb.shape[0]
        num_egos = actor_features.shape[0]
        if num_graphs != num_egos and num_graphs > 0:
            agents = num_egos // num_graphs
            ally_emb = ally_emb.repeat_interleave(agents, dim=0)
            enemy_emb = enemy_emb.repeat_interleave(agents, dim=0)

        critic_input = torch.cat([actor_features, ally_emb, enemy_emb], dim=1)
        return self.critic_head(critic_input)

    def get_action_and_value(self, obs, graph_data=None, action=None, gru_state=None, done=None):
        context, _ = self.extract_actor_features(obs)

        # TRM Forward
        # recursions=3 default in Config
        action_mean, history_y = self.trm(context, recursions=self.cfg.TRM_RECURSIONS)

        logstd_clamped = torch.clamp(self.actor_logstd, -2.0, 2.0)
        action_std = torch.exp(logstd_clamped).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            action = torch.clamp(action, -1.0, 1.0)

        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)

        value = None
        if graph_data is not None:
            # Re-use value logic
            # (Inlined to avoid recursion issues)
            ally_emb, enemy_emb = self._process_critic_graph(graph_data)
            num_graphs = ally_emb.shape[0]
            num_egos = context.shape[0]
            if num_graphs != num_egos and num_graphs > 0:
                agents = num_egos // num_graphs
                ally_emb = ally_emb.repeat_interleave(agents, dim=0)
                enemy_emb = enemy_emb.repeat_interleave(agents, dim=0)
            value = self.critic_head(torch.cat([context, ally_emb, enemy_emb], dim=1))

        return action, log_prob, entropy, value, None  # No GRU state

    def get_action_history(self, obs):
        """Helper for Pretraining to access intermediate TRM outputs"""
        context, _ = self.extract_actor_features(obs)
        _, history_y = self.trm(context, recursions=self.cfg.TRM_RECURSIONS)
        return history_y