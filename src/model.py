# ================================================
# FILE: src/model.py
# ================================================
import torch
import torch.nn as nn
import numpy as np
from config import Config
from src.gnn_layers import EdgeGCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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

        # Broadcast Keys/Values if num_agents > num_graphs
        if query.shape[0] >= num_graphs and num_graphs > 0:
            agents_per_env = query.shape[0] // num_graphs
            keys = dense_keys.repeat_interleave(agents_per_env, dim=0)
            vals = dense_vals.repeat_interleave(agents_per_env, dim=0)
            mask_expanded = mask.repeat_interleave(agents_per_env, dim=0)
            ally_context = ally_context.repeat_interleave(agents_per_env, dim=0)
            enemy_context = enemy_context.repeat_interleave(agents_per_env, dim=0)
        else:
            keys, vals, mask_expanded = dense_keys, dense_vals, mask

        # Attention: Q * K^T
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * self.attention_scale
        scores = scores.masked_fill(~mask_expanded.unsqueeze(1), -1e4)
        weights = F.softmax(scores, dim=-1)
        subject_emb = torch.bmm(weights, vals).squeeze(1)

        # C. Fusion
        if action.ndim == 3: action = action.reshape(-1, self.cfg.ACTION_DIM)
        action_emb = self.action_encoder(action)
        fusion = torch.cat([subject_emb, ally_context, enemy_context, action_emb], dim=1)
        return self.head(fusion)


class RecursiveActionHead(nn.Module):
    """
    TRM (Tiny Recursive Model) Implementation for Action Refinement.
    """

    def __init__(self, input_dim, action_dim, latent_dim=64, recursions=3):
        super().__init__()
        self.recursions = recursions
        self.action_dim = action_dim

        # The Core Network: Inputs (Context + Action + Latent) -> Hidden
        input_total = input_dim + action_dim + latent_dim
        self.core_net = nn.Sequential(
            layer_init(nn.Linear(input_total, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU()
        )

        # Update Heads: Hidden -> Delta Y (Action) and Delta Z (Latent)
        self.head_action = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.head_latent = layer_init(nn.Linear(256, latent_dim), std=0.01)

        # Learnable Initial States
        self.y_init = nn.Parameter(torch.zeros(1, action_dim))
        self.z_init = nn.Parameter(torch.zeros(1, latent_dim))

    def forward(self, x, return_history=False):
        """
        x: Context vector (Batch, D_MODEL) or (Batch, Seq_Len, D_MODEL)
        return_history: If True, returns list of all 'y' steps (for Deep Supervision)
        """
        # --- FIX: Handle 3D (Sequence) vs 2D (Flat) Inputs ---
        if x.ndim == 3:
            B, T, _ = x.shape
            # Expand init to (Batch, Seq, Dim)
            y = self.y_init.view(1, 1, -1).expand(B, T, -1)
            z = self.z_init.view(1, 1, -1).expand(B, T, -1)
        else:
            B, _ = x.shape
            # Expand init to (Batch, Dim)
            y = self.y_init.expand(B, -1)
            z = self.z_init.expand(B, -1)
        # -----------------------------------------------------

        history = []

        # 2. Recursive Loop
        steps_with_grad = self.recursions if return_history else 1
        steps_no_grad = 0 if return_history else (self.recursions - 1)

        # A. Thinking Phase (Inference Optimization)
        if steps_no_grad > 0:
            with torch.no_grad():
                for _ in range(steps_no_grad):
                    inp = torch.cat([x, y, z], dim=-1)
                    feat = self.core_net(inp)
                    y = y + 0.1 * self.head_action(feat)
                    z = z + 0.1 * self.head_latent(feat)

        # B. Learning Phase (Gradient Tracking)
        for _ in range(steps_with_grad):
            inp = torch.cat([x, y, z], dim=-1)
            feat = self.core_net(inp)

            y = y + 0.1 * self.head_action(feat)
            z = z + 0.1 * self.head_latent(feat)

            if return_history:
                history.append(y)

        if return_history:
            return history

        return y


class HybridActorCritic(nn.Module):
    """
    Hybrid Architecture:
    - Actor: Transformer + GRU + TRM Head
    - Critic: GNN
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # =================================================================
        # 1. ACTOR (Local Transformer)
        # =================================================================
        self.ego_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL),
            nn.ReLU()
        )

        self.edge_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.EDGE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL,
            nhead=self.cfg.N_HEADS,
            dim_feedforward=self.cfg.D_MODEL * 4,
            batch_first=True,
            norm_first=True
        )
        self.actor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.N_LAYERS)

        self.actor_gru = nn.GRU(
            input_size=self.cfg.D_MODEL,
            hidden_size=self.cfg.D_MODEL,
            batch_first=True
        )

        # TRM Head
        self.actor_head = RecursiveActionHead(
            input_dim=self.cfg.D_MODEL,
            action_dim=self.cfg.ACTION_DIM,
            latent_dim=64,
            recursions=3
        )

        # Init bias for throttle (index 2)
        with torch.no_grad():
            self.actor_head.y_init[0, 2] = 1.0

        self.actor_logstd = nn.Parameter(torch.ones(1, self.cfg.ACTION_DIM) * -0.5)

        # =================================================================
        # 2. AUXILIARY WORLD MODEL
        # =================================================================
        self.world_model = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL + self.cfg.ACTION_DIM, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, self.cfg.NODE_DIM + 1), std=1.0)
        )

        # =================================================================
        # 3. CRITIC (Global GNN)
        # =================================================================
        # Shared Physics Encoder Maps NODE_DIM (20) -> 128
        self.shared_physics_encoder = nn.Sequential(layer_init(nn.Linear(self.cfg.NODE_DIM, 128)), nn.ReLU())

        self.gnn_conv1 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)
        self.gnn_conv2 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)

        self.attention_scale = 1.0 / np.sqrt(128)

        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(128 + 128 + 128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def extract_actor_features(self, x, gru_state=None, done=None):
        """Actor perception stack."""
        has_seq_dim = (x.ndim == 3)

        if has_seq_dim:
            batch_size, seq_len, obs_dim = x.shape
            x_flat = x.reshape(-1, obs_dim)
        else:
            batch_size = x.shape[0]
            seq_len = 1
            x_flat = x

        ego_raw = x_flat[:, :self.cfg.NODE_DIM]
        track_raw_flat = x_flat[:, self.cfg.NODE_DIM:]

        num_tracks = self.cfg.MAX_ENTITIES - 1
        track_raw = track_raw_flat.reshape(batch_size * seq_len, num_tracks, self.cfg.EDGE_DIM)

        ego_emb = self.ego_encoder(ego_raw).unsqueeze(1)
        track_emb = self.edge_encoder(track_raw)

        transformer_input = torch.cat([ego_emb, track_emb], dim=1)

        track_dists = track_raw[:, :, 0]
        track_mask = (track_dists < 1e-5)
        ego_mask = torch.zeros(batch_size * seq_len, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([ego_mask, track_mask], dim=1)

        out = self.actor_transformer(transformer_input, src_key_padding_mask=full_mask)
        ego_out_flat = out[:, 0, :]

        # GRU Logic
        ego_gru_in = ego_out_flat.reshape(batch_size, seq_len, self.cfg.D_MODEL)

        if gru_state is None:
            gru_state = torch.zeros(1, batch_size, self.cfg.D_MODEL, device=x.device)

        if has_seq_dim and done is not None:
            outputs = []
            for t in range(seq_len):
                current_mask = 1.0 - done[:, t].view(1, -1, 1)
                gru_state = gru_state * current_mask
                step_input = ego_gru_in[:, t:t + 1, :]
                step_out, gru_state = self.actor_gru(step_input, gru_state)
                outputs.append(step_out)
            gru_out = torch.cat(outputs, dim=1)
            new_gru_state = gru_state
        else:
            if done is not None and not has_seq_dim:
                mask = 1.0 - done.view(1, -1, 1)
                gru_state = gru_state * mask
            gru_out, new_gru_state = self.actor_gru(ego_gru_in, gru_state)

        if not has_seq_dim:
            gru_out = gru_out.squeeze(1)

        return gru_out, new_gru_state

    def get_aux_prediction(self, actor_features, action):
        if actor_features.ndim == 3: actor_features = actor_features.reshape(-1, self.cfg.D_MODEL)
        if action.ndim == 3: action = action.reshape(-1, self.cfg.ACTION_DIM)
        inp = torch.cat([actor_features, action], dim=-1)
        preds = self.world_model(inp)
        return preds[:, :self.cfg.NODE_DIM], preds[:, -1]

    def _process_critic_graph(self, graph_data, obs_flat):
        """Critic Graph Processing with Attention."""
        x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch

        # 1. GNN
        node_embeddings = self.shared_physics_encoder(x)
        x_gnn = torch.relu(self.gnn_conv1(node_embeddings, edge_index, edge_attr))
        gnn_out = torch.relu(self.gnn_conv2(x_gnn, edge_index, edge_attr))

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        num_graphs = batch.max().item() + 1

        is_blue = x[:, 1]
        ally_mask = (is_blue > 0.5)
        enemy_mask = (is_blue <= 0.5)

        ally_emb = global_mean_pool(gnn_out[ally_mask], batch[ally_mask], size=num_graphs)
        enemy_emb = global_mean_pool(gnn_out[enemy_mask], batch[enemy_mask], size=num_graphs)

        if ally_mask.sum() == 0: ally_emb = torch.zeros(num_graphs, 128, device=x.device)
        if enemy_mask.sum() == 0: enemy_emb = torch.zeros(num_graphs, 128, device=x.device)

        # 2. Attention
        ego_raw = obs_flat[:, :self.cfg.NODE_DIM]
        query = self.shared_physics_encoder(ego_raw)

        dense_keys, mask = to_dense_batch(node_embeddings, batch)
        dense_vals, _ = to_dense_batch(gnn_out, batch)

        if query.shape[0] >= num_graphs and num_graphs > 0:
            agents_per_env = query.shape[0] // num_graphs
            keys = dense_keys.repeat_interleave(agents_per_env, dim=0)
            vals = dense_vals.repeat_interleave(agents_per_env, dim=0)
            mask_expanded = mask.repeat_interleave(agents_per_env, dim=0)
            ally_emb = ally_emb.repeat_interleave(agents_per_env, dim=0)
            enemy_emb = enemy_emb.repeat_interleave(agents_per_env, dim=0)
        else:
            keys, vals, mask_expanded = dense_keys, dense_vals, mask

        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * self.attention_scale
        scores = scores.masked_fill(~mask_expanded.unsqueeze(1), -1e4)
        weights = F.softmax(scores, dim=-1)
        subject_emb = torch.bmm(weights, vals).squeeze(1)

        return subject_emb, ally_emb, enemy_emb

    def get_value(self, graph_batch, obs, gru_state=None, done=None):
        if obs.ndim == 3:
            obs_flat = obs.reshape(-1, self.cfg.OBS_DIM)
        else:
            obs_flat = obs

        subject_emb, ally_emb, enemy_emb = self._process_critic_graph(graph_batch, obs_flat)
        critic_input = torch.cat([subject_emb, ally_emb, enemy_emb], dim=1)
        value = self.critic_head(critic_input)

        if obs.ndim == 3:
            value = value.view(obs.shape[0], obs.shape[1], 1)
        return value

    def get_action_and_value(self, obs, graph_data=None, action=None, gru_state=None, done=None):
        # 1. Actor Pipeline
        actor_features, new_gru_state = self.extract_actor_features(obs, gru_state, done)

        # 2. Action Head (Recursive TRM)
        action_mean = self.actor_head(actor_features)

        # Flatten Mean if 3D to match PPO's flattened action input
        if action_mean.ndim == 3:
            action_mean = action_mean.reshape(-1, self.cfg.ACTION_DIM)

        # Clamp logstd
        logstd_clamped = torch.clamp(self.actor_logstd, -10.0, 2.0)
        action_std = torch.exp(logstd_clamped).expand_as(action_mean)

        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            action = torch.clamp(action, -1.0, 1.0)
        else:
            # Flatten action if it comes in as 3D (Batch, Seq, Dim)
            if action.ndim == 3:
                action = action.reshape(-1, self.cfg.ACTION_DIM)

        # Sum log probs over action dimension
        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)

        # 3. Critic Pipeline (GNN)
        value = None
        if graph_data is not None:
            # FIX 1: Pass RAW OBS (flattened) to the critic, not actor_features.
            # The Attention mechanism needs the raw node features (20 dims) for the query.
            if obs.ndim == 3:
                obs_flat = obs.reshape(-1, self.cfg.OBS_DIM)
            else:
                obs_flat = obs

            # FIX 2: Unpack ALL 3 values (Subject, Ally, Enemy)
            subject_emb, ally_emb, enemy_emb = self._process_critic_graph(graph_data, obs_flat)

            # Handle Batch Alignment
            num_graphs = subject_emb.shape[0]
            num_egos = obs_flat.shape[0]

            if num_graphs != num_egos and num_graphs > 0:
                agents_per_env = num_egos // num_graphs
                ally_emb = ally_emb.repeat_interleave(agents_per_env, dim=0)
                enemy_emb = enemy_emb.repeat_interleave(agents_per_env, dim=0)
                # subject_emb is already aligned by _process_critic_graph attention logic

            # Input: 128 (Subject) + 128 (Ally) + 128 (Enemy) = 384
            critic_input = torch.cat([subject_emb, ally_emb, enemy_emb], dim=1)
            value = self.critic_head(critic_input)

            if actor_features.ndim == 3:
                value = value.reshape(actor_features.shape[0], actor_features.shape[1], 1)

        return action, log_prob, entropy, value, new_gru_state

    # --- ADDED FOR PRETRAINING COMPATIBILITY ---
    def get_action_history(self, obs):
        """
        Returns list of action refinements for Deep Supervision during pretraining.
        """
        actor_features, _ = self.extract_actor_features(obs)
        # Pass return_history=True to get [y_0, y_1, y_final]
        return self.actor_head(actor_features, return_history=True)