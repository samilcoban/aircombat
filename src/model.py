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


class HybridActorCritic(nn.Module):
    """
    Hybrid Architecture:
    - Actor: Transformer + GRU (Object Permanence) + World Model (Auxiliary)
    - Critic: GNN + Attention (Self-State Distinction)
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # =================================================================
        # 1. ACTOR (Local Transformer + GRU)
        # =================================================================
        self.ego_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL), nn.ReLU(), nn.Dropout(self.cfg.DROPOUT))

        self.edge_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.EDGE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL), nn.ReLU(), nn.Dropout(self.cfg.DROPOUT))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL, nhead=self.cfg.N_HEADS,
            dim_feedforward=self.cfg.D_MODEL * 2, dropout=self.cfg.DROPOUT,
            batch_first=True, norm_first=True
        )
        self.actor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.N_LAYERS)

        # GRU for Memory (Restored)
        self.actor_gru = nn.GRU(
            self.cfg.D_MODEL, self.cfg.D_MODEL, batch_first=True
        )

        # Standard Linear Action Head
        self.actor_mean = layer_init(nn.Linear(self.cfg.D_MODEL, self.cfg.ACTION_DIM), std=0.01)

        # Init throttle bias (index 2) to 1.0 (Full Afterburner)
        with torch.no_grad():
            self.actor_mean.bias[2] = 1.0

        self.actor_logstd = nn.Parameter(torch.ones(1, self.cfg.ACTION_DIM) * -0.5)

        # =================================================================
        # 2. AUXILIARY WORLD MODEL (Regularizer)
        # =================================================================
        self.world_model = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL + self.cfg.ACTION_DIM, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, self.cfg.NODE_DIM + 1), std=1.0)
        )

        # =================================================================
        # 3. CRITIC (GNN + Attention)
        # =================================================================
        # Shared Physics Encoder Maps NODE_DIM (20) -> 128
        self.shared_physics_encoder = nn.Sequential(layer_init(nn.Linear(self.cfg.NODE_DIM, 128)), nn.ReLU())

        self.gnn_conv1 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)
        self.gnn_conv2 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)

        self.attention_scale = 1.0 / np.sqrt(128)

        # Input: Attention_Emb(128) + Ally_Context(128) + Enemy_Context(128) = 384
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(128 + 128 + 128, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def extract_actor_features(self, x, gru_state=None, done=None):
        """
        Processes observation via Transformer -> GRU.
        Returns: (features, new_gru_state)
        """
        # 1. Dimensions
        has_seq_dim = (x.ndim == 3)
        if has_seq_dim:
            batch, seq, dim = x.shape
            x_flat = x.reshape(-1, dim)
        else:
            batch, seq = x.shape[0], 1
            x_flat = x

        # 2. Embed
        ego_raw = x_flat[:, :self.cfg.NODE_DIM]
        track_raw = x_flat[:, self.cfg.NODE_DIM:].reshape(batch * seq, -1, self.cfg.EDGE_DIM)

        ego_emb = self.ego_encoder(ego_raw).unsqueeze(1)
        track_emb = self.edge_encoder(track_raw)

        # 3. Transformer
        mask = torch.cat([torch.zeros(batch * seq, 1, dtype=torch.bool, device=x.device),
                          (track_raw[:, :, 0] < 1e-5)], dim=1)

        transformer_out = self.actor_transformer(
            torch.cat([ego_emb, track_emb], dim=1),
            src_key_padding_mask=mask
        )

        # Extract Ego Token (Index 0)
        ego_features = transformer_out[:, 0, :]  # (Batch*Seq, D_MODEL)

        # 4. GRU
        if gru_state is None:
            # Initialize to 0. NOTE: 'batch' here refers to the number of independent agents/envs
            gru_state = torch.zeros(1, batch, self.cfg.D_MODEL, device=x.device)

        if has_seq_dim:
            # Reshape for GRU: (Batch, Seq, D_MODEL)
            gru_in = ego_features.reshape(batch, seq, self.cfg.D_MODEL)

            # Manual loop to handle resets in sequence
            outputs = []
            if done is not None:
                current_h = gru_state
                for t in range(seq):
                    # If done=1, reset hidden state to 0
                    mask_t = 1.0 - done[:, t].view(1, -1, 1)
                    current_h = current_h * mask_t

                    out_t, current_h = self.actor_gru(gru_in[:, t:t + 1, :], current_h)
                    outputs.append(out_t)

                gru_out = torch.cat(outputs, dim=1)
                new_gru_state = current_h
            else:
                # Optimized call
                gru_out, new_gru_state = self.actor_gru(gru_in, gru_state)

            gru_out = gru_out.reshape(batch * seq, self.cfg.D_MODEL)
        else:
            # Inference step (Batch, 1, D_MODEL)
            gru_in = ego_features.unsqueeze(1)

            # Masking for single step
            if done is not None:
                mask = 1.0 - done.view(1, -1, 1)
                gru_state = gru_state * mask

            gru_out, new_gru_state = self.actor_gru(gru_in, gru_state)
            gru_out = gru_out.squeeze(1)

        return gru_out, new_gru_state

    def get_aux_prediction(self, actor_features, action):
        if actor_features.ndim == 3: actor_features = actor_features.reshape(-1, self.cfg.D_MODEL)
        if action.ndim == 3: action = action.reshape(-1, self.cfg.ACTION_DIM)
        preds = self.world_model(torch.cat([actor_features, action], dim=-1))
        return preds[:, :self.cfg.NODE_DIM], preds[:, -1]

    def _process_critic_graph(self, graph_data, obs_flat):
        """
        Attention-based Graph Processing.
        Query: Ego Physics embedding.
        Key/Value: GNN Node embeddings.
        """
        x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch

        # 1. GNN Pass
        node_embeddings = self.shared_physics_encoder(x)  # (TotalNodes, 128)
        gnn_x = torch.relu(self.gnn_conv1(node_embeddings, edge_index, edge_attr))
        gnn_out = torch.relu(self.gnn_conv2(gnn_x, edge_index, edge_attr))

        if batch is None: batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        num_graphs = batch.max().item() + 1

        # 2. Context Aggregation (Teams)
        is_blue = x[:, 1]
        ally_mask = (is_blue > 0.5)
        enemy_mask = (is_blue <= 0.5)

        ally_context = global_mean_pool(gnn_out[ally_mask], batch[ally_mask], size=num_graphs)
        enemy_context = global_mean_pool(gnn_out[enemy_mask], batch[enemy_mask], size=num_graphs)

        # Handle empty sets
        if ally_mask.sum() == 0: ally_context = torch.zeros(num_graphs, 128, device=x.device)
        if enemy_mask.sum() == 0: enemy_context = torch.zeros(num_graphs, 128, device=x.device)

        # 3. ATTENTION MECHANISM (Restored)
        # Ego State is the Query
        ego_raw = obs_flat[:, :self.cfg.NODE_DIM]
        query = self.shared_physics_encoder(ego_raw)  # (Batch, 128)

        # Prepare Keys/Values (Dense Batching)
        dense_keys, mask = to_dense_batch(node_embeddings, batch)
        dense_vals, _ = to_dense_batch(gnn_out, batch)

        # Handle Batch Misalignment (PPO batch vs Graph batch)
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

        # Weighted Sum of Values
        subject_emb = torch.bmm(weights, vals).squeeze(1)

        return subject_emb, ally_context, enemy_context

    def get_value(self, graph_batch, obs, gru_state=None, done=None):
        if obs.ndim == 3:
            obs = obs.reshape(-1, self.cfg.OBS_DIM)

        subject_emb, ally, enemy = self._process_critic_graph(graph_batch, obs)

        critic_input = torch.cat([subject_emb, ally, enemy], dim=1)
        value = self.critic_head(critic_input)

        # Reshape value back to sequence if needed
        # PPO usually expects flat value, but keeping for compatibility
        return value

    def get_action_and_value(self, obs, graph_data=None, action=None, gru_state=None, done=None):
        # 1. Actor Features (Transformer + GRU)
        # Returns: (Batch*Seq, D_MODEL), New_GRU
        actor_features, new_gru_state = self.extract_actor_features(obs, gru_state, done)

        # 2. Action Head
        action_mean = self.actor_mean(actor_features)

        logstd_clamped = torch.clamp(self.actor_logstd, -2.0, 2.0)
        action_std = torch.exp(logstd_clamped).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            action = torch.clamp(action, -1.0, 1.0)
        else:
            # <--- FIX IS HERE --->
            # Flatten action if it comes in as 3D (Batch, Seq, Dim)
            # because 'probs' is flattened (Batch*Seq, Dim)
            if action.ndim == 3:
                action = action.reshape(-1, self.cfg.ACTION_DIM)
            # <--- END FIX --->

        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)

        # 3. Critic (Optional)
        value = None
        if graph_data is not None:
            if obs.ndim == 3:
                obs_flat = obs.reshape(-1, self.cfg.OBS_DIM)
            else:
                obs_flat = obs

            subject_emb, ally, enemy = self._process_critic_graph(graph_data, obs_flat)
            critic_input = torch.cat([subject_emb, ally, enemy], dim=1)
            value = self.critic_head(critic_input)

            # Reshape value back to sequence if needed
            if obs.ndim == 3:
                value = value.view(obs.shape[0], obs.shape[1], 1)

        return action, log_prob, entropy, value, new_gru_state

    def get_action_history(self, obs):
        """
        Compatibility for pretraining.
        Returns single-step action in a list.
        Uses a zero-state GRU approximation for stateless supervised training.
        """
        features, _ = self.extract_actor_features(obs)
        action = self.actor_mean(features)
        return [action]