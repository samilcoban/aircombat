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
    """
    Standard orthogonal initialization for RL.
    Helps prevents vanishing/exploding gradients at start of training.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HybridActorCritic(nn.Module):
    """
    Hybrid Actor-Critic Model for Air Combat (Dual Projection Architecture).

    Architecture:
    - Actor: Dual-Encoder -> Transformer -> GRU -> Action Head
      * Ego Encoder: Processes Cockpit state (Fuel, Alt, Speed).
      * Edge Encoder: Processes Radar Tracks (Relative Range, Azimuth, Closure).
      * Transformer: Cross-correlates Ego with all Radar Tracks.

    - Critic: GNN -> Pooling -> Fusion -> Value Head
      * GNN: Processes the global topology (who is aiming at whom).
      * Semantic Pooling: Separately aggregates Ally and Enemy contexts.
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # ==========================================
        # 1. ACTOR: DUAL PROJECTION ENCODERS
        # ==========================================
        # Encoder for Ego (Cockpit) features
        # Input: FEAT_DIM_EGO (e.g., 18) -> Output: D_MODEL (e.g., 128)
        self.ego_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.FEAT_DIM_EGO, self.cfg.D_MODEL)),
            nn.ReLU()
        )

        # Encoder for Edge (Track) features
        # Input: FEAT_DIM_EDGE (e.g., 14) -> Output: D_MODEL
        self.edge_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.FEAT_DIM_EDGE, self.cfg.D_MODEL)),
            nn.ReLU()
        )

        # Transformer Backbone
        # Processes the sequence [Ego_Emb, Track_1, Track_2, ...]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL,
            nhead=self.cfg.N_HEADS,
            dim_feedforward=512,
            batch_first=True,
            norm_first=True
        )
        self.actor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.N_LAYERS)

        # Temporal Memory (GRU)
        # Allows the agent to estimate derivatives (turn rate, acceleration)
        # that aren't explicitly in the observation.
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

        # Bias throttle high (start with engines running)
        with torch.no_grad():
            self.actor_head[-1].bias[2].fill_(1.0)

        # Learnable Log-Std for continuous action exploration
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.cfg.ACTION_DIM))

        # ==========================================
        # 3. CRITIC (GNN + FUSION)
        # ==========================================
        # Node features (12) + Edge features (6)
        # Note: GNN uses the global graph state, not the relative actor state.
        self.gnn_conv1 = EdgeGCNConv(node_channels=12, edge_channels=6, out_channels=128)
        self.gnn_conv2 = EdgeGCNConv(node_channels=128, edge_channels=6, out_channels=128)

        # Critic Input Fusion: Ego (128) + Ally Pool (128) + Enemy Pool (128)
        input_dim = self.cfg.D_MODEL + 128 + 128

        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def _extract_ego_features(self, x, gru_state=None, done=None):
        """
        Runs the Actor's Dual Projection -> Transformer -> GRU pipeline.

        Args:
            x: Flat observation vector [Batch, OBS_DIM]
            gru_state: Previous hidden state [1, Batch, D_MODEL]
            done: Reset flags for GRU [Batch]
        """
        batch_size = x.shape[0]

        # 1. SLICE THE INPUT (Dual Projection)
        # ------------------------------------
        # The input x is flat. We slice it into the Ego part and the Tracks part.

        # Ego part is at the beginning
        ego_raw = x[:, :self.cfg.FEAT_DIM_EGO]  # Shape: (Batch, Ego_Dim)

        # Track part is the rest
        track_raw_flat = x[:, self.cfg.FEAT_DIM_EGO:]

        # Reshape flat tracks to (Batch, N_Tracks, Track_Dim)
        # N_Tracks = MAX_ENTITIES - 1 (Ego is removed)
        num_tracks = self.cfg.MAX_ENTITIES - 1
        track_raw = track_raw_flat.view(batch_size, num_tracks, self.cfg.FEAT_DIM_EDGE)

        # 2. PROJECT TO EMBEDDING SPACE
        # -----------------------------
        # Project Ego: (Batch, Ego_Dim) -> (Batch, 1, D_Model)
        ego_emb = self.ego_encoder(ego_raw).unsqueeze(1)

        # Project Tracks: (Batch, N, Edge_Dim) -> (Batch, N, D_Model)
        track_emb = self.edge_encoder(track_raw)

        # 3. CONSTRUCT SEQUENCE & MASKING
        # -------------------------------
        # Sequence: [Ego, Track_1, Track_2, ..., Track_N]
        transformer_input = torch.cat([ego_emb, track_emb], dim=1)

        # Masking Logic:
        # We must mask padding tracks (zeros).
        # We calculate norm of track vector. If ~0, it's padding.
        track_norms = torch.norm(track_raw, dim=2)
        track_mask = (track_norms < 1e-5)  # True = Padding/Ignore

        # FIX (Priority 2): Ego Mask is ALWAYS False (Visible).
        # Even if Ego is dead (inputs are 0), we want the GRU to process the
        # "Dead Bias Vector" rather than random noise.
        ego_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)

        # Combine masks: (Batch, 1 + N)
        full_mask = torch.cat([ego_mask, track_mask], dim=1)

        # 4. TRANSFORMER PASS
        # -------------------
        out = self.actor_transformer(transformer_input, src_key_padding_mask=full_mask)

        # 5. EXTRACT OUTPUT (Index 0 = Ego)
        # ---------------------------------
        # The first token has now attended to all relevant radar tracks via Self-Attention
        ego_out = out[:, 0, :]  # Shape: (Batch, D_Model)

        # 6. GRU UPDATE
        # -------------
        if gru_state is None:
            gru_state = torch.zeros(1, batch_size, self.cfg.D_MODEL, device=x.device)

        if done is not None:
            # Reset GRU state for agents that finished an episode
            # Reshape done for broadcasting: (1, Batch, 1)
            gru_state = gru_state * (1.0 - done).view(1, -1, 1)

        # GRU expects input (Batch, D_Model) -> unsqueeze not needed for batch_first=True
        # provided input_size matches. Wait, nn.GRU input is (Batch, Seq, Feature) if batch_first.
        # We are processing one step, so Seq=1.
        ego_gru_in = ego_out.unsqueeze(1)  # (Batch, 1, D_Model)

        gru_out, new_gru_state = self.actor_gru(ego_gru_in, gru_state)

        # Return the processed vector (Batch, D_Model) and the new state
        return gru_out.squeeze(1), new_gru_state

    def _process_critic_graph(self, graph_data):
        """
        Runs the GNN Critic layers.
        Handles Semantic Masking (Pooling Allies vs Enemies separately).
        """
        x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch

        # GNN Layers
        x = torch.relu(self.gnn_conv1(x, edge_index, edge_attr))
        x = torch.relu(self.gnn_conv2(x, edge_index, edge_attr))

        # Semantic Masking
        # Feature 10 is 'is_blue' (1.0 for Blue, 0.0 for Red)
        # Note: This assumes standard core features in GNN, not the new Actor features.
        is_blue = x[:, 10]

        # Ally Mask (Blue Team)
        ally_mask = (is_blue > 0.5)
        # Enemy Mask (Red Team)
        enemy_mask = (is_blue <= 0.5)

        # Global Pooling
        # If a graph has no enemies (training mode), pool returns 0 vector.
        ally_emb = global_mean_pool(x[ally_mask], batch[ally_mask], size=batch.max().item() + 1)
        enemy_emb = global_mean_pool(x[enemy_mask], batch[enemy_mask], size=batch.max().item() + 1)

        return ally_emb, enemy_emb

    def get_value(self, graph_batch, obs, gru_state=None, done=None):
        """
        Returns Value V(s) for the current state.
        Combines Ego Context (Actor) with Global Tactical Context (Critic).
        """
        # 1. Get Global Context from GNN
        ally_emb, enemy_emb = self._process_critic_graph(graph_batch)

        # 2. Expand Global Embeddings to match Agents
        # Graph batch contains N environments. Obs contains N * Agents.
        num_envs = ally_emb.shape[0]
        total_agents = obs.shape[0]
        agents_per_env = total_agents // num_envs

        ally_expanded = ally_emb.repeat_interleave(agents_per_env, dim=0)
        enemy_expanded = enemy_emb.repeat_interleave(agents_per_env, dim=0)

        # 3. Get Ego Context from Actor
        # We run the partial actor forward pass to get the Ego embedding
        ego_emb, _ = self._extract_ego_features(obs, gru_state, done)

        # 4. Fusion
        critic_input = torch.cat([ego_emb, ally_expanded, enemy_expanded], dim=1)

        return self.critic_head(critic_input)

    def get_action_and_value(self, obs, graph_data=None, action=None, gru_state=None, done=None):
        """
        Main Forward Pass.
        Returns: Action, LogProb, Entropy, Value, New_GRU_State
        """
        # 1. Actor Forward Pass
        ego_emb, new_gru_state = self._extract_ego_features(obs, gru_state, done)

        # 2. Action Heads
        action_mean = self.actor_head(ego_emb)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        # 3. Critic Forward Pass (Optional)
        value = None
        if graph_data is not None:
            ally_emb, enemy_emb = self._process_critic_graph(graph_data)

            num_envs = ally_emb.shape[0]
            total_agents = obs.shape[0]
            agents_per_env = total_agents // num_envs

            ally_expanded = ally_emb.repeat_interleave(agents_per_env, dim=0)
            enemy_expanded = enemy_emb.repeat_interleave(agents_per_env, dim=0)

            critic_input = torch.cat([ego_emb, ally_expanded, enemy_expanded], dim=1)
            value = self.critic_head(critic_input)

        return action, log_prob, entropy, value, new_gru_state