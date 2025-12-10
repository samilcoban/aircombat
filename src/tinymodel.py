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


class RecursiveActionHead(nn.Module):
    """
    TRM (Tiny Recursive Model) Implementation for Action Refinement.

    Instead of a single forward pass, this head maintains:
    1. Context (x): The features from the Transformer/GRU.
    2. Solution (y): The current draft of the action vector.
    3. Latent (z): A hidden 'reasoning' scratchpad.

    It iterates (T) times. For T-1 steps, it runs in inference mode (no grad)
    to save memory, then runs 1 final step with gradients to learn.
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
        # initialized with small std to ensure stability at start
        self.head_action = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.head_latent = layer_init(nn.Linear(256, latent_dim), std=0.01)

        # Learnable Initial States (The "First Guess")
        # y_init corresponds to the initial action draft
        # z_init corresponds to the initial empty thought
        self.y_init = nn.Parameter(torch.zeros(1, action_dim))
        self.z_init = nn.Parameter(torch.zeros(1, latent_dim))

    def forward(self, x):
        """
        x: Context vector (Batch, D_MODEL)
        """
        batch_size = x.shape[0]

        # 1. Initialize State
        y = self.y_init.expand(batch_size, -1)
        z = self.z_init.expand(batch_size, -1)

        # 2. Recursive Loop (Thinking Phase)
        # Run T-1 times without gradient tracking to allow "deep" reasoning
        # without "deep" memory costs.
        with torch.no_grad():
            for _ in range(self.recursions - 1):
                inp = torch.cat([x, y, z], dim=-1)
                feat = self.core_net(inp)

                # Residual Update with step size 0.1 (dampening)
                y = y + 0.1 * self.head_action(feat)
                z = z + 0.1 * self.head_latent(feat)

        # 3. Final Step (Learning Phase)
        # Re-connect to computation graph here
        inp = torch.cat([x, y, z], dim=-1)
        feat = self.core_net(inp)

        # Final Update
        y_final = y + 0.1 * self.head_action(feat)

        return y_final


class HybridActorCritic(nn.Module):
    """
    Hybrid Architecture:
    - Actor: Transformer + GRU (Local Observation) + World Model (Auxiliary) + TRM Head
    - Critic: GNN (Global Graph State)
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # =================================================================
        # 1. ACTOR (Local Transformer)
        # =================================================================
        # Ego Encoder: Consumes Unified Node (Private State)
        self.ego_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL),
            nn.ReLU()
        )

        # Edge Encoder: Consumes Unified Edge (Public/Sensor State)
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

        # --- REPLACED STANDARD HEAD WITH TRM HEAD ---
        # The RecursiveActionHead iteratively refines the action based on the
        # context provided by the Transformer/GRU.
        self.actor_head = RecursiveActionHead(
            input_dim=self.cfg.D_MODEL,
            action_dim=self.cfg.ACTION_DIM,
            latent_dim=64,  # Reasoning scratchpad size
            recursions=3  # T steps
        )

        # Init bias for throttle (index 2) to start high
        # In TRM, we set the initial guess (y_init) instead of the linear bias
        with torch.no_grad():
            self.actor_head.y_init[0, 2] = 1.0

        self.actor_logstd = nn.Parameter(torch.ones(1, self.cfg.ACTION_DIM) * -0.5)

        # =================================================================
        # 2. AUXILIARY WORLD MODEL (Attached to Actor)
        # =================================================================
        # Input:  Actor Features (D_MODEL) + Action Taken (ACTION_DIM)
        # Output: Predicted Next Ego State (NODE_DIM) + Predicted Reward (1)
        self.world_model = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL + self.cfg.ACTION_DIM, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, self.cfg.NODE_DIM + 1), std=1.0)
        )

        # =================================================================
        # 3. CRITIC (Global GNN) - UNCHANGED
        # =================================================================
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

        # Critic Input: Ego(Actor) Embedding + GNN Global Context
        input_dim = self.cfg.D_MODEL + 128 + 128

        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def extract_actor_features(self, x, gru_state=None, done=None):
        """
        Runs the Actor's perception stack (Embed -> Transformer -> GRU).
        Returns the latent feature vector (gru_out) used for Policy and World Model.
        """
        # 1. Detect Input Shape
        has_seq_dim = (x.ndim == 3)

        if has_seq_dim:
            batch_size, seq_len, obs_dim = x.shape
            x_flat = x.reshape(-1, obs_dim)
        else:
            batch_size = x.shape[0]
            seq_len = 1
            x_flat = x

        # 2. Dual Projection
        ego_raw = x_flat[:, :self.cfg.NODE_DIM]
        track_raw_flat = x_flat[:, self.cfg.NODE_DIM:]

        num_tracks = self.cfg.MAX_ENTITIES - 1
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
        ego_out_flat = out[:, 0, :]  # Extract Ego Token (Batch * Seq, D_Model)

        # 4. GRU with Sequence Masking
        # We must handle internal resets if processing a sequence
        ego_gru_in = ego_out_flat.reshape(batch_size, seq_len, self.cfg.D_MODEL)

        if gru_state is None:
            gru_state = torch.zeros(1, batch_size, self.cfg.D_MODEL, device=x.device)

        if has_seq_dim and done is not None:
            # Training Mode with Sequence + Resets
            # done shape: (Batch, SeqLen). Indicates if step 't' is a fresh start (reset).
            outputs = []
            for t in range(seq_len):
                # Apply mask to hidden state BEFORE processing step t
                # done[:, t] == 1 means step t is a new episode, so prev state should be ignored.
                current_mask = 1.0 - done[:, t].view(1, -1, 1)
                gru_state = gru_state * current_mask

                # GRU Step (Input: Batch x 1 x D)
                step_input = ego_gru_in[:, t:t + 1, :]
                step_out, gru_state = self.actor_gru(step_input, gru_state)
                outputs.append(step_out)

            # Concatenate back to (Batch, Seq, D)
            gru_out = torch.cat(outputs, dim=1)
            new_gru_state = gru_state

        else:
            # Inference Mode or No Resets
            # Apply mask once at the start
            if done is not None and not has_seq_dim:
                mask = 1.0 - done.view(1, -1, 1)
                gru_state = gru_state * mask

            # Fast C-level processing for the whole chunk/step
            gru_out, new_gru_state = self.actor_gru(ego_gru_in, gru_state)

        if not has_seq_dim:
            gru_out = gru_out.squeeze(1)

        return gru_out, new_gru_state

    def get_aux_prediction(self, actor_features, action):
        """
        Forward pass for the World Model.
        Args:
            actor_features: The output of the Actor GRU (Batch, D_MODEL)
            action: The action taken (Batch, ACTION_DIM)
        Returns:
            pred_next_state: (Batch, NODE_DIM)
            pred_reward: (Batch, 1)
        """
        # Concatenate Features + Action
        # Flatten sequence dims if present
        if actor_features.ndim == 3:
            actor_features = actor_features.reshape(-1, self.cfg.D_MODEL)
        if action.ndim == 3:
            action = action.reshape(-1, self.cfg.ACTION_DIM)

        inp = torch.cat([actor_features, action], dim=-1)

        preds = self.world_model(inp)

        pred_next_state = preds[:, :self.cfg.NODE_DIM]
        pred_reward = preds[:, -1]

        return pred_next_state, pred_reward

    def _process_critic_graph(self, graph_data):
        """Processes the Global Graph for the Critic."""
        x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch
        x = torch.relu(self.gnn_conv1(x, edge_index, edge_attr))
        x = torch.relu(self.gnn_conv2(x, edge_index, edge_attr))

        # Check Team (Index 1) -> 1.0 is Blue
        is_blue = x[:, 1]
        ally_mask = (is_blue > 0.5)
        enemy_mask = (is_blue <= 0.5)

        # Pooling
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1

        ally_emb = global_mean_pool(x[ally_mask], batch[ally_mask], size=num_graphs)
        if ally_mask.sum() == 0: ally_emb = torch.zeros(num_graphs, 128, device=x.device)

        enemy_emb = global_mean_pool(x[enemy_mask], batch[enemy_mask], size=num_graphs)
        if enemy_mask.sum() == 0: enemy_emb = torch.zeros(num_graphs, 128, device=x.device)

        return ally_emb, enemy_emb

    def get_value(self, graph_batch, obs, gru_state=None, done=None):
        ally_emb, enemy_emb = self._process_critic_graph(graph_batch)

        # Actor Features needed for Critic (Ego state)
        # Note: We re-run the actor extractor here. In optimized training,
        # you might pass features in, but this keeps the API clean.
        actor_features, _ = self.extract_actor_features(obs, gru_state, done)

        if actor_features.ndim == 3:
            actor_features = actor_features.reshape(-1, self.cfg.D_MODEL)

        num_graphs = ally_emb.shape[0]
        num_egos = actor_features.shape[0]

        if num_graphs != num_egos and num_graphs > 0:
            agents_per_env = num_egos // num_graphs
            ally_emb = ally_emb.repeat_interleave(agents_per_env, dim=0)
            enemy_emb = enemy_emb.repeat_interleave(agents_per_env, dim=0)

        critic_input = torch.cat([actor_features, ally_emb, enemy_emb], dim=1)
        return self.critic_head(critic_input)

    def get_action_and_value(self, obs, graph_data=None, action=None, gru_state=None, done=None):
        # 1. Actor Pipeline
        actor_features, new_gru_state = self.extract_actor_features(obs, gru_state, done)

        # 2. Action Head (Recursive TRM)
        # This iteratively refines the action 'y' and latent 'z' using the context 'x'
        action_mean = self.actor_head(actor_features)

        # Clamp logstd to prevent underflow (std=0) or explosion (std=NaN/Inf)
        logstd_clamped = torch.clamp(self.actor_logstd, -10.0, 2.0)
        action_std = torch.exp(logstd_clamped).expand_as(action_mean)

        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            action = torch.clamp(action, -1.0, 1.0)

        # Sum log probs over action dimension
        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)

        # 3. Critic Pipeline (GNN)
        value = None
        if graph_data is not None:
            critic_ego = actor_features.reshape(-1, self.cfg.D_MODEL) if actor_features.ndim == 3 else actor_features
            ally_emb, enemy_emb = self._process_critic_graph(graph_data)

            num_graphs = ally_emb.shape[0]
            num_egos = critic_ego.shape[0]

            if num_graphs != num_egos and num_graphs > 0:
                agents_per_env = num_egos // num_graphs
                ally_emb = ally_emb.repeat_interleave(agents_per_env, dim=0)
                enemy_emb = enemy_emb.repeat_interleave(agents_per_env, dim=0)

            critic_input = torch.cat([critic_ego, ally_emb, enemy_emb], dim=1)
            value = self.critic_head(critic_input)

            if actor_features.ndim == 3:
                value = value.reshape(actor_features.shape[0], actor_features.shape[1], 1)

        return action, log_prob, entropy, value, new_gru_state