# ================================================
# FILE: src/model.py
# ================================================
"""
Neural network architecture for air combat agents.

This module defines the core neural network models:
1. HybridActorCritic: Main policy network combining:
   - Actor: Transformer + GRU + Recursive Action Head (TRM)
   - Critic: Graph Neural Network (GNN) with attention

2. AirCombatDiscriminator: GAIL discriminator for imitation learning

3. RecursiveActionHead: Tiny Recursive Model (TRM) for action refinement

Architecture Overview:
- Actor processes local observations through a Transformer encoder,
  uses GRU for temporal memory, and outputs actions via TRM.
- Critic processes the full entity graph via GNN with team-aware
  pooling and attention mechanisms.
"""
import torch
import torch.nn as nn
import numpy as np
from config import Config
from src.gnn_layers import EdgeGCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a linear layer with orthogonal weights.
    
    Orthogonal initialization helps with training stability,
    especially for deep networks and recurrent architectures.
    
    Args:
        layer: nn.Linear layer to initialize.
        std: Standard deviation for weight scaling.
        bias_const: Constant value for bias initialization.
        
    Returns:
        Initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AirCombatDiscriminator(nn.Module):
    """
    GAIL Discriminator D(s, a).
    
    Distinguishes between Expert (1) and Agent (0) behavior.
    Used to provide reward shaping during training by rewarding
    expert-like state-action pairs.
    
    Architecture:
    1. Shared physics encoder for node features
    2. Two-layer GNN for graph processing
    3. Team-aware pooling (ally vs enemy)
    4. Attention mechanism for agent-specific context
    5. Action encoder
    6. Fusion head for final classification
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # 1. Shared Physics Encoder - maps raw node features to embeddings.
        self.shared_physics_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, 128)),
            nn.ReLU()
        )

        # 2. GNN Layers - process graph structure with edge features.
        self.gnn_conv1 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)
        self.gnn_conv2 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)

        # 3. Action Encoder - embed action vector.
        self.action_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.ACTION_DIM, 64)),
            nn.Tanh()
        )

        # 4. Classification Head - fuses all features for discrimination.
        # Input: subject_emb + ally_context + enemy_context + action_emb
        input_dim = 128 + 128 + 128 + 64
        self.head = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)),
            nn.ReLU(),
            nn.Dropout(0.1),
            layer_init(nn.Linear(128, 1))
        )
        
        # Attention scaling factor for numerical stability.
        self.attention_scale = 1.0 / np.sqrt(128)

    def forward(self, graph_batch, obs, action):
        """
        Forward pass through discriminator.
        
        Args:
            graph_batch: PyG Batch of graphs.
            obs: Observations [batch, obs_dim] or [batch, seq, obs_dim].
            action: Actions [batch, action_dim] or [batch, seq, action_dim].
            
        Returns:
            Logits [batch, 1] - positive means expert-like.
        """
        # A. Process Graph
        raw_nodes = graph_batch.x
        node_embeddings = self.shared_physics_encoder(raw_nodes)
        x = torch.relu(self.gnn_conv1(node_embeddings, graph_batch.edge_index, graph_batch.edge_attr))
        gnn_out = torch.relu(self.gnn_conv2(x, graph_batch.edge_index, graph_batch.edge_attr))

        # Get batch assignment for graph pooling.
        batch_idx = graph_batch.batch if graph_batch.batch is not None else torch.zeros(raw_nodes.size(0),
                                                                                        dtype=torch.long,
                                                                                        device=raw_nodes.device)
        num_graphs = batch_idx.max().item() + 1

        # Team-aware pooling: separate ally and enemy embeddings.
        is_ally = (raw_nodes[:, 1] > 0.5)    # Team flag > 0.5 means ally.
        is_enemy = (raw_nodes[:, 1] <= 0.5)

        ally_context = global_mean_pool(gnn_out[is_ally], batch_idx[is_ally], size=num_graphs)
        enemy_context = global_mean_pool(gnn_out[is_enemy], batch_idx[is_enemy], size=num_graphs)
        
        # Handle empty cases.
        if is_ally.sum() == 0: ally_context = torch.zeros(num_graphs, 128, device=obs.device)
        if is_enemy.sum() == 0: enemy_context = torch.zeros(num_graphs, 128, device=obs.device)

        # B. Attention mechanism for subject-specific embedding.
        if obs.ndim == 3: obs = obs.reshape(-1, self.cfg.OBS_DIM)
        ego_raw = obs[:, :self.cfg.NODE_DIM]
        query = self.shared_physics_encoder(ego_raw)

        # Convert to dense batch for attention.
        dense_keys, mask = to_dense_batch(node_embeddings, batch_idx)
        dense_vals, _ = to_dense_batch(gnn_out, batch_idx)

        # Broadcast keys/values if multiple agents per graph.
        if query.shape[0] >= num_graphs and num_graphs > 0:
            agents_per_env = query.shape[0] // num_graphs
            keys = dense_keys.repeat_interleave(agents_per_env, dim=0)
            vals = dense_vals.repeat_interleave(agents_per_env, dim=0)
            mask_expanded = mask.repeat_interleave(agents_per_env, dim=0)
            ally_context = ally_context.repeat_interleave(agents_per_env, dim=0)
            enemy_context = enemy_context.repeat_interleave(agents_per_env, dim=0)
        else:
            keys, vals, mask_expanded = dense_keys, dense_vals, mask

        # Compute attention: Q * K^T -> softmax -> weighted sum of V.
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * self.attention_scale
        scores = scores.masked_fill(~mask_expanded.unsqueeze(1), -1e4)
        weights = F.softmax(scores, dim=-1)
        subject_emb = torch.bmm(weights, vals).squeeze(1)

        # C. Fusion and classification.
        if action.ndim == 3: action = action.reshape(-1, self.cfg.ACTION_DIM)
        action_emb = self.action_encoder(action)
        fusion = torch.cat([subject_emb, ally_context, enemy_context, action_emb], dim=1)
        return self.head(fusion)


class RecursiveActionHead(nn.Module):
    """
    TRM (Tiny Recursive Model) Implementation for Action Refinement.
    
    This module implements iterative action refinement where actions
    are progressively improved through multiple "thinking" steps.
    
    Key concepts:
    - y: Action output, refined iteratively
    - z: Latent state, maintains internal memory across iterations
    - Residual updates: y_{t+1} = y_t + 0.1 * delta_y
    
    During training with return_history=True, all intermediate
    actions are returned for deep supervision.
    During inference, only the final action is returned.
    """

    def __init__(self, input_dim, action_dim, latent_dim=64, recursions=3):
        """
        Args:
            input_dim: Dimension of input context vector.
            action_dim: Dimension of action output.
            latent_dim: Dimension of latent state.
            recursions: Number of refinement iterations.
        """
        super().__init__()
        self.recursions = recursions
        self.action_dim = action_dim

        # Core Network: processes (Context + Action + Latent) -> Hidden.
        input_total = input_dim + action_dim + latent_dim
        self.core_net = nn.Sequential(
            layer_init(nn.Linear(input_total, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU()
        )

        # Update Heads: Hidden -> Delta Y (Action) and Delta Z (Latent).
        self.head_action = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.head_latent = layer_init(nn.Linear(256, latent_dim), std=0.01)

        # Learnable Initial States (shared across batch).
        self.y_init = nn.Parameter(torch.zeros(1, action_dim))
        self.z_init = nn.Parameter(torch.zeros(1, latent_dim))

    def forward(self, x, return_history=False):
        """
        Forward pass with iterative refinement.
        
        Args:
            x: Context vector (Batch, D_MODEL) or (Batch, Seq_Len, D_MODEL).
            return_history: If True, returns list of all 'y' steps for deep supervision.
            
        Returns:
            If return_history: List of action tensors [y_1, y_2, ..., y_n].
            Otherwise: Final action tensor y_n.
        """
        # Initialize action and latent state.
        if x.ndim == 3:
            B, T, _ = x.shape
            y = self.y_init.view(1, 1, -1).expand(B, T, -1)
            z = self.z_init.view(1, 1, -1).expand(B, T, -1)
        else:
            B, _ = x.shape
            y = self.y_init.expand(B, -1)
            z = self.z_init.expand(B, -1)

        history = []

        # Determine gradient tracking strategy.
        # For training: track gradients for all steps.
        # For inference: only track final step (faster).
        steps_with_grad = self.recursions if return_history else 1
        steps_no_grad = 0 if return_history else (self.recursions - 1)

        # A. Thinking Phase (Inference Optimization) - no gradients.
        if steps_no_grad > 0:
            with torch.no_grad():
                for _ in range(steps_no_grad):
                    inp = torch.cat([x, y, z], dim=-1)
                    feat = self.core_net(inp)
                    y = y + 0.1 * self.head_action(feat)  # Residual update.
                    z = z + 0.1 * self.head_latent(feat)

        # B. Learning Phase (Gradient Tracking).
        for _ in range(steps_with_grad):
            inp = torch.cat([x, y, z], dim=-1)
            feat = self.core_net(inp)

            y = y + 0.1 * self.head_action(feat)  # Residual update to action.
            z = z + 0.1 * self.head_latent(feat)  # Residual update to latent.

            if return_history:
                history.append(y)

        if return_history:
            return history

        return y


class HybridActorCritic(nn.Module):
    """
    Hybrid Architecture for Air Combat RL.
    
    Actor Pipeline (Local Processing):
    1. Ego/Edge Encoders: Embed raw observations
    2. Transformer: Process entity relationships with self-attention
    3. GRU: Temporal memory for sequential decision making
    4. TRM Head: Recursive action refinement
    
    Critic Pipeline (Global Processing):
    1. GNN: Process full entity graph with edge features
    2. Team-Aware Pooling: Separate ally/enemy contexts
    3. Attention: Agent-specific state extraction
    4. Value Head: Estimate expected returns
    
    Auxiliary:
    - World Model: Predicts next state for representation learning
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # =================================================================
        # 1. ACTOR (Local Transformer)
        # =================================================================
        
        # Ego encoder: process own aircraft state.
        self.ego_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.NODE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL),
            nn.ReLU()
        )

        # Edge encoder: process track/relationship features.
        self.edge_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.EDGE_DIM, self.cfg.D_MODEL)),
            nn.LayerNorm(self.cfg.D_MODEL),
            nn.ReLU()
        )

        # Transformer encoder for processing entity relationships.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL,
            nhead=self.cfg.N_HEADS,
            dim_feedforward=self.cfg.D_MODEL * 4,
            batch_first=True,
            norm_first=True  # Pre-norm architecture for training stability.
        )
        self.actor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.N_LAYERS)

        # GRU for temporal memory.
        self.actor_gru = nn.GRU(
            input_size=self.cfg.D_MODEL,
            hidden_size=self.cfg.D_MODEL,
            batch_first=True
        )

        # TRM Head for recursive action refinement.
        self.actor_head = RecursiveActionHead(
            input_dim=self.cfg.D_MODEL,
            action_dim=self.cfg.ACTION_DIM,
            latent_dim=64,
            recursions=3
        )

        # Initialize throttle bias to 1.0 (default full throttle).
        with torch.no_grad():
            self.actor_head.y_init[0, 2] = 1.0

        # Learnable log standard deviation for action distribution.
        # Initialized to -0.5 which gives std ~0.6 for exploration.
        self.actor_logstd = nn.Parameter(torch.ones(1, self.cfg.ACTION_DIM) * -0.5)

        # =================================================================
        # 2. AUXILIARY WORLD MODEL
        # =================================================================
        # Predicts next state and reward for representation learning.
        self.world_model = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL + self.cfg.ACTION_DIM, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, self.cfg.NODE_DIM + 1), std=1.0)  # +1 for reward.
        )

        # =================================================================
        # 3. CRITIC (Global GNN)
        # =================================================================
        
        # Shared Physics Encoder: Maps NODE_DIM (20) -> 128.
        self.shared_physics_encoder = nn.Sequential(layer_init(nn.Linear(self.cfg.NODE_DIM, 128)), nn.ReLU())

        # Two-layer GNN for graph processing.
        self.gnn_conv1 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)
        self.gnn_conv2 = EdgeGCNConv(128, self.cfg.EDGE_DIM, 128)

        # Attention scaling factor.
        self.attention_scale = 1.0 / np.sqrt(128)

        # Value head: fuses subject + ally + enemy embeddings.
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(128 + 128 + 128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def extract_actor_features(self, x, gru_state=None, done=None):
        """
        Actor perception stack.
        
        Processes observations through encoder, transformer, and GRU
        to produce context features for action generation.
        
        Args:
            x: Observations [batch, obs_dim] or [batch, seq_len, obs_dim].
            gru_state: Hidden state for GRU [1, batch, d_model].
            done: Done flags for resetting GRU state.
            
        Returns:
            Tuple of (features, new_gru_state).
        """
        has_seq_dim = (x.ndim == 3)

        # Handle both 2D and 3D inputs.
        if has_seq_dim:
            batch_size, seq_len, obs_dim = x.shape
            x_flat = x.reshape(-1, obs_dim)
        else:
            batch_size = x.shape[0]
            seq_len = 1
            x_flat = x

        # Split observation into ego state and track data.
        ego_raw = x_flat[:, :self.cfg.NODE_DIM]
        track_raw_flat = x_flat[:, self.cfg.NODE_DIM:]

        num_tracks = self.cfg.MAX_ENTITIES - 1
        track_raw = track_raw_flat.reshape(batch_size * seq_len, num_tracks, self.cfg.EDGE_DIM)

        # Encode ego and tracks.
        ego_emb = self.ego_encoder(ego_raw).unsqueeze(1)
        track_emb = self.edge_encoder(track_raw)

        # Concatenate for transformer: [ego, track_1, track_2, ...].
        transformer_input = torch.cat([ego_emb, track_emb], dim=1)

        # Create attention mask for invalid tracks (distance = 0).
        track_dists = track_raw[:, :, 0]
        track_mask = (track_dists < 1e-5)  # Invalid tracks.
        ego_mask = torch.zeros(batch_size * seq_len, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([ego_mask, track_mask], dim=1)

        # Apply transformer.
        out = self.actor_transformer(transformer_input, src_key_padding_mask=full_mask)
        ego_out_flat = out[:, 0, :]  # Take ego token output.

        # GRU processing.
        ego_gru_in = ego_out_flat.reshape(batch_size, seq_len, self.cfg.D_MODEL)

        if gru_state is None:
            gru_state = torch.zeros(1, batch_size, self.cfg.D_MODEL, device=x.device)

        # Handle done flags for GRU state reset.
        if has_seq_dim and done is not None:
            # Process step-by-step with state resets.
            outputs = []
            for t in range(seq_len):
                current_mask = 1.0 - done[:, t].view(1, -1, 1)
                gru_state = gru_state * current_mask  # Reset state on done.
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
        """
        Auxiliary world model prediction.
        
        Predicts next state and reward for representation learning.
        
        Args:
            actor_features: Features from actor pipeline.
            action: Action taken.
            
        Returns:
            Tuple of (predicted_next_state, predicted_reward).
        """
        if actor_features.ndim == 3: actor_features = actor_features.reshape(-1, self.cfg.D_MODEL)
        if action.ndim == 3: action = action.reshape(-1, self.cfg.ACTION_DIM)
        inp = torch.cat([actor_features, action], dim=-1)
        preds = self.world_model(inp)
        return preds[:, :self.cfg.NODE_DIM], preds[:, -1]

    def _process_critic_graph(self, graph_data, obs_flat):
        """
        Critic Graph Processing with Attention.
        
        Processes the entity graph through GNN and computes
        team-aware embeddings with attention.
        
        Args:
            graph_data: PyG graph batch.
            obs_flat: Flattened observations for attention query.
            
        Returns:
            Tuple of (subject_emb, ally_emb, enemy_emb).
        """
        x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch

        # 1. GNN Processing.
        node_embeddings = self.shared_physics_encoder(x)
        x_gnn = torch.relu(self.gnn_conv1(node_embeddings, edge_index, edge_attr))
        gnn_out = torch.relu(self.gnn_conv2(x_gnn, edge_index, edge_attr))

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        num_graphs = batch.max().item() + 1

        # Team-aware pooling.
        is_blue = x[:, 1]  # Team flag.
        ally_mask = (is_blue > 0.5)
        enemy_mask = (is_blue <= 0.5)

        ally_emb = global_mean_pool(gnn_out[ally_mask], batch[ally_mask], size=num_graphs)
        enemy_emb = global_mean_pool(gnn_out[enemy_mask], batch[enemy_mask], size=num_graphs)

        if ally_mask.sum() == 0: ally_emb = torch.zeros(num_graphs, 128, device=x.device)
        if enemy_mask.sum() == 0: enemy_emb = torch.zeros(num_graphs, 128, device=x.device)

        # 2. Attention for agent-specific context.
        ego_raw = obs_flat[:, :self.cfg.NODE_DIM]
        query = self.shared_physics_encoder(ego_raw)

        dense_keys, mask = to_dense_batch(node_embeddings, batch)
        dense_vals, _ = to_dense_batch(gnn_out, batch)

        # Broadcast for multiple agents per graph.
        if query.shape[0] >= num_graphs and num_graphs > 0:
            agents_per_env = query.shape[0] // num_graphs
            keys = dense_keys.repeat_interleave(agents_per_env, dim=0)
            vals = dense_vals.repeat_interleave(agents_per_env, dim=0)
            mask_expanded = mask.repeat_interleave(agents_per_env, dim=0)
            ally_emb = ally_emb.repeat_interleave(agents_per_env, dim=0)
            enemy_emb = enemy_emb.repeat_interleave(agents_per_env, dim=0)
        else:
            keys, vals, mask_expanded = dense_keys, dense_vals, mask

        # Compute attention.
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * self.attention_scale
        scores = scores.masked_fill(~mask_expanded.unsqueeze(1), -1e4)
        weights = F.softmax(scores, dim=-1)
        subject_emb = torch.bmm(weights, vals).squeeze(1)

        return subject_emb, ally_emb, enemy_emb

    def get_value(self, graph_batch, obs, gru_state=None, done=None):
        """
        Compute state value using critic network.
        
        Args:
            graph_batch: PyG graph batch.
            obs: Observations.
            gru_state: Unused (for API compatibility).
            done: Unused (for API compatibility).
            
        Returns:
            Value estimates [batch, 1] or [batch, seq, 1].
        """
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
        """
        Get action from policy and value from critic.
        
        Main inference method combining actor and critic pipelines.
        
        Args:
            obs: Observations.
            graph_data: PyG graph batch for critic.
            action: If provided, compute log_prob for this action (for PPO).
            gru_state: Hidden state for GRU.
            done: Done flags for GRU state reset.
            
        Returns:
            Tuple of (action, log_prob, entropy, value, new_gru_state).
        """
        # 1. Actor Pipeline.
        actor_features, new_gru_state = self.extract_actor_features(obs, gru_state, done)

        # 2. Action Head (Recursive TRM).
        action_mean = self.actor_head(actor_features)

        if action_mean.ndim == 3:
            action_mean = action_mean.reshape(-1, self.cfg.ACTION_DIM)

        # Clamp log_std for numerical stability and exploration control.
        # Range [-2.0, 0.0] corresponds to std in [0.135, 1.0].
        logstd_clamped = torch.clamp(self.actor_logstd, -2.0, 0.0)
        action_std = torch.exp(logstd_clamped).expand_as(action_mean)

        # Create action distribution.
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            # NO CLAMP HERE: Keep raw sample for PPO validity.
        else:
            if action.ndim == 3:
                action = action.reshape(-1, self.cfg.ACTION_DIM)

        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)

        # 3. Critic Pipeline (GNN).
        value = None
        if graph_data is not None:
            if obs.ndim == 3:
                obs_flat = obs.reshape(-1, self.cfg.OBS_DIM)
            else:
                obs_flat = obs

            subject_emb, ally_emb, enemy_emb = self._process_critic_graph(graph_data, obs_flat)
            critic_input = torch.cat([subject_emb, ally_emb, enemy_emb], dim=1)
            value = self.critic_head(critic_input)

            if actor_features.ndim == 3:
                value = value.reshape(actor_features.shape[0], actor_features.shape[1], 1)

        return action, log_prob, entropy, value, new_gru_state

    def get_action_history(self, obs):
        """
        Get action refinement history for deep supervision.
        
        Used during pretraining to supervise all intermediate
        actions from the TRM, not just the final output.
        
        Args:
            obs: Observations.
            
        Returns:
            List of action tensors from each TRM recursion.
        """
        actor_features, _ = self.extract_actor_features(obs)
        return self.actor_head(actor_features, return_history=True)

    def get_policy_distribution(self, action_mean):
        """
        Get the policy distribution for given action means.
        
        Used during pretraining for NLL loss computation.
        
        Args:
            action_mean: Action means from TRM head.
            
        Returns:
            torch.distributions.Normal distribution.
        """
        logstd_clamped = torch.clamp(self.actor_logstd, -2.0, 0.0)
        action_std = torch.exp(logstd_clamped).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)