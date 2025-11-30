# ================================================
# FILE: src/model.py
# ================================================

import torch
import torch.nn as nn
import numpy as np
from config import Config


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization for neural network layers.
    Helps maintain gradient magnitude through deep networks, critical for PPO stability.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentTransformer(nn.Module):
    """
    Hybrid Transformer-LSTM Architecture for Air Combat.
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config

        # === 1. FEATURE EMBEDDING ===
        self.embed = nn.Sequential(
            layer_init(nn.Linear(self.cfg.FEAT_DIM, self.cfg.D_MODEL)),
            nn.ReLU()
        )

        # === 2. TRANSFORMER ENCODER ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL,
            nhead=self.cfg.N_HEADS,
            dim_feedforward=512,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.cfg.N_LAYERS
        )

        # === 3. LSTM MEMORY ===
        self.lstm = nn.LSTM(
            input_size=self.cfg.D_MODEL,
            hidden_size=self.cfg.D_MODEL,
            batch_first=True
        )

        # === 4. CLS TOKEN ===
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg.D_MODEL))

        # === 5. ACTOR HEAD ===
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, self.cfg.ACTION_DIM), std=0.01)
        )

        # Bias throttle (index 2) to be high initially to prevent stalling
        with torch.no_grad():
            self.actor_mean[-1].bias[2].fill_(1.0)

        self.actor_logstd = nn.Parameter(torch.zeros(1, self.cfg.ACTION_DIM))

        # === 6. CRITIC HEAD ===
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def get_states(self, x, lstm_state=None, done=None):
        # 1. Handle Input Shapes (Batch vs Sequence)
        if x.dim() == 3:  # Sequence Input: (Batch, Seq_Len, Obs_Dim)
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.cfg.OBS_DIM)
        else:  # Single Step Input: (Batch, Obs_Dim)
            batch_size, seq_len = x.shape[0], 1
            x_flat = x

        # 2. Reshape Flat Obs -> Entity List
        current_batch_size = x_flat.shape[0]
        x_reshaped = x_flat.view(current_batch_size, self.cfg.MAX_ENTITIES, self.cfg.FEAT_DIM)

        # 3. Generate Attention Mask
        entity_teams = x_reshaped[:, :, 5]
        src_key_padding_mask = (entity_teams == 0.0)

        # 4. Embed Features
        embeddings = self.embed(x_reshaped)

        # 5. Prepend CLS Token
        cls_tokens = self.cls_token.expand(current_batch_size, -1, -1)
        embeddings_with_cls = torch.cat([cls_tokens, embeddings], dim=1)

        cls_mask = torch.zeros(current_batch_size, 1, dtype=torch.bool, device=x.device)
        mask_with_cls = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        # 6. Transformer Pass
        context = self.transformer(embeddings_with_cls, src_key_padding_mask=mask_with_cls)

        # 7. Extract States
        global_state_flat = context[:, 0, :]
        ego_state_flat = context[:, 1, :]

        # 8. LSTM Pass
        ego_state_seq = ego_state_flat.reshape(batch_size, seq_len, self.cfg.D_MODEL)

        if lstm_state is None:
            device = x.device
            h0 = torch.zeros(1, batch_size, self.cfg.D_MODEL, device=device)
            c0 = torch.zeros(1, batch_size, self.cfg.D_MODEL, device=device)
            lstm_state = (h0, c0)

        lstm_out, new_lstm_state = self.lstm(ego_state_seq, lstm_state)

        # Flatten back for Heads: (Batch*Seq, D_Model)
        actor_features = lstm_out.reshape(-1, self.cfg.D_MODEL)

        return actor_features, global_state_flat, new_lstm_state

    def get_value(self, x, global_state=None, lstm_state=None, done=None):
        input_obs = global_state if global_state is not None else x
        _, global_context, _ = self.get_states(input_obs, lstm_state, done)
        return self.critic(global_context)

    def get_action_and_value(self, x, global_state=None, action=None, lstm_state=None, done=None):
        """
        Main PPO method: Get action, log_prob, entropy, value, and next memory state.
        """
        # 1. Forward Pass (Actor Path)
        # actor_features is flattened: (Batch*Seq, D_Model)
        actor_features, local_global_context, new_lstm_state = self.get_states(x, lstm_state, done)

        # 2. Select Global Context for Critic (CTDE)
        if global_state is not None:
            _, global_context, _ = self.get_states(global_state, lstm_state, done)
        else:
            global_context = local_global_context

        # 3. Actor Head
        action_mean = self.actor_mean(actor_features)
        action_mean = torch.tanh(action_mean)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)

        # --- FIX: FLATTEN ACTION IF NEEDED ---
        # The input 'action' might be (Batch, Seq, 5) if coming from PPO buffer.
        # The distribution 'probs' is (Batch*Seq, 5).
        # We must flatten 'action' to match 'probs'.
        if action is not None:
            if action.dim() == 3:  # (Batch, Seq, ActDim)
                action = action.reshape(-1, self.cfg.ACTION_DIM)
        # -------------------------------------

        if action is None:
            action = probs.sample()

        # 4. Critic Head
        value = self.critic(global_context)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value, new_lstm_state