import torch
import torch.nn as nn
import numpy as np
from config import Config


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = Config

        # 1. Feature Embedding
        self.embed = nn.Sequential(
            layer_init(nn.Linear(self.cfg.FEAT_DIM, self.cfg.D_MODEL)),
            nn.ReLU()
        )

        # 2. Transformer Encoder (The "Tactical Brain")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL,
            nhead=self.cfg.N_HEADS,
            dim_feedforward=512,  # Increased from 256
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.N_LAYERS)

        # 3. Actor Head (Decisions)
        # Input: The embedding of the Agent itself (Index 0)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),  # Increased from 64
            nn.Tanh(),
            layer_init(nn.Linear(128, self.cfg.ACTION_DIM), std=0.01)  # Increased from 64
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.cfg.ACTION_DIM))

        # 4. Critic Head (Global Situation Evaluator)
        # Input: Max-Pooled embedding of ALL entities (Global Context)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),  # Increased from 64
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)  # Increased from 64
        )

    def get_states(self, x):
        """
        x shape: (Batch, MAX_ENTITIES * FEAT_DIM) -> Flattened
        Returns: (Actor_Feats, Global_Feats)
        """
        batch_size = x.shape[0]

        # Reshape to (Batch, Entities, Features)
        x = x.view(batch_size, self.cfg.MAX_ENTITIES, self.cfg.FEAT_DIM)

        # Embed
        embeddings = self.embed(x)  # (B, N, D)

        # Process Interactions via Transformer
        # Note: We don't mask "empty" padding entities here for speed,
        # assuming the network learns they are zero-vectors (type=0) and ignores them.
        context = self.transformer(embeddings)  # (B, N, D)

        # Entity-Centric: Index 0 is ALWAYS the agent acting
        ego_state = context[:, 0, :]

        # Global Context: Max Pool over all entities to get "Battlefield State"
        global_state, _ = torch.max(context, dim=1)

        return ego_state, global_state

    def get_value(self, x):
        _, global_state = self.get_states(x)
        return self.critic(global_state)

    def get_action_and_value(self, x, action=None):
        ego_state, global_state = self.get_states(x)

        # Critic
        value = self.critic(global_state)

        # Actor
        action_mean = self.actor_mean(ego_state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value