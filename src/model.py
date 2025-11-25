# Import PyTorch libraries for neural network construction
import torch
import torch.nn as nn
import numpy as np
from config import Config


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize neural network layer weights using orthogonal initialization.
    
    Orthogonal initialization helps with gradient flow in deep networks by
    preserving gradient magnitudes across layers. This is particularly important
    for PPO's value function which needs stable gradients.
    
    Args:
        layer: PyTorch Linear layer to initialize
        std: Standard deviation for orthogonal initialization (default: sqrt(2))
        bias_const: Constant value for bias initialization (default: 0.0)
        
    Returns:
        layer: The initialized layer (for chaining)
    """
    torch.nn.init.orthogonal_(layer.weight, std)  # Orthogonal weight init
    torch.nn.init.constant_(layer.bias, bias_const)  # Constant bias init
    return layer


class AgentTransformer(nn.Module):
    """
    Transformer-based Actor-Critic network for air combat.
    
    Architecture:
    1. Feature Embedding: Projects raw entity features to transformer dimension
    2. Transformer Encoder: Processes entity interactions via self-attention
    3. Actor Head: Generates continuous actions from ego state
    4. Critic Head: Evaluates state value from global context
    
    Key Design Choices:
    - Entity-centric observation: Index 0 is always the acting agent
    - Self-attention: Transformer learns tactical relationships (e.g., "missile threat", "tail position")
    - Dual representation: Ego state for actions, global state for value
    - Continuous actions: Gaussian policy for smooth control
    
    Input: Flattened observation (MAX_ENTITIES × FEAT_DIM)
    Output: Actions (5D continuous), log_probs, entropy, value
    """
    
    def __init__(self):
        """
        Initialize the Transformer-based policy and value networks.
        
        Network components:
        - Embedding layer: Maps 17D entity features → 64D (or D_MODEL)
        - Transformer: 2-4 layers of multi-head self-attention
        - Actor: Maps ego embedding → action means (5D)
        - Critic: Maps global embedding → value scalar
        """
        super().__init__()
        self.cfg = Config  # Global configuration parameters

        # === 1. FEATURE EMBEDDING ===
        # Projects raw entity features (17D) to transformer model dimension (64D)
        # Input: (Batch, Entities, 17) → Output: (Batch, Entities, D_MODEL)
        self.embed = nn.Sequential(
            layer_init(nn.Linear(self.cfg.FEAT_DIM, self.cfg.D_MODEL)),
            nn.ReLU()  # Nonlinearity for expressiveness
        )

        # === 2. TRANSFORMER ENCODER ("Tactical Brain") ===
        # Multi-head self-attention processes entity interactions
        # Learns tactical concepts like:
        # - "This missile is tracking me" (MAWS + position correlation)
        # - "I'm behind the enemy" (geometry + relative velocity)
        # - "Enemy has lock on me" (RWR signal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.D_MODEL,       # Embedding dimension (64/128)
            nhead=self.cfg.N_HEADS,         # Number of attention heads (4)
            dim_feedforward=512,             # FFN hidden size (increased from 256 for capacity)
            batch_first=True,                # Input shape: (Batch, Seq, Features)
            norm_first=True                  # Pre-LayerNorm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.cfg.N_LAYERS  # Stack 2-4 layers for deeper reasoning
        )

        # === 2.5. CLS TOKEN (Global Context Aggregation) ===
        # Learnable token prepended to entity sequence (like BERT [CLS])
        # Transformer updates this token based on ALL entities via attention
        # Replaces max pooling for better multi-modal information preservation
        # Example: Can attend to BOTH missiles from different angles, not just loudest
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg.D_MODEL))

        # === 3. ACTOR HEAD (Policy Network) ===
        # Generates action distribution parameters from ego agent's embedding
        # Input: Ego state embedding (index 1 after transformer, shifted by CLS)
        # Output: Action means (μ) for 5D continuous action
        # Actions: [roll_rate, g_pull, throttle, fire, countermeasures]
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),  # Increased from 64 for expressiveness
            nn.Tanh(),  # Bounded activation for stability
            layer_init(nn.Linear(128, self.cfg.ACTION_DIM), std=0.01)  # Small std for initial exploration
        )
        # Action standard deviation (log scale, learned parameter)
        # Shared across all action dimensions but learned during training
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.cfg.ACTION_DIM))

        # === 4. CRITIC HEAD (Value Network) ===
        # Evaluates expected return from global battlefield state
        # Input: CLS token (attention-based global aggregation)
        # Output: Scalar value estimate V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),  # Increased from 64
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)  # Value head (std=1.0 for appropriate scale)
        )

    def get_states(self, x):
        """
        Process flattened observation through transformer to extract ego and global state.
        
        ARCHITECTURE: CLS Token + Attention Masking
        - Prepends learnable [CLS] token for global context aggregation
        - Masks padding to prevent noise from zero-padded entities
        - Uses attention-based pooling (CLS) instead of max pooling
        
        This is the core forward pass that transforms raw observations into
        meaningful tactical representations:
        1. Reshape flat observation → entity list
        2. Embed each entity independently
        3. Prepend CLS token to sequence
        4. Generate attention mask from team field
        5. Apply transformer with masking
        6. Extract ego state (index 1) and global state (CLS token)
        
        Args:
            x: Flattened observation tensor
               Shape: (Batch, MAX_ENTITIES × FEAT_DIM)
               
        Returns:
            tuple: (ego_state, global_state)
                ego_state: Embedding of the acting agent (Batch, D_MODEL)
                global_state: CLS token aggregating all entity information (Batch, D_MODEL)
        """
        batch_size = x.shape[0]

        # === RESHAPE TO ENTITY LIST ===
        # Convert flat observation to structured format
        # (Batch, Flat) → (Batch, Entities, Features)
        x = x.view(batch_size, self.cfg.MAX_ENTITIES, self.cfg.FEAT_DIM)

        # === GENERATE ATTENTION MASK ===
        # Use team field (index 5): padding=0.0, real entities=±1.0
        entity_teams = x[:, :, 5]  # Extract team field
        src_key_padding_mask = (entity_teams == 0.0)  # True for padding, False for real

        # === EMBEDDING ===
        # Project each entity's features to transformer dimension
        embeddings = self.embed(x)  # (B, N=MAX_ENTITIES, D=D_MODEL)

        # === PREPEND CLS TOKEN ===
        # Add learnable [CLS] token at index 0 (before ego entity)
        # CLS will aggregate global battlefield context via attention
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        embeddings_with_cls = torch.cat([cls_tokens, embeddings], dim=1)  # (B, N+1, D)
        
        # Update mask: CLS token is NOT padding (False)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        mask_with_cls = torch.cat([cls_mask, src_key_padding_mask], dim=1)  # (B, N+1)

        # === TRANSFORMER PROCESSING WITH MASKING ===
        # Self-attention allows each entity (+ CLS) to attend to all others
        # CLS token aggregates information from all entities via attention
        # This preserves multi-modal information (e.g., multiple threats)
        context = self.transformer(
            embeddings_with_cls, 
            src_key_padding_mask=mask_with_cls
        )  # (B, N+1, D)

        # === STATE EXTRACTION ===
        # Index 0: CLS token (global battlefield summary via attention)
        # Index 1: Ego entity (acting agent's self-awareness)
        # Indices 2+: Other entities
        global_state = context[:, 0, :]  # CLS token for critic (B, D)
        ego_state = context[:, 1, :]     # Ego entity for actor (B, D)

        return ego_state, global_state

    def get_value(self, x, global_state=None):
        """
        Evaluate state value (expected return) for critic training.
        
        CTDE Support:
        If global_state is provided (Centralized Training), uses privileged info.
        Otherwise falls back to local observation x.
        
        Args:
            x: Local observation tensor (Batch, OBS_DIM)
            global_state: Optional unmasked global state (Batch, OBS_DIM)
            
        Returns:
            torch.Tensor: Value estimates V(s) (Batch, 1)
        """
        # Use global state if available (CTDE), else local obs
        critic_input = global_state if global_state is not None else x
        
        # Process through transformer to get global context (CLS token)
        _, global_context = self.get_states(critic_input)
        
        return self.critic(global_context)

    def get_action_and_value(self, x, global_state=None, action=None):
        """
        Get action distribution and value estimate.
        
        Args:
            x: Local observation (for Actor)
            global_state: Global state (for Critic) - CTDE
            action: Optional action to evaluate (for training)
            
        Returns:
            tuple: (action, logprob, entropy, value)
        """
        # 1. Actor uses LOCAL observation (Decentralized Execution)
        ego_state, local_global_context = self.get_states(x)
        
        # 2. Critic uses GLOBAL state if available (Centralized Training)
        if global_state is not None:
            _, global_context = self.get_states(global_state)
        else:
            global_context = local_global_context  # Fallback
            
        # Actor Head
        action_mean = self.actor_mean(ego_state)
        # CRITICAL: Apply tanh to bound actions to [-1, 1]
        # Without this, actions can be unbounded (e.g., roll_rate=9.0 instead of 1.0)
        action_mean = torch.tanh(action_mean)
        
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            # Clip sampled actions to ensure they stay in valid range
            action = torch.clamp(action, -1.0, 1.0)
            
        logprob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        
        # Critic Head
        value = self.critic(global_context)
        
        return action, logprob, entropy, value