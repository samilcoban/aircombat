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

        # === 3. ACTOR HEAD (Policy Network) ===
        # Generates action distribution parameters from ego agent's embedding
        # Input: Ego state embedding (index 0 after transformer)
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
        # Input: Max-pooled embedding of ALL entities (global situation awareness)
        # Output: Scalar value estimate V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.cfg.D_MODEL, 128)),  # Increased from 64
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)  # Value head (std=1.0 for appropriate scale)
        )

    def get_states(self, x):
        """
        Process flattened observation through transformer to extract ego and global state.
        
        This is the core forward pass that transforms raw observations into
        meaningful tactical representations:
        1. Reshape flat observation → entity list
        2. Embed each entity independently
        3. Apply transformer to model entity interactions
        4. Extract ego state (for actions) and global state (for value)
        
        Args:
            x: Flattened observation tensor
               Shape: (Batch, MAX_ENTITIES × FEAT_DIM)
               
        Returns:
            tuple: (ego_state, global_state)
                ego_state: Embedding of the acting agent (Batch, D_MODEL)
                global_state: Max-pooled embedding of all entities (Batch, D_MODEL)
        """
        batch_size = x.shape[0]

        # === RESHAPE TO ENTITY LIST ===
        # Convert flat observation to structured format
        # (Batch, Flat) → (Batch, Entities, Features)
        x = x.view(batch_size, self.cfg.MAX_ENTITIES, self.cfg.FEAT_DIM)

        # === EMBEDDING ===
        # Project each entity's features to transformer dimension
        # Each entity independently embedded (no interaction yet)
        embeddings = self.embed(x)  # (B, N=MAX_ENTITIES, D=D_MODEL)

        # === TRANSFORMER PROCESSING ===
        #Self-attention allows each entity to "attend" to all others
        # Network learns tactical relationships:
        # - Distance/angle to threats
        # - Enemy lock status (RWR correlation)
        # - Missile trajectory prediction (MAWS + velocity)
        # 
        # Note: No attention mask for padding (zero entities)
        # Network learns to ignore zero vectors (entity type = 0.0)
        context = self.transformer(embeddings)  # (B, N, D)

        # === EGO STATE EXTRACTION ===
        # Index 0 is ALWAYS the acting agent (entity-centric observation)
        # This is the agent's "self-awareness" after seeing the battlefield
        ego_state = context[:, 0, :]  # (B, D)

        # === GLOBAL STATE EXTRACTION ===
        # Max-pool over all entities to get "battlefield summary"
        # Critic uses this to evaluate overall situation value
        # Max-pool captures "most important feature" across all entities
        # (e.g., closest threat, highest energy state, etc.)
        global_state, _ = torch.max(context, dim=1)  # (B, D)

        return ego_state, global_state

    def get_value(self, x):
        """
        Evaluate state value (expected return) for critic training.
        
        Used during PPO advantage calculation to estimate V(s).
        
        Args:
            x: Observation tensor (Batch, OBS_DIM)
            
        Returns:
            torch.Tensor: Value estimates V(s) (Batch, 1)
        """
        _, global_state = self.get_states(x)  # Get global battlefield state
        return self.critic(global_state)  # Evaluate value

    def get_action_and_value(self, x, action=None):
        """
        Main forward pass for PPO training and inference.
        
        Generates actions from policy and evaluates state value.
        If action is provided, computes log probability and entropy (for PPO loss).
        If action is None, samples new action from policy (for rollout collection).
        
        Args:
            x: Observation tensor (Batch, OBS_DIM)
            action: Optional action tensor for evaluation (Batch, ACTION_DIM)
            
        Returns:
            tuple: (action, log_prob, entropy, value)
                action: Sampled or provided actions (Batch, ACTION_DIM)
                log_prob: Log probability of actions under policy (Batch,)
                entropy: Policy entropy for exploration bonus (Batch,)
                value: State value estimate (Batch, 1)
        """
        # Process observation to get state representations
        ego_state, global_state = self.get_states(x)

        # === CRITIC (Value Estimation) ===
        # Evaluate how good the current state is
        value = self.critic(global_state)

        # === ACTOR (Policy) ===
        # Generate Gaussian action distribution
        action_mean = self.actor_mean(ego_state)  # μ(s) - policy mean
        action_logstd = self.actor_logstd.expand_as(action_mean)  # log(σ) - broadcast to batch
        action_std = torch.exp(action_logstd)  # σ(s) - standard deviation

        # Create Gaussian distribution: π(a|s) = N(μ(s), σ(s))
        probs = torch.distributions.Normal(action_mean, action_std)

        # Sample action if not provided (rollout collection)
        if action is None:
            action = probs.sample()  # a ~ π(·|s)

        # Compute log probability: log π(a|s)
        # Sum over action dimensions (assumes independence)
        log_prob = probs.log_prob(action).sum(1)  # (Batch,)
        
        # Compute entropy: H[π(·|s)] for exploration bonus
        # Higher entropy = more exploration
        entropy = probs.entropy().sum(1)  # (Batch,)

        return action, log_prob, entropy, value