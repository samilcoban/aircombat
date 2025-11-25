# Import PyTorch libraries for optimization and gradient scaling
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent implementation.
    
    PPO is an on-policy RL algorithm that optimizes a clipped surrogate objective
    to prevent large policy updates that can destabilize training. This implementation
    includes:
    - Clipped policy gradient objective (prevents destructive updates)
    - Value function loss with coefficient
    - Entropy bonus for exploration
    - Advantage normalization for stability
    - Mixed precision training support (GPU acceleration)
    - Gradient clipping for stability
    
    Key hyperparameters:
    - CLIP_COEF: Clipping range for policy ratio (typically 0.2)
    - UPDATE_EPOCHS: Number of optimization epochs per batch
    - MINIBATCH_SIZE: Size of minibatches for SGD
    - ENT_COEF: Entropy coefficient for exploration bonus
    - VF_COEF: Value function loss coefficient
    """
    
    def __init__(self, model):
        """
        Initialize PPO agent with policy network and optimizer.
        
        Args:
            model: AgentTransformer neural network (actor-critic architecture)
        """
        self.cfg = Config
        # Move model to configured device (CPU or CUDA)
        self.model = model.to(self.cfg.DEVICE)
        # Adam optimizer with small epsilon for numerical stability
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE, eps=1e-5)

    def update(self, obs, actions, logprobs, returns, advantages, global_states=None, scaler=None):
        """
        Perform PPO update on collected rollout data.
        
        This is the core PPO training loop that:
        1. Converts rollout data to tensors on correct device
        2. Performs multiple epochs of minibatch SGD
        3. Computes clipped policy loss, value loss, and entropy bonus
        4. Updates policy parameters via backpropagation
        
        PPO Objective:
        L = L^CLIP(θ) - c₁ * L^VF(θ) + c₂ * H[π_θ]
        
        where:
        - L^CLIP: Clipped surrogate policy loss
        - L^VF: Value function mean squared error
        - H: Entropy bonus for exploration
        
        Args:
            obs: Observations from rollout (N, OBS_DIM)
            actions: Actions taken (N, ACTION_DIM)
            logprobs: Log probabilities of actions under old policy (N,)
            returns: Discounted returns (targets for critic) (N,)
            advantages: Advantage estimates (N,)
            advantages: Advantage estimates (N,)
            global_states: Optional global states for CTDE (N, OBS_DIM)
            scaler: Optional GradScaler for mixed precision training
            
        Returns:
            float: Final loss value for logging
        """
        # === PREPARE DATA ===
        # Helper function to ensure tensors are on correct device (GPU/CPU)
        def to_device(x):
            """Convert input to tensor on configured device."""
            if isinstance(x, torch.Tensor):
                return x.to(self.cfg.DEVICE)  # Already tensor, just move
            return torch.tensor(x, dtype=torch.float32).to(self.cfg.DEVICE)  # Convert from numpy

        # Move all rollout data to device
        b_obs = to_device(obs)
        b_actions = to_device(actions)
        b_logprobs = to_device(logprobs)          # Old policy log probs (fixed)
        b_returns = to_device(returns)            # Targets for value function
        b_returns = to_device(returns)            # Targets for value function
        b_advantages = to_device(advantages)      # Advantage estimates A(s,a)
        
        # Handle global states for CTDE
        b_global_states = None
        if global_states is not None:
            b_global_states = to_device(global_states)

        # Dynamic batch sizing and shuffling for stochastic gradient descent
        batch_size = b_obs.shape[0]
        # Shuffle indices for random minibatch sampling
        indices = torch.randperm(batch_size, device=self.cfg.DEVICE)

        loss_val = 0.0  # Track final loss for logging

        # === PPO OPTIMIZATION LOOP ===
        # Multiple epochs over the same data (on-policy optimization)
        for _ in range(self.cfg.UPDATE_EPOCHS):
            # Iterate over minibatches
            for start in range(0, batch_size, self.cfg.MINIBATCH_SIZE):
                end = start + self.cfg.MINIBATCH_SIZE
                mb_idx = indices[start:end]  # Random minibatch indices

                # === FORWARD PASS ===
                # Re-evaluate actions under current policy (new log probs)
                # Note: Mixed precision (autocast) is handled by training loop wrapper
                # Note: Mixed precision (autocast) is handled by training loop wrapper
                _, new_logprob, entropy, new_value = self.model.get_action_and_value(
                    b_obs[mb_idx], 
                    global_state=b_global_states[mb_idx] if b_global_states is not None else None,
                    action=b_actions[mb_idx]
                )

                # === COMPUTE POLICY LOSS (PPO Clipped Objective) ===
                # Calculate probability ratio: π_θ(a|s) / π_θ_old(a|s)
                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = logratio.exp()  # r(θ) = exp(log π_θ - log π_θ_old)

                # Advantage normalization (per-minibatch)
                # Reduces variance and stabilizes training
                mb_adv = b_advantages[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # PPO Clipped Objective:
                # L^CLIP = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
                # 
                # This prevents large policy updates by clipping the ratio:
                # - If advantage is positive (good action), clip prevents over-optimism
                # - If advantage is negative (bad action), clip prevents over-pessimism
                pg_loss1 = -mb_adv * ratio  # Unclipped objective
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.cfg.CLIP_COEF, 1 + self.cfg.CLIP_COEF)  # Clipped
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # Pessimistic bound (take worst case)

                # === COMPUTE VALUE LOSS ===
                # Mean squared error between predicted values and returns
                # L^VF = E[(V(s) - V_target)²]
                v_loss = 0.5 * ((new_value.view(-1) - b_returns[mb_idx]) ** 2).mean()

                # === TOTAL LOSS ===
                # Combine all objectives:
                # L = L^CLIP - c₁ * H[π] + c₂ * L^VF
                # - Policy loss (maximize expected advantage)
                # - Entropy bonus (maximize exploration, negative because we minimize loss)
                # - Value loss (minimize prediction error)
                loss = pg_loss - (self.cfg.ENT_COEF * entropy.mean()) + (self.cfg.VF_COEF * v_loss)

                # === BACKPROPAGATION ===
                self.optimizer.zero_grad()  # Clear gradients from previous step
                
                # Mixed Precision Training (if enabled)
                if scaler is not None:
                    # Gradient scaling for numerical stability with fp16
                    scaler.scale(loss).backward()              # Backward pass with scaled loss
                    scaler.unscale_(self.optimizer)            # Unscale gradients before clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)  # Clip
                    scaler.step(self.optimizer)                # Update weights
                    scaler.update()                            # Update scaler state
                else:
                    # Standard fp32 training
                    loss.backward()                            # Backward pass
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)  # Clip
                    self.optimizer.step()                      # Update weights

                loss_val = loss.item()  # Save for logging

        return loss_val