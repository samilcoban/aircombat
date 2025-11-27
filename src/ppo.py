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

    def update(self, obs, actions, logprobs, returns, advantages, global_states=None, lstm_states=None, dones=None, scaler=None):
        """
        Perform PPO update on collected rollout data.
        
        Args:
            obs: Observations (N, OBS_DIM)
            actions: Actions (N, ACTION_DIM)
            logprobs: Log probs (N,)
            returns: Returns (N,)
            advantages: Advantages (N,)
            global_states: Optional global states (N, OBS_DIM)
            lstm_states: Optional tuple of (h, c) states (N, 1, D)
            dones: Optional done flags (N,)
            scaler: Optional GradScaler
            
        Returns:
            float: Final loss value
        """
        # Sequence Length for BPTT
        SEQ_LEN = 32
        # === PREPARE DATA ===
        # Helper function to ensure tensors are on correct device (GPU/CPU)
        def to_device(x):
            """Convert input to tensor on configured device."""
            if isinstance(x, torch.Tensor):
                return x.to(self.cfg.DEVICE)  # Already tensor, just move
            return torch.tensor(x, dtype=torch.float32).to(self.cfg.DEVICE)  # Convert from numpy

        # Move all rollout data to device
        # Move all rollout data to device
        b_obs = to_device(obs)
        b_actions = to_device(actions)
        b_logprobs = to_device(logprobs)
        b_returns = to_device(returns)
        b_advantages = to_device(advantages)
        b_dones = to_device(dones) if dones is not None else None
        
        # Handle global states
        b_global_states = None
        if global_states is not None:
            b_global_states = to_device(global_states)

        # Handle LSTM States
        # lstm_states is tuple (h, c). Each is (N, 1, D) or (1, N, D) depending on storage
        # We assume storage is (N, D) or similar.
        # Let's assume passed as tuple of tensors (N, D)
        b_lstm_h = None
        b_lstm_c = None
        if lstm_states is not None:
            # lstm_states is (h_tensor, c_tensor)
            # h_tensor shape: (N, 1, D) or (N, D)
            b_lstm_h = to_device(lstm_states[0])
            b_lstm_c = to_device(lstm_states[1])

        # === SEQUENCE RESHAPING ===
        # If using LSTM, we must reshape flat batch (N, ...) into sequences (N/SEQ_LEN, SEQ_LEN, ...)
        batch_size = b_obs.shape[0]
        use_lstm = (b_lstm_h is not None)
        
        if use_lstm:
            num_seqs = batch_size // SEQ_LEN
            if batch_size % SEQ_LEN != 0:
                print(f"Warning: Batch size {batch_size} not divisible by SEQ_LEN {SEQ_LEN}. Truncating.")
                # Truncate to multiple of SEQ_LEN
                trunc_len = num_seqs * SEQ_LEN
                b_obs = b_obs[:trunc_len]
                b_actions = b_actions[:trunc_len]
                b_logprobs = b_logprobs[:trunc_len]
                b_returns = b_returns[:trunc_len]
                b_advantages = b_advantages[:trunc_len]
                if b_dones is not None: b_dones = b_dones[:trunc_len]
                if b_global_states is not None: b_global_states = b_global_states[:trunc_len]
                b_lstm_h = b_lstm_h[:trunc_len]
                b_lstm_c = b_lstm_c[:trunc_len]
                batch_size = trunc_len

            # Reshape to (NumSeqs, SeqLen, ...)
            # We assume data is already ordered by (Env, Time) so reshaping works
            def make_seq(x):
                return x.reshape(num_seqs, SEQ_LEN, *x.shape[1:])
            
            s_obs = make_seq(b_obs)
            s_actions = make_seq(b_actions)
            s_logprobs = make_seq(b_logprobs)
            s_returns = make_seq(b_returns)
            s_advantages = make_seq(b_advantages)
            s_dones = make_seq(b_dones) if b_dones is not None else None
            s_global_states = make_seq(b_global_states) if b_global_states is not None else None
            
            # For LSTM states, we only need the INITIAL state for each sequence
            # b_lstm_h is (N, 1, D). Reshape to (NumSeqs, SeqLen, 1, D)
            # Take index 0 along SeqLen dim -> (NumSeqs, 1, D)
            s_lstm_h_init = b_lstm_h.reshape(num_seqs, SEQ_LEN, *b_lstm_h.shape[1:])[:, 0]
            s_lstm_c_init = b_lstm_c.reshape(num_seqs, SEQ_LEN, *b_lstm_c.shape[1:])[:, 0]
            
            # Permute for LSTM input (1, Batch, D)
            # Current: (NumSeqs, D). Unsqueeze to (1, NumSeqs, D)
            s_lstm_h_init = s_lstm_h_init.unsqueeze(0)
            s_lstm_c_init = s_lstm_c_init.unsqueeze(0)
            
            # Update batch size to number of sequences
            optim_batch_size = num_seqs
        else:
            optim_batch_size = batch_size

        # Shuffle indices
        indices = torch.randperm(optim_batch_size, device=self.cfg.DEVICE)

        loss_val = 0.0  # Track final loss for logging

        # === ADVANTAGE NORMALIZATION (BATCH-WISE) ===
        # Standard PPO practice: Normalize advantages across the entire batch
        # before minibatch iteration. This reduces noise in gradient estimates.
        if use_lstm:
             # s_advantages is (NumSeqs, SeqLen) -> Flatten to normalize
             flat_adv = s_advantages.flatten()
             mean_adv = flat_adv.mean()
             std_adv = flat_adv.std()
             s_advantages = (s_advantages - mean_adv) / (std_adv + 1e-8)
        else:
             # b_advantages is (N,)
             b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # === PPO OPTIMIZATION LOOP ===
        # Multiple epochs over the same data (on-policy optimization)
        for _ in range(self.cfg.UPDATE_EPOCHS):
            # Iterate over minibatches
            # Iterate over minibatches
            # If using LSTM, MINIBATCH_SIZE refers to number of SEQUENCES?
            # Or samples? Usually samples.
            # If samples, we need to adjust stride.
            # Let's assume MINIBATCH_SIZE is samples (e.g. 512).
            # If using LSTM, we step by MINIBATCH_SIZE // SEQ_LEN sequences.
            
            step_size = self.cfg.MINIBATCH_SIZE
            if use_lstm:
                step_size = self.cfg.MINIBATCH_SIZE // SEQ_LEN
                if step_size < 1: step_size = 1
            
            for start in range(0, optim_batch_size, step_size):
                end = start + step_size
                mb_idx = indices[start:end]  # Random minibatch indices (sequences or samples)

                # === FORWARD PASS ===
                if use_lstm:
                    # Get minibatch of sequences
                    mb_obs = s_obs[mb_idx]           # (MB, Seq, Obs)
                    mb_actions = s_actions[mb_idx]   # (MB, Seq, Act)
                    mb_global = s_global_states[mb_idx] if s_global_states is not None else None # (MB, Seq, Obs)
                    mb_dones = s_dones[mb_idx] if s_dones is not None else None # (MB, Seq)
                    
                    # Initial LSTM state for this minibatch
                    # (1, MB, D)
                    mb_h = s_lstm_h_init[:, mb_idx, :]
                    mb_c = s_lstm_c_init[:, mb_idx, :]
                    mb_lstm_state = (mb_h, mb_c)
                    
                    # Run Model on Sequence
                    # Returns (MB, Seq, ...)
                    _, new_logprob, entropy, new_value, _ = self.model.get_action_and_value(
                        mb_obs,
                        global_state=mb_global,
                        action=mb_actions,
                        lstm_state=mb_lstm_state,
                        done=mb_dones
                    )
                    
                    # Flatten outputs for loss calculation
                    # (MB, Seq, ...) -> (MB*Seq, ...)
                    new_logprob = new_logprob.flatten()
                    entropy = entropy.flatten()
                    new_value = new_value.flatten()
                    
                    # Get targets corresponding to these sequences
                    mb_logprobs_old = s_logprobs[mb_idx].flatten()
                    mb_returns = s_returns[mb_idx].flatten()
                    mb_advantages = s_advantages[mb_idx].flatten()
                    
                else:
                    # Standard FeedForward
                    _, new_logprob, entropy, new_value, _ = self.model.get_action_and_value(
                        b_obs[mb_idx], 
                        global_state=b_global_states[mb_idx] if b_global_states is not None else None,
                        action=b_actions[mb_idx]
                    )
                    mb_logprobs_old = b_logprobs[mb_idx]
                    mb_returns = b_returns[mb_idx]
                    mb_advantages = b_advantages[mb_idx]

                # === COMPUTE POLICY LOSS (PPO Clipped Objective) ===
                logratio = new_logprob - mb_logprobs_old
                ratio = logratio.exp()

                # Advantage normalization
                # ALREADY DONE BATCH-WISE
                mb_adv = mb_advantages

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
                v_loss = 0.5 * ((new_value.view(-1) - mb_returns) ** 2).mean()

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