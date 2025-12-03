# ================================================
# FILE: src/ppo.py
# ================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import Config


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent implementation.
    
    PPO is a policy gradient method that alternates between sampling data through interaction with the environment
    and optimizing a "surrogate" objective function using stochastic gradient descent.
    
    Key features:
    - Clipped objective function to prevent large policy updates (stability).
    - Value function clipping (optional) to stabilize critic training.
    - Support for GRU recurrent policies via sequence handling.
    """

    def __init__(self, model):
        self.cfg = Config
        self.model = model.to(self.cfg.DEVICE)

        # Optimization: Compile Model
        # Intuition: PyTorch 2.0 compilation fuses kernels and optimizes the graph for faster execution.
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("✅ PyTorch 2.0 Compilation Enabled")
        except Exception as e:
            print(f"⚠️ PyTorch Compile Skipped: {e}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE, eps=1e-5)

    def update(self, obs, actions, logprobs, returns, advantages, global_states=None, gru_states=None, dones=None,
               old_values=None, scaler=None):
        """
        Update the policy and value networks using the collected experience buffer.
        """

        self.model.train()
        SEQ_LEN = 32

        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.cfg.DEVICE)
            return torch.tensor(x, dtype=torch.float32).to(self.cfg.DEVICE)

        b_obs = to_device(obs)
        b_actions = to_device(actions)
        b_logprobs = to_device(logprobs)
        b_returns = to_device(returns)
        b_advantages = to_device(advantages)
        b_dones = to_device(dones) if dones is not None else None

        b_old_values = None
        if old_values is not None:
            b_old_values = to_device(old_values)

        b_global_states = None
        if global_states is not None:
            b_global_states = to_device(global_states)

        b_gru_h = None
        if gru_states is not None:
            # GRU only has hidden state h
            b_gru_h = to_device(gru_states)

        batch_size = b_obs.shape[0]
        use_gru = (b_gru_h is not None)

        # --- SEQUENCE HANDLING ---
        # Intuition: For RNNs, we can't just shuffle random transitions. We must preserve temporal order.
        # We break the batch into sequences of length SEQ_LEN.
        if use_gru:
            num_seqs = batch_size // SEQ_LEN
            if batch_size % SEQ_LEN != 0:
                trunc_len = num_seqs * SEQ_LEN
                b_obs = b_obs[:trunc_len]
                b_actions = b_actions[:trunc_len]
                b_logprobs = b_logprobs[:trunc_len]
                b_returns = b_returns[:trunc_len]
                b_advantages = b_advantages[:trunc_len]
                if b_dones is not None: b_dones = b_dones[:trunc_len]
                if b_global_states is not None: b_global_states = b_global_states[:trunc_len]
                if b_old_values is not None: b_old_values = b_old_values[:trunc_len]
                b_gru_h = b_gru_h[:trunc_len]
                batch_size = trunc_len

            def make_seq(x):
                return x.reshape(num_seqs, SEQ_LEN, *x.shape[1:])

            s_obs = make_seq(b_obs)
            s_actions = make_seq(b_actions)
            s_logprobs = make_seq(b_logprobs)
            s_returns = make_seq(b_returns)
            s_advantages = make_seq(b_advantages)
            s_dones = make_seq(b_dones) if b_dones is not None else None
            s_global_states = make_seq(b_global_states) if b_global_states is not None else None
            s_old_values = make_seq(b_old_values) if b_old_values is not None else None

            # Extract initial GRU states for each sequence
            # Shape: (Batch, D_Model) -> (Num_Seqs, Seq_Len, D_Model) -> (Num_Seqs, D_Model)
            s_gru_h_init = b_gru_h.reshape(num_seqs, SEQ_LEN, *b_gru_h.shape[1:])[:, 0]
            
            # GRU expects (Num_Layers, Batch, Hidden) -> (1, Num_Seqs, Hidden)
            s_gru_h_init = s_gru_h_init.unsqueeze(0)

            optim_batch_size = num_seqs
        else:
            optim_batch_size = batch_size

        indices = torch.randperm(optim_batch_size, device=self.cfg.DEVICE)

        epoch_losses = []
        epoch_pg_losses = []
        epoch_v_losses = []
        epoch_entropies = []
        epoch_kls = []

        # Advantage Normalization
        # Intuition: Stabilizes training by ensuring advantages have 0 mean and 1 std dev.
        if use_gru:
            flat_adv = s_advantages.flatten()
            mean_adv = flat_adv.mean()
            std_adv = flat_adv.std()
            s_advantages = (s_advantages - mean_adv) / (std_adv + 1e-8)
        else:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        for _ in range(self.cfg.UPDATE_EPOCHS):
            step_size = self.cfg.MINIBATCH_SIZE
            if use_gru:
                step_size = max(1, self.cfg.MINIBATCH_SIZE // SEQ_LEN)

            for start in range(0, optim_batch_size, step_size):
                end = start + step_size
                mb_idx = indices[start:end]

                if use_gru:
                    mb_obs = s_obs[mb_idx]
                    mb_actions = s_actions[mb_idx]
                    mb_global = s_global_states[mb_idx] if s_global_states is not None else None
                    mb_dones = s_dones[mb_idx] if s_dones is not None else None

                    mb_h = s_gru_h_init[:, mb_idx, :]
                    mb_gru_state = mb_h

                    _, new_logprob, entropy, new_value, _ = self.model.get_action_and_value(
                        mb_obs,
                        graph_data=mb_global,
                        action=mb_actions,
                        gru_state=mb_gru_state,
                        done=mb_dones
                    )

                    new_logprob = new_logprob.flatten()
                    entropy = entropy.flatten()
                    new_value = new_value.flatten()

                    mb_logprobs_old = s_logprobs[mb_idx].flatten()
                    mb_returns = s_returns[mb_idx].flatten()
                    mb_advantages = s_advantages[mb_idx].flatten()
                    mb_old_values = s_old_values[mb_idx].flatten() if s_old_values is not None else None

                else:
                    _, new_logprob, entropy, new_value, _ = self.model.get_action_and_value(
                        b_obs[mb_idx],
                        graph_data=b_global_states[mb_idx] if b_global_states is not None else None,
                        action=b_actions[mb_idx]
                    )
                    mb_logprobs_old = b_logprobs[mb_idx]
                    mb_returns = b_returns[mb_idx]
                    mb_advantages = b_advantages[mb_idx]
                    mb_old_values = b_old_values[mb_idx] if b_old_values is not None else None

                with torch.no_grad():
                    approx_kl = (mb_logprobs_old - new_logprob).mean()
                    epoch_kls.append(approx_kl.item())

                # --- PPO LOSS CALCULATION ---
                # 1. Probability Ratio r(theta) = pi_new / pi_old
                logratio = new_logprob - mb_logprobs_old
                ratio = logratio.exp()

                # 2. Surrogate Objective
                # L_clip = min( r * A, clip(r, 1-eps, 1+eps) * A )
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.CLIP_COEF, 1 + self.cfg.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 3. Value Loss
                # L_vf = (V_pred - V_target)^2
                if mb_old_values is not None:
                    # Optional: Clip value updates for stability
                    v_loss_unclipped = (new_value.view(-1) - mb_returns) ** 2
                    v_clipped = mb_old_values + torch.clamp(
                        new_value.view(-1) - mb_old_values,
                        -self.cfg.CLIP_COEF,
                        self.cfg.CLIP_COEF
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value.view(-1) - mb_returns) ** 2).mean()

                # 4. Total Loss
                # L = L_clip - c1 * L_vf + c2 * Entropy
                # Note: We minimize loss, so we subtract entropy (maximize exploration)
                entropy_loss = entropy.mean()
                loss = pg_loss - (self.cfg.ENT_COEF * entropy_loss) + (self.cfg.VF_COEF * v_loss)

                self.optimizer.zero_grad()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
                    self.optimizer.step()

                epoch_losses.append(loss.item())
                epoch_pg_losses.append(pg_loss.item())
                epoch_v_losses.append(v_loss.item())
                epoch_entropies.append(entropy_loss.item())

        return {
            "loss": np.mean(epoch_losses),
            "policy_loss": np.mean(epoch_pg_losses),
            "value_loss": np.mean(epoch_v_losses),
            "entropy": np.mean(epoch_entropies),
            "approx_kl": np.mean(epoch_kls)
        }