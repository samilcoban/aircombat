# ================================================
# FILE: src/ppo.py
# ================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from config import Config
from torch_geometric.data import Batch


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent implementation.
    Includes robust NaN guards, Sequence handling, and KL Early Stopping.
    """

    def __init__(self, model):
        self.cfg = Config
        self.model = model.to(self.cfg.DEVICE)

        # Optimization: Compile Model
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("✅ PyTorch 2.0 Compilation Enabled")
        except Exception as e:
            print(f"⚠️ PyTorch Compile Skipped: {e}")

        # Weight decay prevents parameter explosion during long training
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE, eps=1e-5, weight_decay=1e-4)
        self.seq_len = self.cfg.SEQ_LEN

    def update(self, obs, next_obs, actions, logprobs, returns, advantages, global_states=None, gru_states=None,
               dones=None, old_values=None, active_masks=None, update_actor=True):
        """
        Update the policy, value networks, and auxiliary world model.
        Args:
            update_actor (bool): If False, freezes the Actor (Policy) updates. Used for Critic warmup.
        """

        self.model.train()

        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.cfg.DEVICE)
            return torch.tensor(x, dtype=torch.float32).to(self.cfg.DEVICE)

        # Convert standard tensors
        b_obs = to_device(obs)
        b_next_obs = to_device(next_obs)  # [Target for World Model]
        b_actions = to_device(actions)
        b_logprobs = to_device(logprobs)
        b_returns = to_device(returns)
        b_advantages = to_device(advantages)
        b_dones = to_device(dones) if dones is not None else None
        b_old_values = to_device(old_values) if old_values is not None else None
        b_graphs = global_states
        b_gru_h = to_device(gru_states) if gru_states is not None else None
        b_active_masks = to_device(active_masks) if active_masks is not None else torch.ones_like(b_returns)

        batch_size = b_obs.shape[0]
        use_gru = (b_gru_h is not None)

        # --- SEQUENCE HANDLING ---
        # If using GRU, we must reshape flat buffers into (Batch, Seq_Len, Dim)
        if use_gru:
            num_seqs = batch_size // self.seq_len

            # Truncate if batch_size is not perfectly divisible
            if batch_size % self.seq_len != 0:
                trunc_len = num_seqs * self.seq_len
                b_obs = b_obs[:trunc_len]
                b_next_obs = b_next_obs[:trunc_len]
                b_actions = b_actions[:trunc_len]
                b_logprobs = b_logprobs[:trunc_len]
                b_returns = b_returns[:trunc_len]
                b_advantages = b_advantages[:trunc_len]
                b_active_masks = b_active_masks[:trunc_len]

                if b_dones is not None: b_dones = b_dones[:trunc_len]
                if b_old_values is not None: b_old_values = b_old_values[:trunc_len]
                b_gru_h = b_gru_h[:trunc_len]
                if b_graphs is not None: b_graphs = b_graphs[:trunc_len]

            def make_seq(x):
                return x.reshape(num_seqs, self.seq_len, *x.shape[1:])

            s_obs = make_seq(b_obs)
            s_next_obs = make_seq(b_next_obs)
            s_actions = make_seq(b_actions)
            s_logprobs = make_seq(b_logprobs)
            s_returns = make_seq(b_returns)
            s_advantages = make_seq(b_advantages)
            s_active_masks = make_seq(b_active_masks)
            s_dones = make_seq(b_dones) if b_dones is not None else None
            s_old_values = make_seq(b_old_values) if b_old_values is not None else None

            # Chunk graphs (List of lists)
            s_graphs = None
            if b_graphs is not None:
                s_graphs = [b_graphs[i * self.seq_len: (i + 1) * self.seq_len] for i in range(num_seqs)]

            # Extract initial GRU states for each Sequence (Time=0)
            # b_gru_h is (TotalSteps, 1, Dim). Reshape to (NumSeqs, SeqLen, 1, Dim) -> Take idx 0
            s_gru_h_init = b_gru_h.reshape(num_seqs, self.seq_len, *b_gru_h.shape[1:])[:, 0].unsqueeze(0)

            optim_batch_size = num_seqs
        else:
            # Fallback for non-recurrent
            optim_batch_size = batch_size

        indices = torch.randperm(optim_batch_size, device=self.cfg.DEVICE)
        epoch_stats = {k: [] for k in ["loss", "pg_loss", "v_loss", "aux_loss", "entropy", "kl", "clip_frac"]}

        # --- UPDATE EPOCHS ---
        target_kl = getattr(self.cfg, 'TARGET_KL', 0.02)

        for epoch_i in range(self.cfg.UPDATE_EPOCHS):

            # --- KL EARLY STOPPING ---
            # If the policy has changed too much in this epoch, stop immediately.
            # This prevents catastrophic forgetting.
            if len(epoch_stats["kl"]) > 0 and np.mean(epoch_stats["kl"][-5:]) > target_kl * 1.5:
                # Optional: print(f"🛑 Early stopping at epoch {epoch_i}")
                break

            step_size = self.cfg.MINIBATCH_SIZE
            if use_gru:
                step_size = max(1, self.cfg.MINIBATCH_SIZE // self.seq_len)

            for start in range(0, optim_batch_size, step_size):
                end = start + step_size
                mb_idx = indices[start:end]
                mb_idx_list = mb_idx.tolist()

                # Prepare Minibatch Data
                if use_gru:
                    # Sequence Slicing
                    mb_obs = s_obs[mb_idx]
                    mb_next_obs = s_next_obs[mb_idx]
                    mb_actions = s_actions[mb_idx]
                    mb_dones = s_dones[mb_idx] if s_dones is not None else None
                    mb_gru = s_gru_h_init[:, mb_idx, :]

                    # Graph Batching
                    mb_global = None
                    if s_graphs is not None:
                        nested_graphs = [s_graphs[i] for i in mb_idx_list]
                        flat_mb_graphs = [g for seq in nested_graphs for g in seq]
                        mb_global = Batch.from_data_list(flat_mb_graphs).to(self.cfg.DEVICE)

                    # Forward Pass (Recurrent)
                    _, new_logprob, entropy, new_value, _ = self.model.get_action_and_value(
                        mb_obs,
                        graph_data=mb_global,
                        action=mb_actions,
                        gru_state=mb_gru,
                        done=mb_dones
                    )

                    # Flatten outputs to match flattened targets
                    new_logprob = new_logprob.flatten()
                    entropy = entropy.flatten()
                    new_value = new_value.flatten()

                    # Aux World Model Pass
                    actor_features, _ = self.model.extract_actor_features(
                        mb_obs, gru_state=mb_gru, done=mb_dones
                    )
                    flat_actions = mb_actions.flatten(0, 1)
                    pred_next_state, pred_reward = self.model.get_aux_prediction(actor_features, flat_actions)

                    # Flatten Targets
                    mb_logprobs_old = s_logprobs[mb_idx].flatten()
                    mb_returns = s_returns[mb_idx].flatten()
                    mb_advantages = s_advantages[mb_idx].flatten()
                    mb_old_values = s_old_values[mb_idx].flatten() if s_old_values is not None else None
                    mb_active = s_active_masks[mb_idx].flatten()

                    flat_next_obs = mb_next_obs.flatten(0, 1)
                    target_next_state = flat_next_obs[:, :self.cfg.NODE_DIM]

                else:
                    # Standard (Non-Recurrent) Slicing
                    # Restored this block which was missing in previous snippet
                    mb_obs = b_obs[mb_idx]
                    mb_next_obs = b_next_obs[mb_idx]
                    mb_actions = b_actions[mb_idx]
                    mb_dones = b_dones[mb_idx] if b_dones is not None else None

                    mb_global = None
                    if b_graphs is not None:
                        # b_graphs is a list of Data objects
                        flat_mb_graphs = [b_graphs[i] for i in mb_idx_list]
                        mb_global = Batch.from_data_list(flat_mb_graphs).to(self.cfg.DEVICE)

                    # Forward Pass
                    _, new_logprob, entropy, new_value, _ = self.model.get_action_and_value(
                        mb_obs,
                        graph_data=mb_global,
                        action=mb_actions,
                        gru_state=None,
                        done=mb_dones
                    )

                    # Aux World Model
                    actor_features, _ = self.model.extract_actor_features(mb_obs, gru_state=None, done=mb_dones)
                    pred_next_state, pred_reward = self.model.get_aux_prediction(actor_features, mb_actions)

                    mb_logprobs_old = b_logprobs[mb_idx]
                    mb_returns = b_returns[mb_idx]
                    mb_advantages = b_advantages[mb_idx]
                    mb_old_values = b_old_values[mb_idx] if b_old_values is not None else None
                    mb_active = b_active_masks[mb_idx]
                    target_next_state = mb_next_obs[:, :self.cfg.NODE_DIM]

                # --- LOSS CALCULATION ---

                # 1. PPO Policy Loss (Conditional Freeze)
                if update_actor:
                    logratio = new_logprob - mb_logprobs_old
                    logratio = torch.clamp(logratio, -10, 10)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # Calculate KL
                        kl_raw = (ratio - 1) - logratio
                        masked_kl = kl_raw * mb_active
                        approx_kl = masked_kl.sum() / (mb_active.sum() + 1e-8)
                        epoch_stats["kl"].append(approx_kl.item())

                        # Clipping stats
                        clipped = (ratio.lt(1 - self.cfg.CLIP_COEF) | ratio.gt(1 + self.cfg.CLIP_COEF)).float()
                        clip_frac = (clipped * mb_active).sum() / (mb_active.sum() + 1e-8)
                        epoch_stats["clip_frac"].append(clip_frac.item())

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.CLIP_COEF, 1 + self.cfg.CLIP_COEF)
                    pg_loss = torch.max(pg_loss1, pg_loss2)
                    pg_loss = (pg_loss * mb_active).sum() / (mb_active.sum() + 1e-8)

                    entropy_loss = (entropy * mb_active).sum() / (mb_active.sum() + 1e-8)
                else:
                    # Frozen Actor: Zero loss, Dummy stats
                    pg_loss = torch.tensor(0.0, device=self.cfg.DEVICE)
                    entropy_loss = torch.tensor(0.0, device=self.cfg.DEVICE)
                    epoch_stats["kl"].append(0.0)
                    epoch_stats["clip_frac"].append(0.0)

                # 2. Value Loss
                if mb_old_values is not None:
                    v_loss_unclipped = (new_value - mb_returns) ** 2
                    v_clipped = mb_old_values + torch.clamp(
                        new_value - mb_old_values, -self.cfg.CLIP_COEF, self.cfg.CLIP_COEF
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_elem = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                else:
                    v_loss_elem = 0.5 * ((new_value - mb_returns) ** 2)
                v_loss = (v_loss_elem * mb_active).sum() / (mb_active.sum() + 1e-8)

                # 3. Auxiliary Loss (World Model)
                aux_loss_elem = F.mse_loss(pred_next_state, target_next_state, reduction='none').mean(dim=-1)
                aux_loss_elem = torch.clamp(aux_loss_elem, 0, 10.0)
                aux_loss = (aux_loss_elem * mb_active).sum() / (mb_active.sum() + 1e-8)

                # TOTAL LOSS
                loss = pg_loss - (self.cfg.ENT_COEF * entropy_loss) + \
                       (self.cfg.VF_COEF * v_loss) + \
                       (getattr(self.cfg, 'AUX_COEF', 0.2) * aux_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)

                # NaN Check
                valid_gradients = True
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            break

                if valid_gradients:
                    self.optimizer.step()

                epoch_stats["loss"].append(loss.item())
                epoch_stats["pg_loss"].append(pg_loss.item())
                epoch_stats["v_loss"].append(v_loss.item())
                epoch_stats["aux_loss"].append(aux_loss.item())
                epoch_stats["entropy"].append(entropy_loss.item())

        with torch.no_grad():
            y_pred = b_old_values
            y_true = b_returns
            var_y = torch.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y

        return {
            "loss": np.mean(epoch_stats["loss"]),
            "pg_loss": np.mean(epoch_stats["pg_loss"]),
            "v_loss": np.mean(epoch_stats["v_loss"]),
            "aux_loss": np.mean(epoch_stats["aux_loss"]),
            "entropy": np.mean(epoch_stats["entropy"]),
            "kl": np.mean(epoch_stats["kl"]),
            "clip_frac": np.mean(epoch_stats["clip_frac"]),
            "explained_var": explained_var.item()
        }