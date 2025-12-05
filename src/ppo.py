# ================================================
# FILE: src/ppo.py
# ================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import Config
from torch_geometric.data import Batch  # Needed for on-the-fly batching


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent implementation.
    UPDATED: Handles PyG Graph Batching manually to avoid Tensor conversion errors.
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE, eps=1e-5)

    def update(self, obs, actions, logprobs, returns, advantages, global_states=None, gru_states=None, dones=None,
               old_values=None, active_masks=None):
        """
        Update the policy and value networks.

        global_states: List of PyG Data objects (Graphs). NOT a Tensor.
        """

        self.model.train()
        SEQ_LEN = 32

        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.cfg.DEVICE)
            return torch.tensor(x, dtype=torch.float32).to(self.cfg.DEVICE)

        # Convert standard tensors
        b_obs = to_device(obs)
        b_actions = to_device(actions)
        b_logprobs = to_device(logprobs)
        b_returns = to_device(returns)
        b_advantages = to_device(advantages)
        b_dones = to_device(dones) if dones is not None else None
        b_old_values = to_device(old_values) if old_values is not None else None

        # Handle Graphs separately (Don't convert to Tensor)
        b_graphs = global_states  # List of Data objects

        b_gru_h = to_device(gru_states) if gru_states is not None else None
        b_active_masks = to_device(active_masks) if active_masks is not None else torch.ones_like(b_returns)

        batch_size = b_obs.shape[0]
        use_gru = (b_gru_h is not None)

        # --- SEQUENCE HANDLING ---
        if use_gru:
            num_seqs = batch_size // SEQ_LEN
            if batch_size % SEQ_LEN != 0:
                trunc_len = num_seqs * SEQ_LEN
                # Truncate tensors
                b_obs, b_actions = b_obs[:trunc_len], b_actions[:trunc_len]
                b_logprobs, b_returns = b_logprobs[:trunc_len], b_returns[:trunc_len]
                b_advantages, b_active_masks = b_advantages[:trunc_len], b_active_masks[:trunc_len]
                if b_dones is not None: b_dones = b_dones[:trunc_len]
                if b_old_values is not None: b_old_values = b_old_values[:trunc_len]
                b_gru_h = b_gru_h[:trunc_len]

                # Truncate Graphs List
                if b_graphs is not None:
                    b_graphs = b_graphs[:trunc_len]

                batch_size = trunc_len

            def make_seq(x):
                return x.reshape(num_seqs, SEQ_LEN, *x.shape[1:])

            s_obs = make_seq(b_obs)
            s_actions = make_seq(b_actions)
            s_logprobs = make_seq(b_logprobs)
            s_returns = make_seq(b_returns)
            s_advantages = make_seq(b_advantages)
            s_active_masks = make_seq(b_active_masks)
            s_dones = make_seq(b_dones) if b_dones is not None else None
            s_old_values = make_seq(b_old_values) if b_old_values is not None else None

            # Handle Graph Sequences (List of Lists)
            s_graphs = None
            if b_graphs is not None:
                # Chunk the flat list into sequences
                s_graphs = [b_graphs[i * SEQ_LEN: (i + 1) * SEQ_LEN] for i in range(num_seqs)]

            # Extract initial GRU states
            s_gru_h_init = b_gru_h.reshape(num_seqs, SEQ_LEN, *b_gru_h.shape[1:])[:, 0].unsqueeze(0)

            optim_batch_size = num_seqs
        else:
            optim_batch_size = batch_size

        indices = torch.randperm(optim_batch_size, device=self.cfg.DEVICE)
        epoch_stats = {k: [] for k in ["loss", "pg_loss", "v_loss", "entropy", "kl", "clip_frac"]}

        for _ in range(self.cfg.UPDATE_EPOCHS):
            step_size = self.cfg.MINIBATCH_SIZE
            if use_gru:
                step_size = max(1, self.cfg.MINIBATCH_SIZE // SEQ_LEN)

            for start in range(0, optim_batch_size, step_size):
                end = start + step_size
                mb_idx = indices[start:end]

                # Helper index list for Python lists
                mb_idx_list = mb_idx.tolist()

                if use_gru:
                    mb_obs = s_obs[mb_idx]
                    mb_actions = s_actions[mb_idx]
                    mb_dones = s_dones[mb_idx] if s_dones is not None else None
                    mb_gru = s_gru_h_init[:, mb_idx, :]

                    # Batch Graphs for this sequence chunk
                    mb_global = None
                    if s_graphs is not None:
                        # Gather sequences of graphs
                        nested_graphs = [s_graphs[i] for i in mb_idx_list]
                        # Flatten to (MiniBatch * SeqLen) for Batch()
                        flat_mb_graphs = [g for seq in nested_graphs for g in seq]
                        mb_global = Batch.from_data_list(flat_mb_graphs).to(self.cfg.DEVICE)

                    # Forward Pass
                    _, new_logprob, entropy, new_value, _ = self.model.get_action_and_value(
                        mb_obs, graph_data=mb_global, action=mb_actions, gru_state=mb_gru, done=mb_dones
                    )

                    # Flatten back to match buffers
                    new_logprob, entropy, new_value = new_logprob.flatten(), entropy.flatten(), new_value.flatten()

                    mb_logprobs_old = s_logprobs[mb_idx].flatten()
                    mb_returns = s_returns[mb_idx].flatten()
                    mb_advantages = s_advantages[mb_idx].flatten()
                    mb_old_values = s_old_values[mb_idx].flatten() if s_old_values is not None else None
                    mb_active = s_active_masks[mb_idx].flatten()

                else:
                    # Non-GRU
                    mb_global = None
                    if b_graphs is not None:
                        # Gather graphs for this minibatch
                        mb_graphs_list = [b_graphs[i] for i in mb_idx_list]
                        mb_global = Batch.from_data_list(mb_graphs_list).to(self.cfg.DEVICE)

                    _, new_logprob, entropy, new_value, _ = self.model.get_action_and_value(
                        b_obs[mb_idx],
                        graph_data=mb_global,
                        action=b_actions[mb_idx]
                    )
                    mb_logprobs_old = b_logprobs[mb_idx]
                    mb_returns = b_returns[mb_idx]
                    mb_advantages = b_advantages[mb_idx]
                    mb_old_values = b_old_values[mb_idx] if b_old_values is not None else None
                    mb_active = b_active_masks[mb_idx]

                # --- PPO LOSS ---
                logratio = new_logprob - mb_logprobs_old
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    epoch_stats["kl"].append(approx_kl.item())
                    
                    # Calculate clip fraction: how many samples were clipped
                    clipped = (ratio.lt(1 - self.cfg.CLIP_COEF) | ratio.gt(1 + self.cfg.CLIP_COEF)).float()
                    clip_frac = (clipped * mb_active).sum() / (mb_active.sum() + 1e-8)
                    epoch_stats["clip_frac"].append(clip_frac.item())

                # 1. Policy Loss (Masked)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.CLIP_COEF, 1 + self.cfg.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2)

                pg_loss = (pg_loss * mb_active).sum() / (mb_active.sum() + 1e-8)

                # 2. Value Loss (Masked)
                if mb_old_values is not None:
                    v_loss_unclipped = (new_value.view(-1) - mb_returns) ** 2
                    v_clipped = mb_old_values + torch.clamp(
                        new_value.view(-1) - mb_old_values, -self.cfg.CLIP_COEF, self.cfg.CLIP_COEF
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_elem = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                else:
                    v_loss_elem = 0.5 * ((new_value.view(-1) - mb_returns) ** 2)

                v_loss = (v_loss_elem * mb_active).sum() / (mb_active.sum() + 1e-8)

                # 3. Entropy (Masked)
                entropy_loss = (entropy * mb_active).sum() / (mb_active.sum() + 1e-8)

                loss = pg_loss - (self.cfg.ENT_COEF * entropy_loss) + (self.cfg.VF_COEF * v_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
                self.optimizer.step()

                epoch_stats["loss"].append(loss.item())
                epoch_stats["pg_loss"].append(pg_loss.item())
                epoch_stats["v_loss"].append(v_loss.item())
                epoch_stats["entropy"].append(entropy_loss.item())

        # Calculate explained variance on the whole batch
        with torch.no_grad():
            y_pred = b_old_values
            y_true = b_returns
            var_y = torch.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y

        return {
            "loss": np.mean(epoch_stats["loss"]),
            "pg_loss": np.mean(epoch_stats["pg_loss"]),
            "v_loss": np.mean(epoch_stats["v_loss"]),
            "entropy": np.mean(epoch_stats["entropy"]),
            "kl": np.mean(epoch_stats["kl"]),
            "clip_frac": np.mean(epoch_stats["clip_frac"]),
            "explained_var": explained_var.item()
        }