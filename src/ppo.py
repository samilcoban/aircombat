# ================================================
# FILE: src/ppo.py
# ================================================
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent implementation.
    """

    def __init__(self, model):
        self.cfg = Config
        self.model = model.to(self.cfg.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE, eps=1e-5)

    def update(self, obs, actions, logprobs, returns, advantages, global_states=None, lstm_states=None, dones=None,
               old_values=None, scaler=None):
        
        self.model.train()
        # Sequence Length for BPTT
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

        # Handle LSTM States
        b_lstm_h = None
        b_lstm_c = None
        if lstm_states is not None:
            b_lstm_h = to_device(lstm_states[0])
            b_lstm_c = to_device(lstm_states[1])

        # === SEQUENCE RESHAPING ===
        batch_size = b_obs.shape[0]
        use_lstm = (b_lstm_h is not None)

        if use_lstm:
            num_seqs = batch_size // SEQ_LEN
            if batch_size % SEQ_LEN != 0:
                # Truncate to multiple of SEQ_LEN
                trunc_len = num_seqs * SEQ_LEN
                b_obs = b_obs[:trunc_len]
                b_actions = b_actions[:trunc_len]
                b_logprobs = b_logprobs[:trunc_len]
                b_returns = b_returns[:trunc_len]
                b_advantages = b_advantages[:trunc_len]
                if b_dones is not None: b_dones = b_dones[:trunc_len]
                if b_global_states is not None: b_global_states = b_global_states[:trunc_len]
                if b_old_values is not None: b_old_values = b_old_values[:trunc_len]
                b_lstm_h = b_lstm_h[:trunc_len]
                b_lstm_c = b_lstm_c[:trunc_len]
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

            # --- FIX FOR LSTM DIMS ---
            # b_lstm_h shape is (Batch, Layers=1, Hidden)
            # Reshape to (NumSeqs, SeqLen, Layers, Hidden)
            # Take first timestep: [:, 0] -> (NumSeqs, Layers, Hidden)
            s_lstm_h_init = b_lstm_h.reshape(num_seqs, SEQ_LEN, *b_lstm_h.shape[1:])[:, 0]
            s_lstm_c_init = b_lstm_c.reshape(num_seqs, SEQ_LEN, *b_lstm_c.shape[1:])[:, 0]

            # The LSTM expects (Layers, Batch, Hidden).
            # Currently we have (Batch, Layers, Hidden).
            # Transpose dim 0 and 1 -> (Layers, NumSeqs, Hidden)
            s_lstm_h_init = s_lstm_h_init.transpose(0, 1)
            s_lstm_c_init = s_lstm_c_init.transpose(0, 1)

            optim_batch_size = num_seqs
        else:
            optim_batch_size = batch_size

        indices = torch.randperm(optim_batch_size, device=self.cfg.DEVICE)
        loss_val = 0.0

        # === ADVANTAGE NORMALIZATION ===
        if use_lstm:
            flat_adv = s_advantages.flatten()
            mean_adv = flat_adv.mean()
            std_adv = flat_adv.std()
            s_advantages = (s_advantages - mean_adv) / (std_adv + 1e-8)
        else:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # === PPO EPOCHS ===
        for _ in range(self.cfg.UPDATE_EPOCHS):
            # Stride logic for LSTM
            step_size = self.cfg.MINIBATCH_SIZE
            if use_lstm:
                step_size = max(1, self.cfg.MINIBATCH_SIZE // SEQ_LEN)

            for start in range(0, optim_batch_size, step_size):
                end = start + step_size
                mb_idx = indices[start:end]

                if use_lstm:
                    mb_obs = s_obs[mb_idx]
                    mb_actions = s_actions[mb_idx]
                    mb_global = s_global_states[mb_idx] if s_global_states is not None else None
                    mb_dones = s_dones[mb_idx] if s_dones is not None else None

                    # LSTM States: (Layers, Batch, Hidden)
                    # s_lstm_h_init is (Layers, NumSeqs, Hidden)
                    # mb_idx indexes the NumSeqs dimension (dim 1)
                    mb_h = s_lstm_h_init[:, mb_idx, :]
                    mb_c = s_lstm_c_init[:, mb_idx, :]
                    mb_lstm_state = (mb_h, mb_c)

                    _, new_logprob, entropy, new_value, _ = self.model.get_action_and_value(
                        mb_obs,
                        global_state=mb_global,
                        action=mb_actions,
                        lstm_state=mb_lstm_state,
                        done=mb_dones
                    )

                    # Flatten sequence outputs
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
                        global_state=b_global_states[mb_idx] if b_global_states is not None else None,
                        action=b_actions[mb_idx]
                    )
                    mb_logprobs_old = b_logprobs[mb_idx]
                    mb_returns = b_returns[mb_idx]
                    mb_advantages = b_advantages[mb_idx]
                    mb_old_values = b_old_values[mb_idx] if b_old_values is not None else None

                # Policy Loss
                logratio = new_logprob - mb_logprobs_old
                ratio = logratio.exp()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.CLIP_COEF, 1 + self.cfg.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                if mb_old_values is not None:
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

                loss = pg_loss - (self.cfg.ENT_COEF * entropy.mean()) + (self.cfg.VF_COEF * v_loss)

                # Update
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

                loss_val = loss.item()

        return loss_val