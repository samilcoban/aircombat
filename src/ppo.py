import torch
import torch.nn as nn
import torch.optim as optim
from config import Config


class PPOAgent:
    def __init__(self, model):
        self.cfg = Config
        self.model = model.to(self.cfg.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LR, eps=1e-5)

    def update(self, obs, actions, logprobs, returns, advantages, scaler=None):
        # --- OPTIMIZATION: Keep Tensors on GPU ---
        # If inputs are already tensors, just ensure device match.
        # If inputs are numpy (legacy/cpu env), convert them.

        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.cfg.DEVICE)
            return torch.tensor(x, dtype=torch.float32).to(self.cfg.DEVICE)

        b_obs = to_device(obs)
        b_actions = to_device(actions)
        b_logprobs = to_device(logprobs)
        b_returns = to_device(returns)
        b_advantages = to_device(advantages)

        # Dynamic Batch Size
        batch_size = b_obs.shape[0]
        indices = torch.randperm(batch_size, device=self.cfg.DEVICE)

        loss_val = 0.0

        for _ in range(self.cfg.UPDATE_EPOCHS):
            for start in range(0, batch_size, self.cfg.MINIBATCH_SIZE):
                end = start + self.cfg.MINIBATCH_SIZE
                mb_idx = indices[start:end]

                # Forward Pass (Autocast handled externally or here?)
                # It's safer to rely on the external autocast context if provided, 
                # but since we are inside the loop, we should probably ensure autocast is active if scaler is present.
                # However, the user code in train.py wraps the whole update call in autocast.
                # So the forward pass here will be autocasted.
                
                _, new_logprob, entropy, new_value = self.model.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])

                # Advantages
                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_idx]
                # Normalize advantages (batch level normalization)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.cfg.CLIP_EPS, 1 + self.cfg.CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((new_value.view(-1) - b_returns[mb_idx]) ** 2).mean()

                # Total Loss
                loss = pg_loss - (self.cfg.ENT_COEF * entropy.mean()) + (self.cfg.VF_COEF * v_loss)

                # Optimize
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