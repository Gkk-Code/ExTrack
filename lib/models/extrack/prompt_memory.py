"""Input prompts and fixed-score memory for offline sequential domain learning."""

import torch
from torch import nn
import torch.distributed as dist


class PromptMemory(nn.Module):
    def __init__(self, feature_dim, prompt_dim=512, prompt_length=5,
                 memory_size=50, num_heads=8, smoothing_window=5):
        super().__init__()
        if feature_dim != prompt_dim:
            raise ValueError("ExTrack requires matching prompt and backbone dimensions")
        if prompt_dim % num_heads or min(prompt_length, memory_size) < 1:
            raise ValueError("Invalid prompt dimensions or memory capacity")
        self.prompt_dim = prompt_dim
        self.prompt_length = prompt_length
        self.capacity = memory_size
        self.smoothing_window = max(1, smoothing_window)
        self.appearance_proj = nn.Linear(feature_dim, prompt_dim)
        self.motion_proj = nn.Sequential(nn.Linear(feature_dim, prompt_dim), nn.GELU(),
                                         nn.Linear(prompt_dim, prompt_dim))
        layer = nn.TransformerEncoderLayer(prompt_dim, num_heads, batch_first=True,
                                           dropout=0.0, activation="gelu")
        self.prompt_fusion = nn.TransformerEncoder(layer, num_layers=1)
        self.history_attention = nn.MultiheadAttention(prompt_dim, num_heads,
                                                       batch_first=True, dropout=0.0)
        self.prompt_expand = nn.Linear(prompt_dim, prompt_length * prompt_dim)
        self.register_buffer("memory", torch.zeros(memory_size, prompt_length, prompt_dim))
        self.register_buffer("importance", torch.full((memory_size,), float("-inf")))
        self.register_buffer("memory_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("score_window", torch.zeros(self.smoothing_window))
        self.register_buffer("score_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("score_cursor", torch.zeros((), dtype=torch.long))
        self.memory_enabled = True
        self._last_prompt = None

    def forward(self, current, history=None, history_mask=None):
        if history is None or history.size(1) == 0:
            residual = torch.zeros_like(current)
        elif history_mask is None:
            residual = current - history.mean(dim=1)
        else:
            weights = history_mask.to(current.dtype).unsqueeze(-1)
            count = weights.sum(dim=1)
            previous = (history * weights).sum(dim=1) / count.clamp_min(1)
            residual = (current - previous) * (count > 0).to(current.dtype)
        appearance = self.appearance_proj(current)
        temporal = self.motion_proj(residual)
        fused = self.prompt_fusion(torch.stack((appearance, temporal), dim=1)).mean(dim=1)
        count = int(self.memory_count.item())
        if self.memory_enabled and count:
            history_tokens = self.memory[:count].clone().flatten(0, 1)
            history_tokens = history_tokens.to(fused.dtype).unsqueeze(0).expand(current.size(0), -1, -1)
            retrieved, _ = self.history_attention(fused.unsqueeze(1), history_tokens,
                                                   history_tokens, need_weights=False)
            fused = fused + retrieved.squeeze(1)
        prompt = self.prompt_expand(fused).view(-1, self.prompt_length, self.prompt_dim)
        self._last_prompt = None
        if self.training and torch.is_grad_enabled() and prompt.requires_grad:
            prompt.retain_grad()
            self._last_prompt = prompt
        return prompt

    @torch.no_grad()
    def reset_smoothing(self):
        """Reset task-local scoring, but retain historical prompts and their scores."""
        self.score_window.zero_()
        self.score_steps.zero_()
        self.score_cursor.zero_()

    @torch.no_grad()
    def update_memory(self, enabled=True, grad_scale=1.0, batch_size=1):
        prompt = self._last_prompt
        self._last_prompt = None
        if not enabled or not self.training or prompt is None or prompt.grad is None:
            return
        # Restore unscaled per-sample sensitivities.
        scores = prompt.grad.detach().float().norm(dim=-1).mean(dim=-1)
        scores = scores * (batch_size / grad_scale)
        candidates = prompt.detach().float()
        if dist.is_available() and dist.is_initialized():
            all_scores = [torch.empty_like(scores) for _ in range(dist.get_world_size())]
            all_prompts = [torch.empty_like(candidates) for _ in all_scores]
            dist.all_gather(all_scores, scores.contiguous())
            dist.all_gather(all_prompts, candidates.contiguous())
            scores, candidates = torch.cat(all_scores), torch.cat(all_prompts)
        if not torch.isfinite(scores).all() or not torch.isfinite(candidates).all():
            return
        # Smooth with the preceding K-1 global step means.
        steps = min(int(self.score_steps.item()), self.smoothing_window - 1)
        cursor = int(self.score_cursor.item())
        if steps:
            indices = [(cursor - i - 1) % self.smoothing_window for i in range(steps)]
            smoothed = (scores + self.score_window[indices].sum()) / (steps + 1)
        else:
            smoothed = scores
        self.score_window[cursor] = scores.mean()
        self.score_cursor.fill_((cursor + 1) % self.smoothing_window)
        self.score_steps.add_(1)
        count = int(self.memory_count.item())
        values = torch.cat((self.importance[:count], smoothed))
        prompts = torch.cat((self.memory[:count], candidates))
        keep = min(self.capacity, values.numel())
        indices = torch.argsort(values, descending=True, stable=True)[:keep]
        self.memory[:keep].copy_(prompts[indices])
        self.importance[:keep].copy_(values[indices])
        self.memory_count.fill_(keep)
