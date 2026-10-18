"""Equations (2)-(6) of the accompanying draft. No future covariate input."""
import math
import torch
from torch import nn

QUANTILES = tuple(i / 10 for i in range(1, 10))
HORIZONS = (6, 12, 24, 48)


class WeatherAdapter(nn.Module):
    def __init__(self, n_features=8, hidden=32, target_scale=1.0,
                 uniform_pool=False, no_horizon=False, no_state_query=False,
                 location_only=False):
        super().__init__()
        if target_scale <= 0:
            raise ValueError('Training target scale must be positive.')
        self.encoder = nn.Linear(n_features + 1, hidden)
        self.query = nn.Linear(n_features + 1, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + n_features + 1, hidden),
                                  nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('target_scale', torch.tensor(float(target_scale)))
        self.register_buffer('horizons', torch.tensor(HORIZONS, dtype=torch.float32))
        self.uniform_pool = uniform_pool
        self.no_horizon = no_horizon
        self.no_state_query = no_state_query
        self.location_only = location_only

    def forward(self, history, base_quantiles):
        # history: batch, context (oldest first), features; q: batch, horizon, quantile.
        if history.ndim != 3 or history.shape[1] < 2:
            raise ValueError('history must be [batch, context >= 2, features]')
        b, length, _ = history.shape
        if tuple(base_quantiles.shape) != (b, len(HORIZONS), len(QUANTILES)):
            raise ValueError('base_quantiles must be [batch, 4, 9]')
        if not torch.isfinite(history).all() or not torch.isfinite(base_quantiles).all():
            raise ValueError('Missing or nonfinite inputs must be handled before inference.')
        lag = torch.linspace(1, 0, length, device=history.device,
                             dtype=history.dtype).view(1, length, 1).expand(b, -1, -1)
        encoded = torch.tanh(self.encoder(torch.cat([history, lag], dim=-1)))
        state = history[:, -1, :].unsqueeze(1).expand(-1, len(HORIZONS), -1)
        lead = (self.horizons / 48).view(1, -1, 1).expand(b, -1, -1)
        if self.no_horizon:
            lead = torch.zeros_like(lead)
        query_state = torch.zeros_like(state) if self.no_state_query else state
        query = torch.tanh(self.query(torch.cat([query_state, lead], dim=-1)))
        logits = torch.einsum('bhd,bld->bhl', query, encoded) / math.sqrt(encoded.shape[-1])
        weights = (torch.ones_like(logits) / length if self.uniform_pool
                   else torch.softmax(logits, dim=-1))
        pooled = torch.einsum('bhl,bld->bhd', weights, encoded)
        a, delta, scale = self.head(torch.cat([pooled, state, lead], dim=-1)).unbind(-1)
        gate = torch.sigmoid(a)
        kappa = torch.ones_like(scale) if self.location_only else torch.exp(gate * torch.tanh(scale))
        base = torch.sort(base_quantiles, dim=-1).values
        median = base[..., 4]
        prediction = (median[..., None] + (gate * delta * self.target_scale)[..., None]
                      + kappa[..., None] * (base - median[..., None])).clamp_min(0)
        return prediction, {'lag_weights': weights, 'gate': gate, 'scale': kappa}


def pinball(prediction, target):
    if target.shape != prediction.shape[:-1]:
        raise ValueError('Target shape must equal prediction batch/horizon dimensions.')
    levels = prediction.new_tensor(QUANTILES)
    residual = target[..., None] - prediction
    return torch.maximum(levels * residual, (levels - 1) * residual).mean()
