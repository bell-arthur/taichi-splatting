from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dino2GaussMLP(nn.Module):
  """Minimal MLP mapping per-cell DINO features (+ coords) to 2D Gaussian params.

  Inputs per Gaussian:
    - feature vector of size C
    - normalized coords (iy = i/Hf, ix = j/Wf)

  Outputs per Gaussian:
    - pos_offset: (dx, dy) in pixels, limited by offset_max after tanh
    - log_scaling: (sx, sy)
    - rotation: (rx, ry) normalized to unit length
    - alpha_logit: (1)
    - color: (r,g,b) in [0,1]
  """

  def __init__(self, in_dim: int, hidden_layers: tuple[int, ...] = (128, 128), offset_max: float = 4.0, k: int = 1):
    super().__init__()
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_layers:
      layers.append(nn.Linear(last, h))
      layers.append(nn.ReLU(inplace=True))
      last = h
    self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

    # Heads
    self.k = int(k)
    self.head_pos = nn.Linear(last, 2 * self.k)
    self.head_log_scale = nn.Linear(last, 2 * self.k)
    self.head_rot = nn.Linear(last, 2 * self.k)
    self.head_alpha = nn.Linear(last, 1 * self.k)
    self.head_color = nn.Linear(last, 3 * self.k)

    self.offset_max = float(offset_max)

  def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    # x: (N, in_dim)
    h = self.trunk(x)
    N = x.shape[0]

    pos_offset = torch.tanh(self.head_pos(h)).view(N, self.k, 2) * self.offset_max
    log_scaling = torch.clamp(self.head_log_scale(h), min=-5.0, max=5.0).view(N, self.k, 2)
    rot_raw = self.head_rot(h).view(N, self.k, 2)
    rotation = F.normalize(rot_raw, dim=-1, eps=1e-6)
    alpha_logit = self.head_alpha(h).view(N, self.k, 1)
    color = torch.sigmoid(self.head_color(h)).view(N, self.k, 3)

    return dict(
      pos_offset=pos_offset,        # (N,K,2)
      log_scaling=log_scaling,      # (N,K,2)
      rotation=rotation,            # (N,K,2)
      alpha_logit=alpha_logit,      # (N,K,1)
      color=color,                  # (N,K,3)
    )


