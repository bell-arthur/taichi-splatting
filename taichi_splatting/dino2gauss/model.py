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



class Dino2GaussConv(nn.Module):
  """Minimal fully-convolutional head mapping feature grid (+coords) to 2D Gaussian params.

  Inputs:
    - features_hwc: (Hf, Wf, C) feature grid
    - internally constructs 2 coord channels (gx, gy) in [0,1] and concatenates → C+2

  Outputs per anchor cell (after stride subsample):
    - pos_offset: (N, K, 2) in pixels, limited by offset_max after tanh
    - log_scaling: (N, K, 2) clamped to [-5, 5]
    - rotation: (N, K, 2) normalized vectors
    - alpha_logit: (N, K, 1)
    - color: (N, K, 3) in [0,1]
  """

  def __init__(self, in_channels: int, hidden: int = 128, offset_max: float = 4.0, k: int = 1):
    super().__init__()
    self.k = int(k)
    self.offset_max = float(offset_max)

    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=True),
      nn.ReLU(inplace=True),
    )

    # 1x1 heads
    self.head_pos = nn.Conv2d(hidden, 2 * self.k, kernel_size=1)
    self.head_log_scale = nn.Conv2d(hidden, 2 * self.k, kernel_size=1)
    self.head_rot = nn.Conv2d(hidden, 2 * self.k, kernel_size=1)
    self.head_alpha = nn.Conv2d(hidden, 1 * self.k, kernel_size=1)
    self.head_color = nn.Conv2d(hidden, 3 * self.k, kernel_size=1)

  def forward(self, features_hwc: torch.Tensor, stride: int = 1) -> dict[str, torch.Tensor]:
    # features_hwc: (Hf, Wf, C)
    Hf, Wf, C = features_hwc.shape
    feats = features_hwc.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,C,Hf,Wf)

    # Coord channels in [0,1]
    gy = torch.linspace(0, 1, Hf, device=feats.device, dtype=feats.dtype).view(1, 1, Hf, 1).expand(1, 1, Hf, Wf)
    gx = torch.linspace(0, 1, Wf, device=feats.device, dtype=feats.dtype).view(1, 1, 1, Wf).expand(1, 1, Hf, Wf)
    x = torch.cat([feats, gx, gy], dim=1)  # (1, C+2, Hf, Wf)

    h = self.stem(x)

    pos = torch.tanh(self.head_pos(h)) * self.offset_max         # (1,2K,Hf,Wf)
    log_scale = torch.clamp(self.head_log_scale(h), -5.0, 5.0)   # (1,2K,Hf,Wf)
    rot_raw = self.head_rot(h)                                   # (1,2K,Hf,Wf)
    alpha = self.head_alpha(h)                                   # (1,1K,Hf,Wf)
    color = torch.sigmoid(self.head_color(h))                    # (1,3K,Hf,Wf)

    # Subsample by stride and reshape to (N, K, ch)
    def to_nk(t: torch.Tensor, ch_per: int) -> torch.Tensor:
      t = t[..., ::stride, ::stride]  # (1, ch, Hs, Ws)
      _, ch, Hs, Ws = t.shape
      t = t.permute(0, 2, 3, 1).reshape(Hs * Ws, self.k, ch_per)
      return t

    pos_nk = to_nk(pos, 2)
    log_scale_nk = to_nk(log_scale, 2)
    rot_nk = to_nk(rot_raw, 2)
    rot_nk = F.normalize(rot_nk, dim=-1, eps=1e-6)
    alpha_nk = to_nk(alpha, 1)
    color_nk = to_nk(color, 3)

    return dict(
      pos_offset=pos_nk,        # (N,K,2)
      log_scaling=log_scale_nk, # (N,K,2)
      rotation=rot_nk,          # (N,K,2)
      alpha_logit=alpha_nk,     # (N,K,1)
      color=color_nk,           # (N,K,3)
    )

