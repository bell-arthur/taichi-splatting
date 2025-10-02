from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from taichi_splatting.data_types import Gaussians2D


def build_anchors(H: int, W: int, Hf: int, Wf: int, stride: int = 1, device: torch.device | str = 'cuda') -> torch.Tensor:
  """Create anchor pixel centers for a Hf×Wf feature grid over an H×W image.
  Returns (N,2) xy pixel positions, N ~ (Hf/stride)*(Wf/stride).
  """
  ys = torch.arange(0, Hf, stride, device=device)
  xs = torch.arange(0, Wf, stride, device=device)
  grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
  r_h = H / float(Hf)
  r_w = W / float(Wf)
  px = (grid_x.to(torch.float32) + 0.5) * r_w
  py = (grid_y.to(torch.float32) + 0.5) * r_h
  anchors = torch.stack([px, py], dim=-1).reshape(-1, 2)
  return anchors


def gather_latents(features_hwc: torch.Tensor, stride: int = 1) -> torch.Tensor:
  """Gather per-cell feature vectors according to stride.
  features_hwc: (Hf, Wf, C) -> (N, C)
  """
  Hf, Wf, C = features_hwc.shape
  feats = features_hwc[::stride, ::stride, :].contiguous().view(-1, C)
  return feats


def pack_gaussians(
    anchors_xy: torch.Tensor,  # (N,2) pixel centers
    preds: Dict[str, torch.Tensor],  # outputs of model, possibly with K>1
    image_size: Tuple[int, int],
    default_depth: float = 0.5,
  ) -> Gaussians2D:
  device = anchors_xy.device
  N = anchors_xy.shape[0]

  # Support K>1 by tiling anchors and flattening
  if preds['pos_offset'].dim() == 3:
    N, K, _ = preds['pos_offset'].shape
    anchors_tiled = anchors_xy.unsqueeze(1).expand(N, K, 2)
    position = anchors_tiled + preds['pos_offset']            # (N,K,2)
    log_scaling = preds['log_scaling']                         # (N,K,2)
    rotation = preds['rotation']                               # (N,K,2)
    alpha_logit = preds['alpha_logit']                         # (N,K,1)
    feature = preds['color']                                   # (N,K,3)

    position = position.reshape(N * K, 2)
    log_scaling = log_scaling.reshape(N * K, 2)
    rotation = rotation.reshape(N * K, 2)
    alpha_logit = alpha_logit.reshape(N * K, 1)
    feature = feature.reshape(N * K, 3)
    N = N * K
  else:
    position = anchors_xy + preds['pos_offset']                # (N,2)
    log_scaling = preds['log_scaling']
    rotation = preds['rotation']
    alpha_logit = preds['alpha_logit']
    feature = preds['color']

  depths = torch.full((N, 1), float(default_depth), device=device, dtype=position.dtype)
  latent = torch.zeros((N, 1), device=device, dtype=position.dtype)

  return Gaussians2D(
    position=position,
    depths=depths,
    log_scaling=log_scaling,
    rotation=rotation,
    alpha_logit=alpha_logit,
    feature=feature,
    latent=latent,
    batch_size=(N,),
  )


def pack_gaussians_tensor(g2d: Gaussians2D) -> torch.Tensor:
  """Pack Gaussians2D fields into (N,7) tensor expected by rasterizer.
  Order: [position(2), log_scaling(2), rotation(2), alpha_logit(1)]
  """
  return torch.cat([g2d.position, g2d.log_scaling, g2d.rotation, g2d.alpha_logit], dim=-1)


