from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch

from taichi_splatting.dino2gauss.model import Dino2GaussMLP, Dino2GaussConv
from taichi_splatting.dino2gauss.utils import build_anchors, gather_latents, pack_gaussians


def load_cache_item(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
  data: Dict[str, Any] = torch.load(path, map_location='cpu')
  features: torch.Tensor = data['features']
  target: torch.Tensor = data['target']
  meta: dict = data.get('meta', {})
  return features.to(device), target.to(device), meta


def build_model_from_ckpt(ckpt_path: Path, device: torch.device):
  ckpt = torch.load(ckpt_path, map_location=device)
  arch = ckpt.get('arch', 'mlp')
  k = int(ckpt.get('k', 1))
  if arch == 'conv':
    model = Dino2GaussConv(
      in_channels=int(ckpt['in_channels']),
      hidden=int(ckpt.get('conv_hidden', 128)),
      offset_max=float(ckpt['offset_max']),
      k=k,
      conv_layers=int(ckpt.get('conv_layers', 1)),
    )
  else:
    model = Dino2GaussMLP(
      in_dim=int(ckpt['in_dim']),
      hidden_layers=tuple(ckpt['hidden']),
      offset_max=float(ckpt['offset_max']),
      k=k,
    )
  model.load_state_dict(ckpt['model_state'], strict=True)
  model.to(device).eval()
  return model, arch


def main() -> None:
  parser = argparse.ArgumentParser(description='Infer Gaussians from cached DINO features using trained model')
  parser.add_argument('--cache_item', type=str, required=True)
  parser.add_argument('--checkpoint', type=str, required=True)
  parser.add_argument('--out', type=str, required=True)
  parser.add_argument('--stride', type=int, default=1)
  parser.add_argument('--k', type=int, default=1)
  parser.add_argument('--device', type=str, default='cuda')
  args = parser.parse_args()

  device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

  features, target, meta = load_cache_item(Path(args.cache_item), device)
  Hf, Wf, C = features.shape
  H, W = target.shape[0], target.shape[1]

  model, arch = build_model_from_ckpt(Path(args.checkpoint), device)

  anchors = build_anchors(H=H, W=W, Hf=Hf, Wf=Wf, stride=args.stride, device=device)
  if arch == 'conv':
    x_in = None
  else:
    feats = gather_latents(features, stride=args.stride)
    ys = torch.arange(0, Hf, args.stride, device=device).float() / float(Hf)
    xs = torch.arange(0, Wf, args.stride, device=device).float() / float(Wf)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([gx, gy], dim=-1).view(-1, 2)
    x_in = torch.cat([feats.to(dtype=torch.float32), coords], dim=-1)

  with torch.no_grad():
    if arch == 'conv':
      preds = model(features, stride=args.stride)
    else:
      preds = model(x_in)
    g2d = pack_gaussians(anchors, preds, image_size=(H, W))

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save(dict(
    position=g2d.position.detach().cpu(),
    depths=g2d.depths.detach().cpu(),
    log_scaling=g2d.log_scaling.detach().cpu(),
    rotation=g2d.rotation.detach().cpu(),
    alpha_logit=g2d.alpha_logit.detach().cpu(),
    feature=g2d.feature.detach().cpu(),
    image_size=(H, W),
    grid_size=(Hf, Wf),
    stride=args.stride,
    k=getattr(model, 'k', args.k),
  ), out)
  print(f'Saved gaussians to {out}')


if __name__ == '__main__':
  main()
