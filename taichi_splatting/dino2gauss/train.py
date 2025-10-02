from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import taichi as ti
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from taichi_splatting.dino2gauss.model import Dino2GaussMLP
from taichi_splatting.dino2gauss.utils import build_anchors, gather_latents, pack_gaussians, pack_gaussians_tensor
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer.function import rasterize
from taichi_splatting.taichi_queue import TaichiQueue


def load_cache_item(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
  data: Dict[str, Any] = torch.load(path, map_location='cpu')
  features: torch.Tensor = data['features']        # (Hf,Wf,C), likely bfloat16
  target: torch.Tensor = data['target']            # (H,W,3) float in [0,1]
  meta: dict = data.get('meta', {})
  return features.to(device), target.to(device), meta


def psnr(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> float:
  mse = torch.mean((pred - tgt) ** 2).item()
  if mse <= eps:
    return 99.0
  import math
  return 20.0 * math.log10(1.0) - 10.0 * math.log10(mse)


def main() -> None:
  parser = argparse.ArgumentParser(description='Train MLP to map DINO features to 2D Gaussians')
  parser.add_argument('--cache_dir', type=str, required=True)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--stride', type=int, default=1)
  parser.add_argument('--hidden', type=str, default='128,128')
  parser.add_argument('--offset_max', type=float, default=4.0)
  parser.add_argument('--k', type=int, default=1)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--opacity_reg', type=float, default=1e-5)
  parser.add_argument('--scale_reg', type=float, default=1e-1)
  parser.add_argument('--save_ckpt', type=str, default='models/dino2gauss_mlp.pth')
  parser.add_argument('--device', type=str, default='cuda')
  args = parser.parse_args()

  device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

  # Initialize Taichi runtime/queue (required before rasterization)
  TaichiQueue.init(arch=ti.cuda if device.type == 'cuda' else ti.cpu,
                   log_level=ti.INFO,
                   device_memory_GB=0.1,
                   threaded=False)

  cache_dir = Path(args.cache_dir)
  cache_items = sorted(cache_dir.glob('*.pt'))
  assert len(cache_items) > 0, f'No cache items in {cache_dir}'

  # Peek first item to infer sizes
  features0, target0, meta0 = load_cache_item(cache_items[0], device)
  Hf, Wf, C = features0.shape
  H, W = target0.shape[0], target0.shape[1]

  hidden = tuple(int(x) for x in args.hidden.split(',')) if args.hidden else tuple()
  in_dim = C + 2  # features + (iy, ix)
  model = Dino2GaussMLP(in_dim=in_dim, hidden_layers=hidden, offset_max=args.offset_max, k=args.k).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=args.lr)

  config = RasterConfig(compute_point_heuristic=False, compute_visibility=False)

  for epoch in range(1, args.epochs + 1):
    total_loss = 0.0
    total_psnr = 0.0
    n_pix = 0
    for path in cache_items:
      features, target, meta = load_cache_item(path, device)
      Hf, Wf, C = features.shape
      H, W = target.shape[0], target.shape[1]

      anchors = build_anchors(H=H, W=W, Hf=Hf, Wf=Wf, stride=args.stride, device=device)
      feats = gather_latents(features, stride=args.stride)
      # add normalized coords
      ys = torch.arange(0, Hf, args.stride, device=device).float() / float(Hf)
      xs = torch.arange(0, Wf, args.stride, device=device).float() / float(Wf)
      gy, gx = torch.meshgrid(ys, xs, indexing='ij')
      coords = torch.stack([gx, gy], dim=-1).view(-1, 2)
      x_in = torch.cat([feats.to(dtype=torch.float32), coords], dim=-1)

      opt.zero_grad(set_to_none=True)
      with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        preds = model(x_in)
        g2d = pack_gaussians(anchors, preds, image_size=(H, W))
        g2d_tensor = pack_gaussians_tensor(g2d)
        raster = rasterize(gaussians2d=g2d_tensor, depth=g2d.depths.clamp(0, 1), features=g2d.feature, image_size=(W, H), config=config)
        img_pred = raster.image
        img_tgt = target
        loss = F.mse_loss(img_pred, img_tgt) + args.opacity_reg * g2d.opacity.mean() + args.scale_reg * torch.exp(g2d.log_scaling).pow(2).mean()
      loss.backward()
      opt.step()

      total_loss += loss.item() * (H * W)
      total_psnr += psnr(img_pred.detach().clamp(0,1), img_tgt) * (H * W)
      n_pix += (H * W)

    print(f'Epoch {epoch:03d} | loss {total_loss / n_pix:.6f} | PSNR {total_psnr / n_pix:.2f} dB')

  Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
  torch.save(dict(model_state=model.state_dict(), in_dim=in_dim, hidden=hidden, offset_max=args.offset_max, k=args.k), args.save_ckpt)
  print(f'Saved checkpoint to {args.save_ckpt}')


if __name__ == '__main__':
  main()


