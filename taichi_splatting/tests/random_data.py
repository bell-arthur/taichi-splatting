import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from taichi_splatting.data_types import Gaussians2D
from taichi_splatting.torch_lib.transforms import join_rt, quat_to_mat


def inverse_sigmoid(x):
    """Inverse sigmoid function"""
    return torch.log(x / (1 - x))


def random_2d_gaussians(n, image_size: Tuple[int, int],
                        num_channels=3, scale_factor=1.0, alpha_range=(0.1, 0.9), depth_range=(0.0, 1.0),
                        latent_dim: int = 16) -> Gaussians2D:

    w, h = image_size
    position = torch.rand(
        n, 2) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)
    depth = torch.rand((n, 1)) * \
        (depth_range[1] - depth_range[0]) + depth_range[0]

    density_scale = scale_factor * w / (1 + math.sqrt(n))
    scaling = (torch.rand(n, 2) + 0.2) * density_scale

    rotation = torch.randn(n, 2)
    rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

    low, high = alpha_range
    alpha = torch.rand(n) * (high - low) + low

    latent = torch.randn(n, latent_dim)

    return Gaussians2D(
        position=position,
        depths=depth,
        log_scaling=torch.log(scaling),
        rotation=rotation,
        alpha_logit=inverse_sigmoid(alpha),
        feature=torch.rand(n, num_channels),
        latent=latent,
        batch_size=(n,)
    )
