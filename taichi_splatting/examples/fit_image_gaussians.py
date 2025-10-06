import argparse
import math
import time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import taichi as ti
import torch
import torch.nn.functional as F
import torch.nn as nn
from beartype import beartype
from logger_utils import TrainingLogger
from mlp_predictors import ConfigurableMLP
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.misc.renderer2d import (point_basis, project_gaussians2d,
                                              uniform_split_gaussians2d)
from taichi_splatting.optim.fractional import FractionalAdam
from taichi_splatting.optim.parameter_class import ParameterClass
from taichi_splatting.optim.visibility_aware import (VisibilityAwareLaProp,
                                                     VisibilityOptimizer)
from taichi_splatting.rasterizer.function import rasterize
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians
from taichi_splatting.torch_lib.util import check_finite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tile_size', type=int, default=16)
    parser.add_argument('--pixel_tile', type=str,
                        help='Pixel tile for backward pass default "2,2"')

    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--target', type=int, default=None)
    parser.add_argument('--prune', action='store_true',
                        help='enable pruning (equivalent to --target=n)')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--max_lr', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=0.1)

    parser.add_argument('--epoch', type=int, default=8,
                        help='base epoch size (increases with t)')
    parser.add_argument('--max_epoch', type=int, default=32)

    parser.add_argument('--prune_rate', type=float, default=0.025,
                        help='Rate of pruning proportional to number of points')
    parser.add_argument('--opacity_reg', type=float, default=0.00001)
    parser.add_argument('--scale_reg', type=float, default=0.1)

    parser.add_argument('--threaded', action='store_true',
                        help='Use taichi dedicated thread')

    parser.add_argument('--antialias', action='store_true')

    parser.add_argument('--write_frames', type=Path, default=None)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--show', action='store_true')

    parser.add_argument('--profile', action='store_true')

    # Initialise from precomputed DINO->Gaussians
    parser.add_argument('--init_from_dino_gaussians', type=str, default=None,
                        help='Path to gaussians.pth from dino2gauss.infer')
    parser.add_argument('--skip_refine', action='store_true',
                        help='If set, render-only without optimization')
    parser.add_argument('--save_render', type=str, default=None,
                        help='If set with --skip_refine, save final render to this image path and exit')

    for attr in ['position', 'feature', 'covariance', 'alpha']:
        parser.add_argument(f'--use_mlp_{attr}', action='store_true')
        parser.add_argument(f'--freeze_mlp_{attr}', action='store_true')
        parser.add_argument(f'--load_mlp_{attr}', type=str, default=None)
        parser.add_argument(f'--save_mlp_{attr}', type=str, default=None)
        parser.add_argument(f'--mlp_{attr}_layers', type=str, default="32")
        parser.add_argument(
            f'--mlp_{attr}_activation', type=str, default="ReLU")

    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--save_csv', type=str, default=None,
                        help='Filename to save training log as CSV')

    parser.add_argument('--use_hash_encoding', action='store_true')

    # TensorBoard logging options
    parser.add_argument('--tb_log_dir', type=str, default=None,
                        help='Enable TensorBoard logging to this directory')
    parser.add_argument('--tb_every', type=int, default=25,
                        help='Log histograms every N iterations')

    args = parser.parse_args()

    for attr in ['position', 'feature', 'covariance', 'alpha']:
        setattr(args, f'mlp_{attr}_layers', list(
            map(int, getattr(args, f'mlp_{attr}_layers').split(','))))

    if args.pixel_tile:
        args.pixel_tile = tuple(map(int, args.pixel_tile.split(',')))

    return args


def log_lerp(t, a, b):
    return math.exp(math.log(b) * t + math.log(a) * (1 - t))


def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(1)


def psnr(a, b):
    return 10 * torch.log10(1 / F.mse_loss(a, b))


def normalize_position(position: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Normalize 2D positions to [0, 1] range for hash encoding."""
    norm_pos = position.clone()
    norm_pos[:, 0] = norm_pos[:, 0] / width
    norm_pos[:, 1] = norm_pos[:, 1] / height
    return norm_pos.clamp(0.0, 1.0)


def train_epoch(opt: FractionalAdam, params: ParameterClass, ref_image,
                config: RasterConfig,
                epoch_size=100,
                opacity_reg=0.0,
                scale_reg=0.0,
                mlps: Optional[dict] = None,
                mlp_optimizers: Optional[dict] = None,
                writer: Optional[SummaryWriter] = None,
                step_counter: Optional[dict] = None,
                tb_every: int = 25):

    h, w = ref_image.shape[:2]

    point_heuristic = torch.zeros(
        (params.batch_size[0], 2), device=params.position.device)
    image = torch.zeros_like(ref_image)

    if mlps is None:
        mlps = {}
    if mlp_optimizers is None:
        mlp_optimizers = {}

    for i in range(epoch_size):
        opt.zero_grad()
        for optim in mlp_optimizers.values():
            optim.zero_grad()

        with torch.enable_grad():
            gaussians = Gaussians2D(**params.tensors)  # type: ignore[misc]

            if 'position' in mlps:
                _mlp = mlps['position']
                _device = next(_mlp.parameters()).device
                gaussians.position = _mlp(params.latent.to(dtype=torch.float32, device=_device))
            else:
                gaussians.position = params.position

            if 'feature' in mlps:
                _mlp = mlps['feature']
                _device = next(_mlp.parameters()).device
                if _mlp.use_hash_encoding:
                    input_feature = normalize_position(
                        gaussians.position, w, h)
                else:
                    input_feature = params.latent

                input_feature = input_feature.to(dtype=torch.float32, device=_device)
                # Per-level HashGrid diagnostics
                if writer is not None and _mlp.use_hash_encoding and step_counter is not None:
                    if (step_counter['i'] % tb_every) == 0:
                        enc = _mlp.encode_only(input_feature)
                        hc = getattr(_mlp, 'hash_config', None)
                        if hc is not None:
                            L = int(hc.get('n_levels', 1))
                            Fpl = int(hc.get('n_features_per_level', enc.shape[-1]))
                            for li in range(L):
                                s, e = li * Fpl, (li + 1) * Fpl
                                writer.add_histogram(
                                    f"mlp_feature/enc/level_{li}", enc[:, s:e].detach().float(),
                                    global_step=step_counter['i'])
                gaussians.feature = _mlp(
                    input_feature).contiguous().to(torch.float32)
            else:
                gaussians.feature = params.feature

            if 'covariance' in mlps:
                _mlp = mlps['covariance']
                _device = next(_mlp.parameters()).device
                if _mlp.use_hash_encoding:
                    input_cov = normalize_position(gaussians.position, w, h)
                else:
                    input_cov = params.latent

                input_cov = input_cov.to(dtype=torch.float32, device=_device)
                if writer is not None and _mlp.use_hash_encoding and step_counter is not None:
                    if (step_counter['i'] % tb_every) == 0:
                        enc = _mlp.encode_only(input_cov)
                        hc = getattr(_mlp, 'hash_config', None)
                        if hc is not None:
                            L = int(hc.get('n_levels', 1))
                            Fpl = int(hc.get('n_features_per_level', enc.shape[-1]))
                            for li in range(L):
                                s, e = li * Fpl, (li + 1) * Fpl
                                writer.add_histogram(
                                    f"mlp_covariance/enc/level_{li}", enc[:, s:e].detach().float(),
                                    global_step=step_counter['i'])
                cov_out = _mlp(
                    input_cov).contiguous().to(torch.float32)
                gaussians.log_scaling = torch.clamp(
                    cov_out[..., :2], min=-5, max=5)
                gaussians.rotation = F.normalize(cov_out[..., 2:], dim=-1)
            else:
                gaussians.log_scaling = params.log_scaling
                gaussians.rotation = params.rotation

            if 'alpha' in mlps:
                _mlp = mlps['alpha']
                _device = next(_mlp.parameters()).device
                if _mlp.use_hash_encoding:
                    input_alpha = normalize_position(gaussians.position, w, h)
                else:
                    input_alpha = params.latent
                input_alpha = input_alpha.to(dtype=torch.float32, device=_device)
                if writer is not None and _mlp.use_hash_encoding and step_counter is not None:
                    if (step_counter['i'] % tb_every) == 0:
                        enc = _mlp.encode_only(input_alpha)
                        hc = getattr(_mlp, 'hash_config', None)
                        if hc is not None:
                            L = int(hc.get('n_levels', 1))
                            Fpl = int(hc.get('n_features_per_level', enc.shape[-1]))
                            for li in range(L):
                                s, e = li * Fpl, (li + 1) * Fpl
                                writer.add_histogram(
                                    f"mlp_alpha/enc/level_{li}", enc[:, s:e].detach().float(),
                                    global_step=step_counter['i'])
                gaussians.alpha_logit = _mlp(input_alpha).squeeze(-1)
            else:
                gaussians.alpha_logit = params.alpha_logit

            gaussians2d = project_gaussians2d(gaussians)

            raster = rasterize(gaussians2d=gaussians2d,
                               depth=gaussians.depths.clamp(0, 1),
                               features=gaussians.feature,
                               image_size=(w, h),
                               config=config)

            image = raster.image

            scale = torch.exp(gaussians.log_scaling) / min(w, h)
            loss = (F.mse_loss(image, ref_image) +
                    opacity_reg * gaussians.opacity.mean() +
                    scale_reg * scale.pow(2).mean())

            loss.backward()
            # Log parameter and gradient histograms
            if writer is not None and step_counter is not None and (step_counter['i'] % tb_every) == 0:
                for name, _mlp in mlps.items():
                    for pname, p in _mlp.named_parameters():
                        writer.add_histogram(f"mlp_{name}/weights/{pname}", p.detach().float(), global_step=step_counter['i'])
                        if p.grad is not None:
                            writer.add_histogram(f"mlp_{name}/grads/{pname}", p.grad.detach().float(), global_step=step_counter['i'])
            for optim in mlp_optimizers.values():
                optim.step()

        check_finite(gaussians, 'gaussians')
        visible = (raster.visibility > 1e-8).nonzero().squeeze(1)  # type: ignore[operator]

        # Create subset of gaussians for visible points
        visible_gaussians = Gaussians2D(
            position=gaussians.position[visible],
            depths=gaussians.depths[visible],
            log_scaling=gaussians.log_scaling[visible],
            rotation=gaussians.rotation[visible],
            alpha_logit=gaussians.alpha_logit[visible],
            feature=gaussians.feature[visible],
            latent=gaussians.latent[visible]
        )
        
        if isinstance(opt, VisibilityOptimizer):
            opt.step(indexes=visible,  # type: ignore[arg-type]
                     visibility=raster.visibility[visible],  # type: ignore[index]
                     basis=point_basis(visible_gaussians).to(torch.float32))
        else:
            opt.step(indexes=visible,
                     weight=torch.ones_like(visible, dtype=torch.float32),
                     basis=point_basis(visible_gaussians).to(torch.float32))

        if 'covariance' not in mlps:
            params.replace(
                rotation=F.normalize(
                    params.rotation.detach()),
                log_scaling=torch.clamp(
                    params.log_scaling.detach(), min=-5, max=5)
            )

        point_heuristic += raster.point_heuristic  # type: ignore[operator]

        if writer is not None and step_counter is not None:
            step_counter['i'] += 1

    return image, (point_heuristic[:, 0], point_heuristic[:, 1])


def make_epochs(total_iters, first_epoch, max_epoch):
    iteration = 0
    epochs = []
    while iteration < total_iters:

        t = iteration / total_iters
        epoch_size = math.ceil(log_lerp(t, first_epoch, max_epoch))

        if iteration + epoch_size * 2 > total_iters:
            # last epoch can just use the extra iterations
            epoch_size = total_iters - iteration

        iteration += epoch_size
        epochs.append(epoch_size)

    return epochs


@beartype
def take_n(t: torch.Tensor, n: int, descending=False):
    """ Return mask of n largest or smallest values in a tensor."""
    idx = torch.argsort(t, descending=descending)[:n]

    # convert to mask
    mask = torch.zeros_like(t, dtype=torch.bool)
    mask[idx] = True

    return mask


def randomize_n(t: torch.Tensor, n: int):
    """ Randomly select n of the largest values in a tensor using torch.multinomial"""
    probs = F.normalize(t, dim=0)
    mask = torch.zeros_like(t, dtype=torch.bool)

    if n > 0:
        selected_indices = torch.multinomial(probs, n, replacement=False)
        mask[selected_indices] = True

    return mask


def find_split_prune(n, target, n_prune, prune_cost, densify_score):
    prune_mask = take_n(prune_cost, n_prune, descending=False)
    n_prune = prune_mask.sum().item()

    target_split = max(0, (target - n) + n_prune)

    # split_mask = randomize_n(densify_score, min(target_split, n))
    split_mask = take_n(densify_score, target_split, descending=True)

    both = (split_mask & prune_mask)
    return split_mask ^ both, prune_mask ^ both


def split_prune(params: ParameterClass, t, target, prune_rate, split_heuristic: Tuple[torch.Tensor, torch.Tensor]):
    n = params.batch_size[0]

    prune_cost, split_score = split_heuristic

    split_mask, prune_mask = find_split_prune(n=n,
                                              target=target,
                                              n_prune=int(
                                                  prune_rate * n * (1 - t)),
                                              # n_prune=int(prune_rate * n),
                                              prune_cost=prune_cost,
                                              densify_score=split_score)

    to_split = params[split_mask]

    splits = uniform_split_gaussians2d(
        Gaussians2D(**to_split.tensors), random_axis=True)  # type: ignore[misc]
    # Ensure splits are on the correct device
    splits = splits.to(params.position.device)
    optim_state = to_split.tensor_state.new_zeros((to_split.batch_size[0], 2))  # type: ignore[arg-type]

    # optim_state['position']['running_vis'][:] = to_split.tensor_state['position']['running_vis'].unsqueeze(1) * 0.5

    params = params[~(split_mask | prune_mask)]
    params = params.append_tensors(
        splits.to_tensordict(), optim_state.reshape(splits.batch_size))  # type: ignore[attr-defined]
    # params.replace(rotation = torch.nn.functional.normalize(params.rotation.detach()))

    return params, dict(
        split=split_mask.sum().item(),
        prune=prune_mask.sum().item()
    )


def main():
    logger = TrainingLogger()

    torch.set_printoptions(precision=4, sci_mode=False)

    cmd_args = parse_args()
    device = torch.device('cuda:0')

    writer: Optional[SummaryWriter] = None
    if cmd_args.tb_log_dir:
        writer = SummaryWriter(log_dir=cmd_args.tb_log_dir)
        print(f"TensorBoard logging to {cmd_args.tb_log_dir}")

    torch.set_grad_enabled(False)

    ref_image = cv2.imread(cmd_args.image_file)
    assert ref_image is not None, f'Could not read {cmd_args.image_file}'

    h, w = ref_image.shape[:2]

    TaichiQueue.init(arch=ti.cuda, log_level=ti.INFO,
                     debug=cmd_args.debug, device_memory_GB=0.1, threaded=cmd_args.threaded)

    print(f'Image size: {w}x{h}')

    if cmd_args.show and not (cmd_args.skip_refine and cmd_args.save_render):
        cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('rendered', w, h)

    torch.manual_seed(cmd_args.seed)
    lr_range = (cmd_args.max_lr, cmd_args.min_lr)

    torch.cuda.random.manual_seed(cmd_args.seed)
    if cmd_args.init_from_dino_gaussians:
        gdata = torch.load(cmd_args.init_from_dino_gaussians, map_location=device)
        gaussians = Gaussians2D(
            position=gdata['position'].to(device),
            depths=gdata['depths'].to(device),
            log_scaling=gdata['log_scaling'].to(device),
            rotation=gdata['rotation'].to(device),
            alpha_logit=gdata['alpha_logit'].to(device).squeeze(-1),
            feature=gdata['feature'].to(device),
            latent=torch.zeros((gdata['position'].shape[0], 1), device=device),
            batch_size=(gdata['position'].shape[0],)
        ).to(device)
        print(f'Loaded {gaussians.batch_size[0]} gaussians from {cmd_args.init_from_dino_gaussians}')

        # Adjust render/target resolution to match cached features' target size
        if 'image_size' in gdata:
            gh, gw = gdata['image_size']
            if (h, w) != (gh, gw):
                # Resize reference image for consistent loss/rendering
                ref_image = cv2.resize(ref_image, (gw, gh), interpolation=cv2.INTER_AREA)
                h, w = gh, gw
                if cmd_args.show:
                    cv2.resizeWindow('rendered', w, h)
    else:
        gaussians = random_2d_gaussians(
            cmd_args.n,
            (w, h),
            alpha_range=(0.5, 1.0),
            scale_factor=0.5,
            latent_dim=cmd_args.latent_dim,
        ).to(device)

    mlps = {}
    mlp_optimizers = {}
    step_counter = {'i': 0}

    def register_activation_hooks(name: str, model: ConfigurableMLP):
        if writer is None:
            return
        for idx, module in enumerate(model.mlp):
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.ELU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.Softplus)):
                module.register_forward_hook(
                    lambda m, inp, out, n=name, i=idx: writer.add_histogram(
                        f"mlp_{n}/act/{m.__class__.__name__}_{i}", out.detach().float(), global_step=step_counter['i']))

    for attr in ['position', 'feature', 'covariance', 'alpha']:
        if getattr(cmd_args, f'use_mlp_{attr}'):
            out_dim = {
                'position': 2,
                'feature': 3,
                'covariance': 4,
                'alpha': 1,
            }[attr]
            mlp = ConfigurableMLP(
                in_dim=cmd_args.latent_dim,
                out_dim=out_dim,
                hidden_layers=getattr(cmd_args, f'mlp_{attr}_layers'),
                activation=getattr(cmd_args, f'mlp_{attr}_activation'),
                use_hash_encoding=(
                    cmd_args.use_hash_encoding and attr != 'position')
            ).to(device)

            path = getattr(cmd_args, f'load_mlp_{attr}')
            if path:
                mlp.load_state_dict(torch.load(path))
                print(f"Loaded MLP for {attr} from {path}")

            if getattr(cmd_args, f'freeze_mlp_{attr}'):
                for param in mlp.parameters():
                    param.requires_grad = False
            else:
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, mlp.parameters()),
                    lr=0.001, betas=(0.9, 0.99)
                )
                mlp_optimizers[attr] = optimizer

            mlps[attr] = mlp
            register_activation_hooks(attr, mlp)

    parameter_groups = {}
    if not cmd_args.use_mlp_position:
        parameter_groups['position'] = dict(
            lr=cmd_args.max_lr, type='local_vector')
    if not cmd_args.use_mlp_feature:
        parameter_groups['feature'] = dict(lr=0.1, type='vector')
    if not cmd_args.use_mlp_covariance:
        parameter_groups['log_scaling'] = dict(lr=0.1)
        parameter_groups['rotation'] = dict(lr=1.0)
    if not cmd_args.use_mlp_alpha:
        parameter_groups['alpha_logit'] = dict(lr=0.1)

    if any(getattr(cmd_args, f'use_mlp_{attr}') for attr in ['position', 'feature', 'covariance', 'alpha']):
        parameter_groups['latent'] = dict(lr=0.01)

    # params = ParameterClass(gaussians.to_tensordict(),
    #       parameter_groups, optimizer=SparseAdam, betas=(0.9, 0.95), eps=1e-16, bias_correction=True)

    params = ParameterClass(gaussians.to_tensordict(),
                            parameter_groups, optimizer=VisibilityAwareLaProp,
                            vis_smooth=0.1, vis_beta=0.8, betas=(0.9, 0.9), eps=1e-16, bias_correction=True)

    keys = set(params.keys())  # type: ignore[arg-type]
    trainable = set(params.optimized_keys())  # type: ignore[arg-type]

    print(f'attributes - trainable: {trainable} other: {keys - trainable}')

    ref_image = torch.from_numpy(ref_image).to(
        dtype=torch.float32, device=device) / 255

    config = RasterConfig(compute_point_heuristic=True,
                          compute_visibility=True,

                          tile_size=cmd_args.tile_size,
                          blur_cov=0.3 if not cmd_args.antialias else 0.0,
                          antialias=cmd_args.antialias,
                          # alpha_threshold=1/8192,
                          pixel_stride=cmd_args.pixel_tile or (2, 2))

    def timed_epoch(*args, **kwargs):
        start = time.time()
        image, split_heuristic = train_epoch(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()

        return image, split_heuristic, end - start

    train = with_benchmark(timed_epoch) if cmd_args.profile else timed_epoch
    epochs = make_epochs(cmd_args.iters, cmd_args.epoch, cmd_args.max_epoch)

    pbar = tqdm(total=cmd_args.iters)
    iteration = 0
    for epoch_size in epochs:

        t = (iteration + epoch_size * 0.5) / cmd_args.iters
        params.set_learning_rate(position=log_lerp(t, *lr_range))
        metrics = {}
        image, split_heuristic, epoch_time = train(
            params.optimizer,
            params,
            ref_image,
            config=config,
            epoch_size=epoch_size,
            opacity_reg=cmd_args.opacity_reg,
            scale_reg=cmd_args.scale_reg,
            mlps=mlps,
            mlp_optimizers=mlp_optimizers,
            writer=writer,
            step_counter=step_counter,
            tb_every=cmd_args.tb_every
        )

        if cmd_args.show and not (cmd_args.skip_refine and cmd_args.save_render):
            display_image('rendered', image)

        if cmd_args.write_frames:
            filename = cmd_args.write_frames / f'{iteration:04d}.png'
            filename.parent.mkdir(exist_ok=True, parents=True)
            print(f'Writing {filename}')
            cv2.imwrite(str(filename),
                        (image.detach().clamp(0, 1) * 255).cpu().numpy())

        psnr_value = psnr(ref_image, image).item()

        # Log PSNR, iteration count, and number of points
        logger.log(iteration=iteration, psnr=psnr_value,
                   n_points=params.batch_size[0])

        metrics['CPSNR'] = psnr_value
        metrics['n'] = params.batch_size[0]
        metrics['time_s'] = epoch_time

        if cmd_args.prune and cmd_args.target is None:
            cmd_args.target = cmd_args.n

        if cmd_args.target and iteration + epoch_size < cmd_args.iters:
            t_points = min(math.pow(t * 2, 0.5), 1.0)
            target = math.ceil(
                params.batch_size[0] * (1 - t_points) + t_points * cmd_args.target)
            params, prune_metrics = split_prune(
                params, t, target, cmd_args.prune_rate, split_heuristic)
            metrics.update(prune_metrics)

        for k, v in metrics.items():
            if isinstance(v, float):
                metrics[k] = f'{v:.2f}'
            if isinstance(v, int):
                metrics[k] = f'{v:4d}'

        pbar.set_postfix(**metrics)

        if cmd_args.skip_refine:
            # Render-only mode: save or show once, then exit
            if cmd_args.save_render:
                out_path = Path(cmd_args.save_render)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), (image.detach().clamp(0, 1) * 255).cpu().numpy())
                print(f'Saved render to {out_path}')
            else:
                if cmd_args.show:
                    display_image('rendered', image)
            break
        iteration += epoch_size
        pbar.update(epoch_size)

    # logger.plot()

    if cmd_args.save_csv:
        logger.save_csv(cmd_args.save_csv)

    if writer is not None:
        writer.flush()
        writer.close()

    for attr, mlp in mlps.items():
        path = getattr(cmd_args, f'save_mlp_{attr}')
        if path:
            torch.save(mlp.state_dict(), path)
            print(f"Saved MLP for {attr} to {path}")


def with_benchmark(f):
    def g(*args, **kwargs):
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                result = f(*args, **kwargs)
                torch.cuda.synchronize()

            prof_table = prof.key_averages().table(sort_by="self_cuda_time_total",
                                                   row_limit=25, max_name_column_width=100)
            print(prof_table)
            return result
    return g


if __name__ == '__main__':
    main()
