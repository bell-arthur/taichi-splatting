# Taichi Splatting

Rasterizer for Guassian Splatting using Taichi and PyTorch - embedded in python library. Currently very usable but in active development, so likely will break with new versions! 

Trainer: [here](https://github.com/uc-vision/splat-trainer)
Viewer: [here](https://github.com/uc-vision/splat-trainer)

This work is originally derived off [Taichi 3D Gaussian Splatting](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting), with significant re-organisation and changes.

Key differences are the rendering algorithm is decomposed into separate operations (projection, shading functions, tile mapping and rasterization) which can be combined in different ways in order to facilitate a more flexible use, and gradients can be enabled on "all the things" as required for the application (and not when disabled, to save performance).

Using the Taichi autodiff for a simpler implementation where possible (e.g. for projection, but not for the rasterization).

Examples:
  * Projecting features for lifting 2D to 3D
  * Colours via. spherical harmonics
  * Depth covariance without needing to build it into the renderer and remaining differentiable.
  * Fully differentiable camera parameters (and ability to swap in new camera models)

## Performance

A document describing some performance benchmarks of taichi-splatting [here](BENCHMARK.md). Through various optimizations, in particular optimizing the summation of gradients in the backward gradient kernel. Taichi-splatting achieves a very large speedup (often an order of magnitude) over the original taichi_3d_gaussian_splatting, and is faster than the reference diff_guassian_rasterization for a complete optimization pass (forward+backward), in particular much faster at higher resolutions.


## Installing

### External dependencies
Create an environment (for example conda with mambaforge) with the following dependencies:

* python >= 3.10
* pytorch - from either conda  Follow instructions [https://pytorch.org/](here).
* taichi-nightly `pip install --upgrade -i https://pypi.taichi.graphics/simple/ taichi-nightly`

### Install

One of:
* `pip install taichi-splatting`
* Clone down with `git clone` and install with `pip install ./taichi-splatting`

## Executables

### fit_image_gaussians

There exists a toy optimizer for fitting a set of randomly initialized gaussians to some 2D images `fit_image_gaussians` - useful for testing rasterization without the rest of the dependencies.

Fitting an image (fixed points): \
`fit_image_gaussians <image file> --show  --n 20000` 

Fitting an image (split and prune to target): \
`fit_image_gaussians <image file> --show --n 1000 --target 20000` 

See `--help` for other options.

### DINO → Gaussians pipeline (new)

Train a small head (MLP or Convolutional) to predict 2D Gaussians from cached DINO features, infer Gaussians for a specific image, then render/refine in the example.

1) Train from cached features:

MLP head:

```bash
pixi run python -m taichi_splatting.dino2gauss.train \
  --cache_dir /csse/users/abe118/Documents/SENG402/dino-viewer/cache/vits16_l11_s025 \
  --epochs 10 --stride 1 --hidden 256,128 --offset_max 64 --k 64 \
  --opacity_reg 1e-6 --scale_reg 1e-3 \
  --save_ckpt /csse/users/abe118/Documents/SENG402/taichi-splatting/taichi_splatting/dino2gauss/models/dino2gauss_mlp_k64.pth
```

Convolutional head:

```bash
pixi run python -m taichi_splatting.dino2gauss.train \
  --arch conv \
  --cache_dir /csse/users/abe118/Documents/SENG402/dino-viewer/cache/vits16_l11_s025 \
  --epochs 10 --stride 1 --conv_hidden 128 --conv_layers 3 --offset_max 64 --k 64 \
  --opacity_reg 1e-6 --scale_reg 1e-3 \
  --save_ckpt /csse/users/abe118/Documents/SENG402/taichi-splatting/taichi_splatting/dino2gauss/models/dino2gauss_conv_k64.pth
```

2) Infer Gaussians for one cached item (MLP or Conv):

```bash
pixi run python -m taichi_splatting.dino2gauss.infer \
  --cache_item /csse/users/abe118/Documents/SENG402/dino-viewer/cache/vits16_l11_s025/00000.pt \
  --checkpoint /csse/users/abe118/Documents/SENG402/taichi-splatting/taichi_splatting/dino2gauss/models/dino2gauss_conv_k64.pth \
  --stride 1 \
  --out /csse/users/abe118/Documents/SENG402/taichi-splatting/taichi_splatting/dino2gauss/models/gaussians_00000_k64.pth
```

3) Render (optionally refine) from precomputed Gaussians:

```bash
pixi run python taichi_splatting/examples/fit_image_gaussians.py \
  /csse/users/abe118/Documents/SENG402/scan_32/right/00000.jpg \
  --init_from_dino_gaussians /csse/users/abe118/Documents/SENG402/taichi-splatting/taichi_splatting/dino2gauss/models/gaussians_00000_k64.pth \
  --skip_refine --show
```

Or brief refinement with split/prune:

```bash
pixi run python taichi_splatting/examples/fit_image_gaussians.py \
  /csse/users/abe118/Documents/SENG402/scan_32/right/00000.jpg \
  --init_from_dino_gaussians /csse/users/abe118/Documents/SENG402/taichi-splatting/taichi_splatting/dino2gauss/models/gaussians_00000_k64.pth \
  --iters 400 --prune --target 20000 --show
```

Notes:
- Use the same DINO `--scale` and layer across caching, training, and inference.
- Increase `--k` and `--offset_max` for denser coverage; reduce `--scale_reg`/`--opacity_reg` to allow larger/stronger Gaussians.
- Inference auto-detects architecture and hyperparameters from the checkpoint (including conv hidden size and layer count).

#### Convolutional dino2gauss (details)

- Inputs: feature grid `(Hf, Wf, C)` from DINO, plus two coordinate channels `(gx, gy)` in `[0,1]` concatenated as additional input channels.
- Stem: a stack of `conv_layers` blocks of `3×3 conv (hidden) + ReLU` (configurable with `--conv_hidden` and `--conv_layers`).
- Heads: five `1×1` conv heads produce per-cell parameter maps:
  - `pos_offset` `(dx, dy)` (tanh-scaled by `offset_max`)
  - `log_scaling` `(sx, sy)` (clamped to `[-5,5]`)
  - `rotation` `(rx, ry)` (normalised to unit length)
  - `alpha_logit` `(1)`
  - `color` `(r,g,b)` via sigmoid
- Stride: maps are optionally sub-sampled by `--stride`; each retained feature cell anchors one or more Gaussians.
- K Gaussians per cell: set with `--k`; outputs are shaped `(N, K, ·)` then flattened to `(N×K, ·)` for rasterisation.
- Anchors: pixel centers are built for `(H, W)` and aligned to the `(Hf, Wf)` grid and stride; final 2D positions are `anchor + pos_offset`.
- Rasterisation: parameters are packed and rendered with the Taichi 2D rasteriser to match the target image.

Usage tips:
- To reduce visible grid patterns, consider `--stride 1` and/or increasing `--conv_layers` and `--k`.
- `--offset_max` bounds per-cell offsets; increase for more flexibility, balance with `--scale_reg`/`--opacity_reg`.

### benchmarks

There exist benchmarks to evaluate performance on individual components in isolation under `taichi_splatting/benchmarks/`

### tests 

Tests (gradient tests and tests comparing to torch-based reference implementations) can be run with pytest, or individually under 
`taichi_splatting/tests/`

### TensorBoard (training visualisation)

Visualise MLP activations/weights/gradients and per-level HashGrid encodings.

Install TensorBoard:

```bash
python -m pip install tensorboard
```

Run training with TensorBoard:

```bash
python taichi_splatting/examples/fit_image_gaussians.py \
  /csse/users/abe118/Documents/SENG402/scan_32/left/00000.jpg \
  --use_mlp_feature --use_mlp_covariance --use_mlp_alpha --use_hash_encoding \
  --tb_log_dir /csse/users/abe118/Documents/SENG402/outputs/tb \
  --tb_every 25
```

### splat-viewer

A viewer for reconstructions created with the original gaussian-splatting repository can be found [here](https://github.com/uc-vision/splat-viewer) or installed with pip. Has dependencies on open3d and Qt. 

### splat-benchmark

A benchmark for a full rendererer (in the same repository as above) with real reconstructions (rendering the original camera viewpoints).  Options exist for tweaking all the renderer parameters, benchmarking backward pass etc.


## Progress

### Done
* Benchmarks with original + taichi_3dgs rasterizer

* Simple view culling 
* Projection with autograd
* Tile mapping (optimized and improved culling) 
* Rasterizer forward pass and optimized backward pass

* Spherical harmonics with autograd
* Gradient tests for most parts (float64) - including rasterizer!
* Fit to image training example/test
* Depth and depth-covariance rendering

* Compute point visibility in backward pass (useful for model pruning)
* Example training on images with split/prune operations
* Novel heuristics for split and prune operations computed optionally in backward pass



### Todo

* Backward projection autograd takes a while to compile and is not cached properly
* 16 bit representations of parameters
* Depth rendering/regularization method (e.g. 2DGS or related method)
* Some ideas for optimized tilemapper with flat representations (no inner loop)


### Improvements

* Exposed all internal constants as parameters
* Switched to matrices as inputs instead of quaternions
* Tile mapping tighter culling for tile overlaps (~30% less rendered splats!)
* All configuration parameters exposed (e.g. tile_size, saturation threshold etc.)
* Warp reduction based backward pass for rasterizer, a decent boost in performance


## Conventions

### Transformation matrices

Transformations are notated `T_x_y`, for example `T_camera_world` can be used to transform points in the world to points in the local camera by `points_camera = T_camera_world @ points_world`

