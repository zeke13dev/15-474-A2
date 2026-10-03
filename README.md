# A2: Gaussian Splatting

15-474 Assignment 2. A differentiable 2D Gaussian rasterizer fit to single
images (P1-P5), then 3D Gaussians projected through posed cameras to
reconstruct a small synthetic scene (P6-P9).

## Layout

| file | purpose |
|---|---|
| `gaussians.py` | P1 covariance from scale + rotation, Gaussian weight, pixel grid |
| `render.py` | P2 front-to-back compositing; tiled renderer (`render`) and its dense reference (`render_dense`) |
| `fit2d.py` | P3 fit one image (CLI); `--densify` turns on P4 |
| `densify.py` | P4/P8 clone, split, prune, opacity reset; shared by 2D and 3D |
| `p5.py` | P5 sweep over images x Gaussian counts, results JSON and plot |
| `gaussians3d.py` | P6 quaternion rotation, 3D covariance, camera projection with Jacobian, `look_at` |
| `fit3d.py` | P7 3D fit, P8 densified fit, P9 held-out evaluation and orbit |
| `utils.py` | device selection, image I/O, PSNR |
| `test_*.py`, `run_tests.py` | unit tests |
| `data/` | the three 2D targets and the `spheres/` multi-view scene |
| `results/` | renders, `p5.json`, `psnr_vs_n.png`, and `3d/<run>/` folders |

## Setup

```sh
python -m venv venv
venv/bin/python -m pip install torch torchvision numpy pillow matplotlib
```

PyTorch picks `cuda`, then `mps`, then `cpu`. Times below are on an M-series
Mac with `mps`.

## Reproduce

```sh
venv/bin/python run_tests.py                # unit tests (pytest also works)

# 2D (P3-P5): 128x128 targets, 2000 Adam steps, lr 1e-2
venv/bin/python fit2d.py data/astronaut.png --n 1024 --output results/astronaut_1024.png
venv/bin/python p5.py                       # fixed-count sweep, ~10 min
venv/bin/python p5.py --densify             # densified sweep, ~8 min
venv/bin/python p5.py --plot-only           # redraw results/psnr_vs_n.png from p5.json

# 3D (P7-P9): 160x160 views, 1500 steps, one random training camera per step
venv/bin/python fit3d.py                    # plain fit, 4000 Gaussians -> results/3d/plain/
venv/bin/python fit3d.py --densify          # densified fit -> results/3d/densify/

# P8 ablations (results/old/ablate_*): no opacity reset, and a stricter prune threshold
venv/bin/python fit3d.py --densify --reset-every 0 --out results/3d/ablate_noreset
venv/bin/python fit3d.py --densify --prune-opacity 0.01 --out results/3d/ablate_reset400_prune01
```

MPS training is not bit-for-bit deterministic; two identical 3D runs differed
by about 0.2 dB held-out PSNR.

Each `fit3d.py` run takes about 2-3 minutes. It writes `params.pt`, a
`results.json` with train / held-out PSNR (mean of per-view PSNR), a
ground-truth | render image for every training and held-out view, and a
36-frame orbit (`orbit.gif`, plus four stills) at 25° elevation, between the
training orbits. `--eval-only` reloads `params.pt` and redoes the evaluation.

## Implementation notes

- **Tiled rasterizer.** `render` cuts the image into 16x16 tiles and composites
  each tile only against the Gaussians whose 3.5σ bounding box overlaps it.
  That brought a 4096-Gaussian 2D step from ~1.25 s to ~0.06 s. Both renderers
  drop contributions beyond 3.5σ, and a test checks that they agree.
- **Densification score** is the mean position-gradient norm since the last
  pass. In 2D it is measured in normalized device coordinates (pixel
  gradient x size/2), the 3DGS paper's units, so the threshold does not depend
  on resolution. In 3D it is the world-space gradient.
- **Opacity reset (3D only).** Floaters in the plain fit sit at opacity ~0.1,
  far above the 0.005 prune threshold, so pruning alone removes nothing.
  Periodically capping all opacities at 0.01 lets the Gaussians the renders
  need grow back while haze stays faint and becomes prunable.
- **3D rendering** skips Gaussians within 0.2 of the camera plane, sorts the
  rest by camera depth, and adds 0.3 px² to each projected covariance (as in
  3DGS) so no splat is thinner than about a pixel.
