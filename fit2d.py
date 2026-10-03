"""P3: fit a target image with N 2D Gaussians by gradient descent on MSE."""

import argparse
import time

import torch

from densify import DensifyConfig, densify
from gaussians import covariance_2d, rotation_2d
from render import render
from utils import get_device, load_image, psnr, write_image

MIN_SCALE = 0.3  # pixels; keeps det(Sigma) away from 0 so Sigma^-1 stays finite


def init_gaussians(n, H, W, device, seed=0):
    """The handout's spread-out init: uniform centers, small round gray blobs."""
    g = torch.Generator().manual_seed(seed)
    params = {
        "mu": torch.rand(n, 2, generator=g) * torch.tensor([W, H], dtype=torch.float32),
        "log_s": torch.log(0.02 * max(H, W) * torch.ones(n, 2)),
        "theta": torch.zeros(n),
        "color": torch.zeros(n, 3),  # sigmoid -> 0.5 gray
        "op_raw": torch.full((n,), -2.0),  # sigmoid -> ~0.12 opacity
    }
    return {k: v.to(device).requires_grad_() for k, v in params.items()}


def render_params(p, H, W):
    Sigma = covariance_2d(p["log_s"].exp().clamp_min(MIN_SCALE), p["theta"])
    order = torch.arange(p["mu"].shape[0], device=p["mu"].device)
    return render(
        p["mu"], Sigma, p["color"].sigmoid(), p["op_raw"].sigmoid(), order, H, W
    )


def fit(
    target,
    n,
    steps=2000,
    lr=1e-2,
    device="cpu",
    log_every=200,
    seed=0,
    densify_cfg=None,
    budget=None,
):
    """Fit n Gaussians to target. With densify_cfg, n is the starting count and
    the set grows toward budget. Returns (params, final image, history), where
    history holds (step, psnr, count)."""
    target = target.to(device)
    H, W = target.shape[:2]
    p = init_gaussians(n, H, W, device, seed)
    opt = torch.optim.Adam(p.values(), lr=lr)
    grad_sum, grad_steps = torch.zeros(n, device=device), 0
    # d(loss)/d(ndc) = d(loss)/d(pixel) * size/2, so g_i is in the 3DGS paper's units
    # and its grad_threshold transfers regardless of image resolution
    to_ndc = torch.tensor([W / 2, H / 2], device=device)
    history = []
    for step in range(1, steps + 1):
        img = render_params(p, H, W)
        loss = ((img - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if densify_cfg:
            grad_sum += (p["mu"].grad * to_ndc).norm(dim=-1)
            grad_steps += 1
        opt.step()

        if (
            densify_cfg
            and step % densify_cfg.every == 0
            and step <= densify_cfg.until * steps
        ):
            p, stats = densify(
                p,
                opt,
                grad_sum / grad_steps,
                budget,
                densify_cfg.size_threshold * W,
                lambda q: rotation_2d(q["theta"]),
                densify_cfg,
            )
            grad_sum, grad_steps = torch.zeros(stats["count"], device=device), 0
            print(f"  densify @ {step}: {stats}")
        if step % log_every == 0 or step == steps:
            history.append((step, psnr(loss.item()), p["mu"].shape[0]))
            print(f"step {step:5d}  PSNR {history[-1][1]:.2f} dB  N={history[-1][2]}")
    with torch.no_grad():
        img = render_params(p, H, W)
    return p, img, history


def main():
    ap = argparse.ArgumentParser(description="P3: fit a 2D image with Gaussians")
    ap.add_argument("image")
    ap.add_argument(
        "--n",
        type=int,
        default=1024,
        help="number of Gaussians (with --densify: the final budget)",
    )
    ap.add_argument(
        "--densify", action="store_true", help="P4 adaptive density control"
    )
    ap.add_argument(
        "--start-n", type=int, help="starting count with --densify (default n/4)"
    )
    ap.add_argument(
        "--grad-threshold", type=float, default=DensifyConfig.grad_threshold
    )
    ap.add_argument("--size", type=int, default=128, help="longer side in pixels")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--output", default="fit.png")
    ap.add_argument("--device", default=get_device())
    args = ap.parse_args()

    target = load_image(args.image, args.size)
    t0 = time.time()
    if args.densify:
        cfg = DensifyConfig(grad_threshold=args.grad_threshold)
        p, img, _ = fit(
            target,
            args.start_n or args.n // 4,
            args.steps,
            args.lr,
            args.device,
            densify_cfg=cfg,
            budget=args.n,
        )
    else:
        p, img, _ = fit(target, args.n, args.steps, args.lr, args.device)
    final = psnr(((img.cpu() - target) ** 2).mean().item())
    write_image(img, args.output)
    print(
        f"{args.image}: N={p['mu'].shape[0]}  {target.shape[1]}x{target.shape[0]}  "
        f"PSNR {final:.2f} dB  ({time.time() - t0:.0f}s)  -> {args.output}"
    )


if __name__ == "__main__":
    main()
