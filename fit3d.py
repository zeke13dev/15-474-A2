"""P7-P9: reconstruct the spheres scene with 3D Gaussians from posed views.

    python fit3d.py                      # P7 plain fit  -> results/3d/plain/
    python fit3d.py --densify            # P8 densified  -> results/3d/densify/
    python fit3d.py --densify --eval-only  # reload saved params, redo P9 eval + orbit
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from densify import DensifyConfig, densify, reset_opacity
from gaussians3d import covariance_3d, look_at, project_gaussian, quaternion_to_rotation
from render import render
from utils import get_device, load_image, psnr, write_image

NEAR = 0.2        # Gaussians closer than this to the camera plane are skipped
DILATION = 0.3    # px^2 added to each 2D covariance so no splat is thinner than ~a pixel (as in 3DGS)
MIN_SCALE = 1e-3  # world units; keeps the 3D covariance from collapsing


def load_scene(root, device):
    meta = json.loads((Path(root) / "cameras.json").read_text())
    K = torch.tensor(meta["K"], dtype=torch.float32, device=device)

    def views(frames):
        return [dict(name=f["file"], image=load_image(Path(root) / f["file"]).to(device),
                     R=torch.tensor(f["R_wc"], dtype=torch.float32, device=device),
                     t=torch.tensor(f["t"], dtype=torch.float32, device=device))
                for f in frames]

    return views(meta["frames"]), views(meta["val_frames"]), K, meta["height"], meta["width"]


def init_gaussians(n, device, seed=0):
    """The handout's init: a random cloud in [-1.5, 1.5]^3 of small, gray, faint blobs."""
    g = torch.Generator().manual_seed(seed)
    quat = torch.zeros(n, 4)
    quat[:, 0] = 1.0
    params = {
        "mu": (torch.rand(n, 3, generator=g) * 2 - 1) * 1.5,
        "log_s": torch.log(0.08 * torch.ones(n, 3)),
        "quat": quat,
        "color": torch.zeros(n, 3),
        "op_raw": torch.full((n,), -2.0),
    }
    return {k: v.to(device).requires_grad_() for k, v in params.items()}


def render_view(p, R, t, K, H, W):
    Sig3 = covariance_3d(p["log_s"].exp().clamp_min(MIN_SCALE), p["quat"])
    mu2, Sig2, depth = project_gaussian(p["mu"], Sig3, R, t, K)
    vis = depth > NEAR
    Sig2 = Sig2[vis] + DILATION * torch.eye(2, device=Sig2.device)
    order = torch.argsort(depth[vis])                             # nearest first
    return render(mu2[vis], Sig2, p["color"][vis].sigmoid(), p["op_raw"][vis].sigmoid(),
                  order, H, W)


def fit(train, K, H, W, n, iters=1500, lr=1e-2, device="cpu", seed=0,
        densify_cfg=None, budget=None, log_every=100):
    p = init_gaussians(n, device, seed)
    opt = torch.optim.Adam(p.values(), lr=lr)
    pick = torch.Generator().manual_seed(seed)
    grad_sum, grad_steps = torch.zeros(n, device=device), 0
    running = []
    for step in range(1, iters + 1):
        cam = train[int(torch.randint(len(train), (1,), generator=pick))]
        img = render_view(p, cam["R"], cam["t"], K, H, W)
        loss = ((img - cam["image"]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if densify_cfg:
            grad_sum += p["mu"].grad.norm(dim=-1)   # world-space position gradient
            grad_steps += 1
        opt.step()
        running.append(loss.item())

        if densify_cfg and step % densify_cfg.every == 0 and step <= densify_cfg.until * iters:
            p, stats = densify(p, opt, grad_sum / grad_steps, budget, densify_cfg.size_threshold,
                               lambda q: quaternion_to_rotation(q["quat"]), densify_cfg)
            grad_sum, grad_steps = torch.zeros(stats["count"], device=device), 0
            print(f"  densify @ {step}: {stats}")
        if (densify_cfg and densify_cfg.reset_every and step % densify_cfg.reset_every == 0
                and step < densify_cfg.until * iters):
            reset_opacity(p, opt, densify_cfg.reset_opacity)
            print(f"  opacity reset @ {step}")
        if step % log_every == 0:
            mse = sum(running) / len(running)
            running = []
            print(f"step {step:5d}  train PSNR (last {log_every} views) {psnr(mse):.2f} dB  "
                  f"N={p['mu'].shape[0]}")
    return p


@torch.no_grad()
def evaluate(p, views, K, H, W):
    """Per-view PSNR and renders. The mean of per-view PSNRs is what we report."""
    out = []
    for v in views:
        img = render_view(p, v["R"], v["t"], K, H, W)
        out.append((v, img, psnr(((img - v["image"]) ** 2).mean().item())))
    return out


@torch.no_grad()
def render_orbit(p, K, H, W, frames=36, elevation=25.0, radius=4.0):
    """A full turn at an elevation between the training orbits."""
    e = math.radians(elevation)
    imgs = []
    for i in range(frames):
        a = 2 * math.pi * i / frames
        center = (radius * math.cos(e) * math.sin(a), radius * math.sin(e), radius * math.cos(e) * math.cos(a))
        R, t = (x.to(K.device) for x in look_at(center))
        imgs.append(render_view(p, R, t, K, H, W).cpu())
    return imgs


def save_outputs(p, train, val, K, H, W, out):
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    tr, va = evaluate(p, train, K, H, W), evaluate(p, val, K, H, W)
    res = dict(count=p["mu"].shape[0],
               train_psnr=sum(r[2] for r in tr) / len(tr),
               val_psnr=sum(r[2] for r in va) / len(va),
               val_per_view={r[0]["name"]: r[2] for r in va})
    for tag, rows in (("train", tr), ("val", va)):
        for v, img, _ in rows:
            write_image(torch.cat([v["image"], img], 1), out / f"{tag}_{Path(v['name']).stem}.png")
    frames = render_orbit(p, K, H, W)
    pil = [Image.fromarray((f.clamp(0, 1) * 255).byte().numpy()) for f in frames]
    pil[0].save(out / "orbit.gif", save_all=True, append_images=pil[1:], duration=80, loop=0)
    for i in (0, 9, 18, 27):
        pil[i].save(out / f"orbit_{i:02d}.png")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"N={res['count']}  train PSNR {res['train_psnr']:.2f} dB  "
          f"held-out PSNR {res['val_psnr']:.2f} dB  -> {out}")
    return res


def main():
    ap = argparse.ArgumentParser(description="P7-P9: 3D Gaussian reconstruction")
    ap.add_argument("--data", default="data/spheres")
    ap.add_argument("--n", type=int, default=4000, help="Gaussians (with --densify: the budget)")
    ap.add_argument("--densify", action="store_true")
    ap.add_argument("--start-n", type=int, help="starting count with --densify (default n/4)")
    ap.add_argument("--grad-threshold", type=float, default=1e-4)
    ap.add_argument("--size-threshold", type=float, default=0.1, help="world units")
    ap.add_argument("--reset-every", type=int, default=400, help="opacity reset period (0 = off)")
    ap.add_argument("--prune-opacity", type=float, default=DensifyConfig.prune_opacity)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", help="output directory (default results/3d/plain|densify)")
    ap.add_argument("--device", default=get_device())
    args = ap.parse_args()

    out = Path(args.out or f"results/3d/{'densify' if args.densify else 'plain'}")
    train, val, K, H, W = load_scene(args.data, args.device)
    if args.eval_only:
        p = {k: v.to(args.device) for k, v in torch.load(out / "params.pt").items()}
    else:
        t0 = time.time()
        if args.densify:
            cfg = DensifyConfig(every=100, grad_threshold=args.grad_threshold,
                                size_threshold=args.size_threshold, until=0.8,
                                reset_every=args.reset_every, prune_opacity=args.prune_opacity)
            p = fit(train, K, H, W, args.start_n or args.n // 4, args.iters, device=args.device,
                    densify_cfg=cfg, budget=args.n)
        else:
            p = fit(train, K, H, W, args.n, args.iters, device=args.device)
        print(f"trained in {time.time() - t0:.0f}s")
        out.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in p.items()}, out / "params.pt")
    save_outputs(p, train, val, K, H, W, out)


if __name__ == "__main__":
    main()
