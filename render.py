"""P2: differentiable front-to-back alpha compositing of 2D Gaussians."""

import math

import torch

from gaussians import gaussian_weight, inverse_2x2, pixel_grid

CUTOFF_SIGMA = 3.5   # a Gaussian has no effect beyond 3.5 sigma (weight < 0.0022)


def composite(alpha: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """Front-to-back "over" compositing, vectorized across Gaussians.

    alpha: (..., P, N) per-pixel alpha with the last axis sorted front -> back,
    color: (..., N, 3) in the same order. Returns (..., P, 3) premultiplied over black.

    T_i = prod_{j<i} (1 - alpha_j) is the transmittance left in front of i,
    and C = sum_i T_i alpha_i c_i. An exclusive cumprod gives every T_i at once,
    which is the same arithmetic as the reference loop without N Python steps.
    """
    one_minus = 1 - alpha
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), one_minus[..., :-1]], -1), -1)
    return (T * alpha) @ color


def composite_loop(alpha: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """The handout's per-Gaussian loop; kept as the reference for composite()."""
    C = torch.zeros(alpha.shape[0], 3, dtype=alpha.dtype, device=alpha.device)
    T = torch.ones(alpha.shape[0], dtype=alpha.dtype, device=alpha.device)
    for i in range(alpha.shape[1]):
        a = alpha[:, i]
        C = C + (T * a)[:, None] * color[i]
        T = T * (1 - a)
    return C


def render_dense(mu, Sigma, color, opacity, order, H, W):
    """Every pixel against every Gaussian. Simple, but O(H*W*N); kept as the
    reference that render() must match."""
    xy = pixel_grid(H, W, device=mu.device)                       # (H*W, 2)
    w = gaussian_weight(xy, mu[order], Sigma[order])              # (P, N), sorted
    w = w * (w >= math.exp(-0.5 * CUTOFF_SIGMA ** 2))             # same window as render()
    alpha = opacity[order][None, :] * w                           # (P, N)
    return composite(alpha, color[order]).reshape(H, W, 3)


def render(mu, Sigma, color, opacity, order, H, W, tile=16):
    """mu: (N, 2), Sigma: (N, 2, 2), color: (N, 3), opacity: (N,) in [0, 1],
    order: (N,) indices front -> back. Returns an (H, W, 3) image.

    Tiled: the image is cut into tile x tile blocks, and each block composites
    only the Gaussians whose 3.5-sigma bounding box overlaps it, padded to the
    busiest tile's count K. Work drops from H*W*N to H*W*K, and K is a small
    fraction of N whenever Gaussians are small relative to the image.
    """
    dev = mu.device
    mu, Sigma, color, opacity = mu[order], Sigma[order], color[order], opacity[order]
    N = mu.shape[0]
    ty, tx = -(-H // tile), -(-W // tile)                         # tiles per side, rounded up

    with torch.no_grad():   # which Gaussians touch which tile: selection only, no gradient
        half = CUTOFF_SIGMA * torch.stack([Sigma[:, 0, 0], Sigma[:, 1, 1]], -1).sqrt()
        lo, hi = mu - half, mu + half                             # (N, 2) bounding boxes
        x0 = torch.arange(tx, device=dev) * tile
        y0 = torch.arange(ty, device=dev) * tile
        hit_x = (hi[None, :, 0] > x0[:, None]) & (lo[None, :, 0] < x0[:, None] + tile)  # (tx, N)
        hit_y = (hi[None, :, 1] > y0[:, None]) & (lo[None, :, 1] < y0[:, None] + tile)  # (ty, N)
        hit = (hit_y[:, None] & hit_x[None]).reshape(ty * tx, N)  # (tiles, N)
        K = max(int(hit.sum(1).max()), 1)
        # sorting the masked index list keeps each tile's Gaussians in front-to-back order
        key = torch.where(hit, torch.arange(N, device=dev), N)
        idx = key.sort(1).values[:, :K]                           # (tiles, K)
        valid = idx < N
        idx = idx.clamp(max=N - 1)

    xy = pixel_grid(ty * tile, tx * tile, device=dev)             # padded image, cropped below
    xy = xy.reshape(ty, tile, tx, tile, 2).permute(0, 2, 1, 3, 4).reshape(ty * tx, tile * tile, 2)

    inv = inverse_2x2(Sigma)[idx]                                 # (tiles, K, 2, 2)
    d = xy[:, :, None, :] - mu[idx][:, None, :, :]                # (tiles, P_t, K, 2)
    m = (inv[:, None, :, 0, 0] * d[..., 0] ** 2
         + 2 * inv[:, None, :, 0, 1] * d[..., 0] * d[..., 1]
         + inv[:, None, :, 1, 1] * d[..., 1] ** 2)                # (tiles, P_t, K)
    inside = (m <= CUTOFF_SIGMA ** 2) & valid[:, None, :]
    alpha = torch.where(inside, opacity[idx][:, None, :] * torch.exp(-0.5 * m), 0.0)
    C = composite(alpha, color[idx])                              # (tiles, P_t, 3)

    img = C.reshape(ty, tx, tile, tile, 3).permute(0, 2, 1, 3, 4).reshape(ty * tile, tx * tile, 3)
    return img[:H, :W]
