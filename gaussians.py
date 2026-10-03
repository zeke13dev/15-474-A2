"""P1: a single 2D Gaussian primitive.

A Gaussian is stored as a center mu, a per-axis scale, and a rotation theta.
The covariance is built from scale and rotation, never optimized directly,
so it is a valid (symmetric positive semi-definite) matrix by construction.
"""

import torch


def rotation_2d(theta: torch.Tensor) -> torch.Tensor:
    """theta: (N,) radians -> (N, 2, 2) counter-clockwise rotation matrices."""
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1),
                        torch.stack([s, c], -1)], -2)


def covariance_2d(scale: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """scale: (N, 2) positive, theta: (N,) -> Sigma = R S S^T R^T, (N, 2, 2)."""
    M = rotation_2d(theta) * scale[:, None, :]   # R @ diag(scale): scales R's columns
    return M @ M.transpose(-1, -2)


def inverse_2x2(Sigma: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of a batch of 2x2 matrices, (N, 2, 2) -> (N, 2, 2)."""
    a, b = Sigma[:, 0, 0], Sigma[:, 0, 1]
    c, d = Sigma[:, 1, 0], Sigma[:, 1, 1]
    det = a * d - b * c
    adj = torch.stack([torch.stack([d, -b], -1),
                       torch.stack([-c, a], -1)], -2)
    return adj / det[:, None, None]


def gaussian_weight(xy: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n)).

    xy: (P, 2) pixel coords, mu: (N, 2), Sigma: (N, 2, 2) -> (P, N).
    Unnormalized: the weight is 1 at the center, so opacity alone sets peak alpha.
    """
    d = xy[:, None, :] - mu[None, :, :]            # (P, N, 2)
    inv = inverse_2x2(Sigma)                       # (N, 2, 2)
    # Mahalanobis distance d^T inv d, expanded using inv's symmetry
    m = (inv[:, 0, 0] * d[..., 0] ** 2
         + 2 * inv[:, 0, 1] * d[..., 0] * d[..., 1]
         + inv[:, 1, 1] * d[..., 1] ** 2)          # (P, N)
    return torch.exp(-0.5 * m)


def pixel_grid(H: int, W: int, device=None) -> torch.Tensor:
    """(H*W, 2) pixel-center coordinates (x, y), row-major to match reshape(H, W)."""
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32) + 0.5,
                            torch.arange(W, device=device, dtype=torch.float32) + 0.5,
                            indexing="ij")
    return torch.stack([xs, ys], -1).reshape(-1, 2)
