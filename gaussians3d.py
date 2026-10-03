"""P6: 3D Gaussians and their projection into a pinhole camera.

Cameras follow the dataset's OpenCV convention: X_cam = R_wc @ X_world + t,
x right, y down, z forward; u = fx * x / z + cx, v = fy * y / z + cy.
"""

import torch


def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """q: (N, 4) as (w, x, y, z), any nonzero length -> (N, 3, 3) rotations."""
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def covariance_3d(scale: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """scale: (N, 3) positive, quat: (N, 4) -> Sigma = R S S^T R^T, (N, 3, 3)."""
    M = quaternion_to_rotation(quat) * scale[:, None, :]   # R @ diag(scale)
    return M @ M.transpose(-1, -2)


def project_gaussian(mu3, Sigma3, R_wc, t, K):
    """mu3: (N, 3) world means, Sigma3: (N, 3, 3) world covariances,
    R_wc: (3, 3), t: (3,), K: (3, 3).
    Returns 2D means (N, 2) in pixels, 2D covariances (N, 2, 2), camera depth (N,)."""
    mu_cam = mu3 @ R_wc.T + t                                   # world -> camera
    x, y, z = mu_cam.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mu2 = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)

    # Jacobian of (u, v) with respect to (x, y, z), evaluated at each mean
    zero = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx / z, zero, -fx * x / z ** 2], -1),
                     torch.stack([zero, fy / z, -fy * y / z ** 2], -1)], -2)   # (N, 2, 3)
    Scam = R_wc @ Sigma3 @ R_wc.T                               # covariance in camera frame
    Sig2 = J @ Scam @ J.transpose(-1, -2)
    return mu2, Sig2, z


def look_at(center, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """World-to-camera (R_wc, t) for a camera at `center` looking at `target`,
    in the OpenCV convention (rows of R_wc are camera right, down, forward)."""
    center, target, up = (torch.as_tensor(v, dtype=torch.float32) for v in (center, target, up))
    fwd = target - center
    fwd = fwd / fwd.norm()
    right = torch.linalg.cross(fwd, up)
    right = right / right.norm()
    down = torch.linalg.cross(fwd, right)
    R = torch.stack([right, down, fwd])
    return R, -R @ center
