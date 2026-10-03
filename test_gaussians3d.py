import json
import math

import torch

from gaussians3d import covariance_3d, look_at, project_gaussian, quaternion_to_rotation


def test_rotation_is_orthonormal_and_ignores_length():
    torch.manual_seed(0)
    q = torch.randn(50, 4)
    R = quaternion_to_rotation(q)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(50, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(50), atol=1e-5)
    assert torch.allclose(quaternion_to_rotation(3.7 * q), R, atol=1e-6)


def test_known_rotations():
    assert torch.allclose(quaternion_to_rotation(torch.tensor([[1.0, 0, 0, 0]]))[0], torch.eye(3))
    h = math.sqrt(0.5)   # 90 degrees about z maps x -> y
    R = quaternion_to_rotation(torch.tensor([[h, 0, 0, h]]))[0]
    assert torch.allclose(R @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1, 0]), atol=1e-6)


def test_covariance_eigenvalues_are_squared_scales():
    torch.manual_seed(1)
    scale = torch.rand(20, 3) + 0.1
    S = covariance_3d(scale, torch.randn(20, 4))
    assert torch.allclose(S, S.transpose(-1, -2), atol=1e-6)
    assert torch.allclose(torch.linalg.eigvalsh(S), (scale ** 2).sort(-1).values, rtol=1e-4)


def _camera():
    R, t = look_at((1.0, 0.8, 3.5))
    K = torch.tensor([[193.0, 0, 80], [0, 193.0, 80], [0, 0, 1]])
    return R, t, K


def test_mean_projects_with_pinhole():
    R, t, K = _camera()
    mu2, _, depth = project_gaussian(torch.zeros(1, 3), torch.eye(3)[None] * 1e-3, R, t, K)
    assert torch.allclose(mu2[0], torch.tensor([80.0, 80.0]), atol=1e-3)   # target -> image center
    assert torch.allclose(depth, torch.tensor([math.sqrt(1 + 0.64 + 12.25)]), atol=1e-4)


def test_jacobian_matches_autograd():
    R, t, K = _camera()
    mu = torch.tensor([0.3, -0.2, 0.4])

    def pix(p):
        c = R @ p + t
        return torch.stack([K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]])

    Jw = torch.autograd.functional.jacobian(pix, mu)            # d pixel / d world = J R
    Sigma = torch.diag(torch.tensor([0.04, 0.01, 0.09]))
    _, Sig2, _ = project_gaussian(mu[None], Sigma[None], R, t, K)
    assert torch.allclose(Sig2[0], Jw @ Sigma @ Jw.T, rtol=1e-4)


def test_projected_covariance_matches_samples():
    """A small Gaussian is nearly linear under projection, so the covariance of
    projected samples should match the Jacobian approximation."""
    torch.manual_seed(2)
    R, t, K = _camera()
    mu = torch.tensor([[0.2, 0.1, -0.3]])
    Sigma = covariance_3d(torch.tensor([[0.05, 0.02, 0.01]]), torch.randn(1, 4))
    _, Sig2, _ = project_gaussian(mu, Sigma, R, t, K)
    pts = torch.distributions.MultivariateNormal(mu[0].double(), Sigma[0].double()).sample((200000,))
    cam = pts @ R.double().T + t.double()
    uv = torch.stack([K[0, 0] * cam[:, 0] / cam[:, 2], K[1, 1] * cam[:, 1] / cam[:, 2]], -1)
    assert torch.allclose(torch.cov(uv.T).float(), Sig2[0], rtol=0.03)


def test_look_at_reproduces_dataset_cameras():
    cams = json.load(open("data/spheres/cameras.json"))
    for f in cams["frames"] + cams["val_frames"]:
        R, t = torch.tensor(f["R_wc"]), torch.tensor(f["t"])
        R2, t2 = look_at(-R.T @ t)
        assert torch.allclose(R2, R, atol=1e-4) and torch.allclose(t2, t, atol=1e-4)
