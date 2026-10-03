import math

import torch

from gaussians import covariance_2d, gaussian_weight, pixel_grid


def test_covariance_is_symmetric_psd():
    torch.manual_seed(0)
    scale = torch.rand(100, 2) * 5 + 0.1
    theta = torch.rand(100) * 2 * math.pi
    S = covariance_2d(scale, theta)
    assert torch.allclose(S, S.transpose(-1, -2), atol=1e-5)
    eig = torch.linalg.eigvalsh(S)
    assert (eig > 0).all()
    # eigenvalues are exactly the squared scales
    assert torch.allclose(eig, (scale ** 2).sort(-1).values, rtol=1e-4)


def test_rotation_90_swaps_axes():
    S = covariance_2d(torch.tensor([[3.0, 1.0]]), torch.tensor([math.pi / 2]))
    assert torch.allclose(S[0], torch.diag(torch.tensor([1.0, 9.0])), atol=1e-5)


def test_weight_peak_and_isotropic_falloff():
    s = 2.0
    S = covariance_2d(torch.tensor([[s, s]]), torch.tensor([0.7]))  # rotation irrelevant
    mu = torch.tensor([[5.0, 5.0]])
    xy = torch.tensor([[5.0, 5.0], [7.0, 5.0], [5.0, 1.0]])
    w = gaussian_weight(xy, mu, S)[:, 0]
    r2 = torch.tensor([0.0, 4.0, 16.0])
    assert torch.allclose(w, torch.exp(-r2 / (2 * s * s)))


def test_weight_matches_linalg_inv():
    torch.manual_seed(1)
    N, P = 7, 50
    S = covariance_2d(torch.rand(N, 2) * 3 + 0.5, torch.rand(N) * 6)
    mu, xy = torch.rand(N, 2) * 10, torch.rand(P, 2) * 10
    d = xy[:, None] - mu[None]
    ref = torch.exp(-0.5 * torch.einsum("pni,nij,pnj->pn", d, torch.linalg.inv(S), d))
    assert torch.allclose(gaussian_weight(xy, mu, S), ref, atol=1e-5)


def test_gradients_flow_to_all_params():
    params = [torch.rand(3, 2, dtype=torch.double, requires_grad=True) + 0.5,  # scale
              torch.rand(3, dtype=torch.double, requires_grad=True),            # theta
              torch.rand(3, 2, dtype=torch.double, requires_grad=True) * 4]     # mu
    xy = pixel_grid(4, 4).double()
    f = lambda sc, th, mu: gaussian_weight(xy, mu, covariance_2d(sc, th))
    assert torch.autograd.gradcheck(f, params)


def test_pixel_grid_layout():
    g = pixel_grid(2, 3)
    assert g.shape == (6, 2)
    assert torch.equal(g[1], torch.tensor([1.5, 0.5]))  # second pixel moves in x
    assert torch.equal(g[3], torch.tensor([0.5, 1.5]))  # next row moves in y
