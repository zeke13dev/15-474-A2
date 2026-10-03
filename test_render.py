import math

import torch

from gaussians import covariance_2d
from render import composite, composite_loop, render, render_dense


def test_vectorized_matches_loop():
    torch.manual_seed(0)
    alpha, color = torch.rand(40, 25), torch.rand(25, 3)
    assert torch.allclose(composite(alpha, color), composite_loop(alpha, color), atol=1e-6)


def test_single_gaussian_center_is_opacity_times_color():
    mu = torch.tensor([[8.5, 8.5]])  # exactly on a pixel center
    S = covariance_2d(torch.tensor([[2.0, 2.0]]), torch.zeros(1))
    img = render(mu, S, torch.tensor([[1.0, 0.5, 0.0]]), torch.tensor([0.8]),
                 torch.arange(1), 16, 16)
    assert torch.allclose(img[8, 8], torch.tensor([0.8, 0.4, 0.0]))
    assert img[0, 0].abs().max() < 1e-3  # far from the center: black background


def test_order_decides_who_is_in_front():
    mu = torch.tensor([[4.5, 4.5], [4.5, 4.5]])
    S = covariance_2d(torch.full((2, 2), 3.0), torch.zeros(2))
    color = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    opacity = torch.ones(2)  # fully opaque at the center
    red_front = render(mu, S, color, opacity, torch.tensor([0, 1]), 9, 9)[4, 4]
    blue_front = render(mu, S, color, opacity, torch.tensor([1, 0]), 9, 9)[4, 4]
    assert torch.allclose(red_front, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(blue_front, torch.tensor([0.0, 0.0, 1.0]))


def test_gradients_reach_every_parameter():
    torch.manual_seed(2)
    N = 4
    p = dict(mu=(torch.rand(N, 2) * 6 + 1), log_s=torch.log(torch.rand(N, 2) + 1),
             theta=torch.rand(N), color=torch.rand(N, 3), opacity=torch.rand(N) * 0.9)
    p = {k: v.double().requires_grad_() for k, v in p.items()}

    def f(mu, log_s, theta, color, opacity):
        S = covariance_2d(log_s.exp(), theta)
        return render(mu, S, color, opacity, torch.arange(N), 8, 8)

    assert torch.autograd.gradcheck(f, tuple(p.values()))


def test_tiled_matches_dense():
    torch.manual_seed(3)
    N, H, W = 300, 37, 50          # not multiples of the tile size, to exercise padding
    mu = torch.rand(N, 2) * torch.tensor([W, H]) * 1.2 - 3   # some centers off-image
    S = covariance_2d(torch.rand(N, 2) * 4 + 0.3, torch.rand(N) * 6)
    color, opacity = torch.rand(N, 3), torch.rand(N)
    order = torch.randperm(N)
    a = render(mu, S, color, opacity, order, H, W)
    b = render_dense(mu, S, color, opacity, order, H, W)
    assert a.shape == (H, W, 3)
    assert torch.allclose(a, b, atol=1e-5)
