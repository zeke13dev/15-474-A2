import math

import torch

from densify import DensifyConfig, densify as _densify, reset_opacity
from gaussians import rotation_2d


def make(n, scales, opacities, seed=0):
    torch.manual_seed(seed)
    p = {"mu": torch.rand(n, 2) * 100,
         "log_s": torch.log(torch.tensor(scales, dtype=torch.float32))[:, None].repeat(1, 2),
         "theta": torch.rand(n),
         "color": torch.rand(n, 3),
         "op_raw": torch.logit(torch.tensor(opacities, dtype=torch.float32))}
    p = {k: v.requires_grad_() for k, v in p.items()}
    opt = torch.optim.Adam(p.values(), lr=1e-2)
    sum(v.sum() for v in p.values()).backward()   # populate Adam state
    opt.step()
    return p, opt


CFG = DensifyConfig(grad_threshold=1.0, size_threshold=0.02)   # width 100 -> 2 px


def densify(p, opt, grad, budget, width, cfg):
    return _densify(p, opt, grad, budget, cfg.size_threshold * width,
                    lambda q: rotation_2d(q["theta"]), cfg)


def test_clone_split_prune():
    # 0: small + high grad -> clone, 1: large + high grad -> split,
    # 2: low grad -> untouched, 3: transparent -> pruned even with high grad
    p, opt = make(4, [1.0, 10.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.001])
    grad = torch.tensor([5.0, 5.0, 0.1, 5.0])
    new, stats = densify(p, opt, grad, budget=100, width=100, cfg=CFG)
    assert stats == dict(pruned=1, cloned=1, split=1, count=5)
    # survivors [0, 2], clone of 0, two children of 1
    assert torch.equal(new["mu"][2], p["mu"][0])
    assert torch.allclose(new["log_s"][3:].exp(), p["log_s"][1].exp().expand(2, 2) / CFG.split_scale)
    assert torch.equal(new["color"][3:], p["color"][1].expand(2, 3))


def test_children_sampled_near_parent():
    p, opt = make(1, [5.0], [0.5])
    torch.manual_seed(1)
    new, _ = densify(p, opt, torch.tensor([5.0]), budget=10, width=100, cfg=CFG)
    assert ((new["mu"] - p["mu"][0]).norm(dim=-1) < 5.0 * 5).all()   # within 5 sigma
    assert not torch.equal(new["mu"][0], new["mu"][1])


def test_budget_keeps_highest_gradients():
    p, opt = make(6, [1.0] * 6, [0.5] * 6)
    grad = torch.tensor([2.0, 9.0, 3.0, 8.0, 1.5, 0.0])
    new, stats = densify(p, opt, grad, budget=8, width=100, cfg=CFG)
    assert stats["count"] == 8 and stats["cloned"] == 2
    assert torch.equal(new["mu"][6:], p["mu"][[1, 3]])   # the two largest gradients


def test_optimizer_state_follows_survivors():
    p, opt = make(3, [1.0, 10.0, 1.0], [0.5, 0.5, 0.001])
    old_avg = opt.state[p["mu"]]["exp_avg"].clone()
    new, _ = densify(p, opt, torch.tensor([0.0, 5.0, 0.0]), budget=10, width=100, cfg=CFG)
    st = opt.state[new["mu"]]
    assert st["exp_avg"].shape == new["mu"].shape
    assert torch.equal(st["exp_avg"][0], old_avg[0])          # survivor keeps its moment
    assert (st["exp_avg"][1:] == 0).all()                       # children start fresh
    assert all(q is new[k] for q, k in zip(opt.param_groups[0]["params"], new))
    # the optimizer still runs on the swapped tensors
    sum(v.sum() for v in new.values()).backward()
    opt.step()


def test_reset_opacity_caps_and_clears_momentum():
    p, opt = make(3, [1.0] * 3, [0.9, 0.005, 0.3])
    reset_opacity(p, opt, 0.01)
    op = p["op_raw"].sigmoid()
    assert torch.allclose(op[[0, 2]], torch.tensor([0.01, 0.01]), atol=1e-6)
    assert op[1] < 0.01                                   # already below the cap: untouched
    assert (opt.state[p["op_raw"]]["exp_avg"] == 0).all()
