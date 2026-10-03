"""P4/P8: adaptive density control — clone, split, and prune Gaussians mid-training.

Works for 2D and 3D alike: the caller supplies the size limit in its own units
and a function giving each Gaussian's rotation matrix (from theta or a quaternion).

Parameters live in a dict of leaf tensors whose first dimension is the Gaussian
index. A pass builds new tensors (survivors, then clones, then split children)
and swaps them into the Adam optimizer, carrying each survivor's moment
estimates and starting every new Gaussian from zeroed moments.
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class DensifyConfig:
    every: int = 200  # run a pass every `every` optimization steps
    # densify Gaussian i if g_i > grad_threshold (g_i in NDC units).
    # The paper's 2e-4 but this worked better
    grad_threshold: float = 1e-4
    size_threshold: float = (
        0.02  # clone if max scale <= this (2D: fraction of image width; 3D: world units)
    )
    split_scale: float = 1.6  # each split child gets (parent scale / split_scale)
    prune_opacity: float = 0.005  # remove Gaussian i if its opacity < this
    until: float = (
        0.75  # no passes after this fraction of training, so new Gaussians can settle
    )
    # every reset_every steps (0 = never), cap all opacities at reset_opacity. Gaussians the
    # renders need regrow; ones that only add haze stay faint and fall to the prune threshold.
    reset_every: int = 0
    reset_opacity: float = 0.01


def split_children(mu, log_s, R, split_scale):
    """Two children per parent, centers sampled from the parent's own Gaussian
    (mu + R S z, z ~ N(0, I)), each with scale shrunk by split_scale.
    mu, log_s: (n, d), R: (n, d, d) for d = 2 or 3."""
    RS = R * log_s.exp()[:, None, :]
    z = torch.randn(2, *mu.shape, device=mu.device)
    child_mu = mu + torch.einsum("nij,knj->kni", RS, z)  # (2, n, d)
    return child_mu.reshape(-1, mu.shape[-1]), (log_s - math.log(split_scale)).repeat(
        2, 1
    )


def densify(params, opt, grad_mag, budget, size_limit, rotation, cfg):
    """One densification pass. grad_mag: (N,) mean position-gradient norm since
    the last pass; size_limit: clone/split boundary on max scale, in the same
    units as exp(log_s); rotation: params dict -> (n, d, d) rotation matrices.
    Returns the new params dict (opt is updated in place) and a summary."""
    with torch.no_grad():
        keep = params["op_raw"].sigmoid() >= cfg.prune_opacity
        cand = (grad_mag > cfg.grad_threshold) & keep
        # each clone or split adds exactly one Gaussian net, so room = candidates allowed
        room = max(budget - int(keep.sum()), 0)
        if int(cand.sum()) > room:
            top = grad_mag.masked_fill(~cand, -math.inf).topk(room).indices
            cand = torch.zeros_like(cand)
            cand[top] = True

        small = params["log_s"].exp().amax(-1) <= size_limit
        clone, split = cand & small, cand & ~small
        survivors = keep & ~split  # split parents are replaced by their children

        parents = {k: v[split] for k, v in params.items()}
        child_mu, child_log_s = split_children(
            parents["mu"], parents["log_s"], rotation(parents), cfg.split_scale
        )
        new_rows = {
            k: torch.cat([v[clone], parents[k].repeat(2, *[1] * (v.dim() - 1))])
            for k, v in params.items()
        }
        n_clone = int(clone.sum())
        new_rows["mu"][n_clone:] = child_mu
        new_rows["log_s"][n_clone:] = child_log_s

        n_added = new_rows["mu"].shape[0]
        new_params = {}
        for k, old in params.items():
            new = torch.cat([old[survivors], new_rows[k]]).requires_grad_()
            _swap_in_optimizer(opt, old, new, survivors, n_added)
            new_params[k] = new

    stats = dict(
        pruned=int((~keep).sum()),
        cloned=n_clone,
        split=int(split.sum()),
        count=new_params["mu"].shape[0],
    )
    return new_params, stats


def reset_opacity(params, opt, value):
    """Cap every opacity at `value` and clear Adam's opacity moments, so the
    regrowth is driven by fresh gradients rather than stale momentum."""
    with torch.no_grad():
        params["op_raw"].clamp_(max=math.log(value / (1 - value)))
    state = opt.state.get(params["op_raw"])
    if state:
        state["exp_avg"].zero_()
        state["exp_avg_sq"].zero_()


def _swap_in_optimizer(opt, old, new, survivors, n_added):
    """Replace `old` with `new` in Adam, keeping survivors' moments and zeroing new rows."""
    state = opt.state.pop(old, None)
    if state:
        for key in ("exp_avg", "exp_avg_sq"):
            m = state[key]
            state[key] = torch.cat([m[survivors], m.new_zeros((n_added, *m.shape[1:]))])
        opt.state[new] = state
    for group in opt.param_groups:
        group["params"] = [new if q is old else q for q in group["params"]]
