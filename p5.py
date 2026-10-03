"""P5: PSNR vs. number of Gaussians for each target image.

Runs every (image, N) fit that is not already in results/p5.json, saving each
render and appending its PSNR as it finishes, so an interrupted sweep resumes
where it stopped. Then plots PSNR against N, one line per image.

    python p5.py              # fixed-count sweep (the P5 requirement)
    python p5.py --densify    # same sweep with P4 densification, as a second set of curves
    python p5.py --plot-only
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from densify import DensifyConfig
from fit2d import fit
from utils import get_device, load_image, psnr, write_image

IMAGES = ["coffee", "astronaut", "cat"]
COUNTS = [256, 1024, 4096]
RESULTS = Path("results/p5.json")
# categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light mode)
COLORS = {"coffee": "#2a78d6", "astronaut": "#eb6834", "cat": "#1baf7a"}


def load_results():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def run_sweep(mode, size, steps, device):
    results = load_results()
    for name in IMAGES:
        target = load_image(f"data/{name}.png", size)
        for n in COUNTS:
            key = f"{mode}/{name}/{n}"
            if key in results:
                continue
            print(f"=== {key}")
            t0 = time.time()
            if mode == "densify":
                p, img, _ = fit(target, n // 4, steps, device=device,
                                densify_cfg=DensifyConfig(), budget=n)
            else:
                p, img, _ = fit(target, n, steps, device=device)
            write_image(img, f"results/{name}_{n}{'_densify' if mode == 'densify' else ''}.png")
            results[key] = dict(psnr=psnr(((img.cpu() - target) ** 2).mean().item()),
                                count=p["mu"].shape[0], seconds=round(time.time() - t0))
            RESULTS.write_text(json.dumps(results, indent=2))
            print(f"=== {key}: {results[key]}")


def plot(path="results/psnr_vs_n.png"):
    results = load_results()
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    for name in IMAGES:
        for mode, style in (("fixed", "-"), ("densify", "--")):
            pts = sorted((r["count"], r["psnr"]) for k, r in results.items()
                         if k.startswith(f"{mode}/{name}/"))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, ls=style, color=COLORS[name], lw=2, marker="o", ms=6,
                    markeredgecolor="white", markeredgewidth=1.5)
            if mode == "fixed":   # direct label at the right end of each fixed curve
                ax.annotate(name, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=9, color="#52514e")
    ax.set_xscale("log", base=2)
    ax.set_xticks(COUNTS, [str(n) for n in COUNTS])
    ax.set_xlim(COUNTS[0] / 1.3, COUNTS[-1] * 1.9)
    ax.set_xlabel("Number of Gaussians N  (model size ≈ 9N floats)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("2D fit quality vs. Gaussian count (128×128, 2000 steps)", fontsize=10)
    ax.grid(True, color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e")
    # images are direct-labeled, so the legend only explains line style
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="#52514e", lw=2, ls="-", label="fixed count"),
                       Line2D([], [], color="#52514e", lw=2, ls="--",
                              label="densified (plotted at final count)")],
              frameon=False, fontsize=8, loc="lower right", handlelength=3)
    fig.tight_layout()
    fig.savefig(path)
    print(f"plot -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--densify", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default=get_device())
    args = ap.parse_args()
    if not args.plot_only:
        run_sweep("densify" if args.densify else "fixed", args.size, args.steps, args.device)
    plot()


if __name__ == "__main__":
    main()
