"""
Figure 4: ablations.

(a) Sensitivity to the subsample ratio rho.
(b) Which selection rule matters (row norm vs curvature proxy vs uniform random).
(c) Which orthogonalizer is admissible (Gram vs SVD vs Newton-Schulz), at both
    full rank and subsampled.
(d) Error trajectories: final loss against wall-clock time for the ablation set.

Data: results/ablation.json (falls back to results/main_bench.json for the
curvature-proxy baseline).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
ABL_FILES = [ROOT / "results" / "ablation.json", ROOT / "results" / "ablation2.json"]
MAIN = ROOT / "results" / "main_bench.json"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 7.0, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "lines.linewidth": 1.6,
})
COL = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#0072B2", "#8C8C8C"]


def load():
    runs = []
    for f in ABL_FILES:
        if f.exists():
            runs.extend(json.loads(f.read_text()))
    main = json.loads(MAIN.read_text()) if MAIN.exists() else []
    for r in main:
        if r["config"]["name"].startswith("csmuon_3e-4_s0"):
            r = dict(r)
            r["config"] = dict(r["config"])
            r["config"]["name"] = "abl_curvature_default"
            runs.append(r)
    return runs


def final(r):
    return r.get("final_val_loss")


def main():
    runs = load()
    if not runs:
        print("no ablation data yet")
        return
    by = {r["config"]["name"]: r for r in runs}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05))

    # (a) rho sweep
    ax = axes[0]
    rho_pts = [("abl_rho005", 0.05, "rho_sweep"),
               ("abl_curvature_default", 0.15, "rho_sweep"),
               ("abl_rho030", 0.30, "rho_sweep"),
               ("abl_full_svd", 1.00, "full")]
    xs, ys = [], []
    for name, rho, kind in rho_pts:
        if name in by and final(by[name]) is not None:
            xs.append(rho); ys.append(final(by[name]))
    if xs:
        ax.plot(xs, ys, "o-", color=COL[0], ms=4)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"subsample ratio $\rho$")
    ax.set_ylabel("final validation loss")
    ax.set_title(r"(a) sensitivity to $\rho$")

    # (b) selection rule
    ax = axes[1]
    rules = [("abl_curvature_default", "curvature\n(ours)"),
             ("abl_selnorm", "row\nnorm"),
             ("abl_selrandom", "uniform\nrandom")]
    labels, vals, cols = [], [], []
    for name, lab in rules:
        if name in by and final(by[name]) is not None:
            labels.append(lab); vals.append(final(by[name]))
            cols.append(COL[4] if "ours" in lab else COL[6])
    if vals:
        bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="white",
                      linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.001, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        lo = min(vals)
        ax.set_ylim(lo * 0.995, max(vals) * 1.01)
    ax.set_ylabel("final validation loss")
    ax.set_title("(b) selection rule")

    # (c) orthogonalizer
    ax = axes[2]
    orths = [("abl_curvature_default", "Gram\n(rho=.15)"),
             ("abl_nsorth", "NS\n(rho=.15)"),
             ("abl_gramfull", "Gram\n(full)"),
             ("abl_full_svd", "SVD\n(full)")]
    labels, vals, cols = [], [], []
    for name, lab in orths:
        if name in by and final(by[name]) is not None:
            labels.append(lab); vals.append(final(by[name]))
            cols.append(COL[0] if "Gram" in lab and ".15" in lab else COL[6])
    if vals:
        bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="white",
                      linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.001, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(min(vals) * 0.995, max(vals) * 1.01)
    ax.set_ylabel("final validation loss")
    ax.set_title("(c) orthogonalizer")

    # (d) loss vs wall clock for every ablation run
    ax = axes[3]
    for i, r in enumerate(runs):
        pts = [(h["elapsed"], h["val_loss"]) for h in r["history"]
               if "val_loss" in h and np.isfinite(h["val_loss"])]
        if not pts:
            continue
        t = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
        ax.plot(t, v, "-", lw=0.9, color=COL[i % len(COL)], alpha=0.85,
                label=r["config"]["name"].replace("abl_", ""))
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("validation loss")
    ax.set_title("(d) all ablation runs")
    ax.set_xscale("log")
    ax.legend(fontsize=5.2, ncol=1, loc="upper right")

    fig.tight_layout()
    out = FIG / "fig4_ablations"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    print(f"wrote {out}.pdf/.png")
    for name in sorted(by):
        f = final(by[name])
        if f is not None:
            print(f"  {name:26s} final={f:.4f} tok/s={by[name]['tokens_per_s']:,.0f}")


if __name__ == "__main__":
    main()
