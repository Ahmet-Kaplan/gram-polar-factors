"""
Decisive test: is the conditioning of the momentum buffer real, or is it the
precision in which the buffer was formed?

An audit of the published measurement showed that sigma_min/sigma_max is below
the unit roundoff of float32 for every momentum matrix we collected, which is
consistent with two very different worlds: the matrices really are that
ill-conditioned, or the number is an artefact of float32 storage.  This script
separates them with a 2x2 design

    gradient precision  x  momentum-accumulation precision

holding everything else fixed (same weights each step, same batches, same seed,
same beta).  Every buffer is measured by a float64 SVD, so the *measurement* is
never the limiting factor; only the *formation* of the buffer is varied.

  grad fp64 / buffer fp64   the true conditioning of the momentum matrix
  grad fp64 / buffer fp32   isolates the accumulation precision
  grad fp32 / buffer fp64   isolates the gradient precision
  grad fp32 / buffer fp32   what a normal training run measures

`--sweep` repeats the fp64/fp64 and fp32/fp32 cells over steps, width and beta,
so that the answer is not a property of one configuration.

Run:  python exp_true_conditioning.py            # main 2x2 cell
      python exp_true_conditioning.py --sweep    # robustness sweep
Out:  ../results/true_conditioning.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import data as dm
from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


@torch.no_grad()
def _spectrum(B: torch.Tensor) -> dict:
    """kappa and effective rank of B, measured in float64 (always)."""
    s = torch.linalg.svdvals(B.detach().double())
    rel = (s / s[0]).numpy()
    return {"kappa": float(rel[-1]),
            "rank_1e3": int((rel > 1e-3).sum()),
            "rank_1e6": int((rel > 1e-6).sum()),
            "stable_rank": float(rel.sum() ** 2 / (rel ** 2).sum()),
            "k": int(rel.size)}


def _hidden(name: str) -> bool:
    return "wte" not in name and "wpe" not in name


def run_cell(steps, width, depth, heads, block, batch, beta, seed, data) -> list:
    """Collect four buffers (grad dtype x buffer dtype) and measure each."""
    train, _, vocab = data
    torch.manual_seed(seed)
    m64 = GPT(GPTConfig(vocab_size=vocab, n_layer=depth, n_head=heads,
                        n_embd=width, block_size=block)).to(torch.float64)
    m32 = GPT(GPTConfig(vocab_size=vocab, n_layer=depth, n_head=heads,
                        n_embd=width, block_size=block)).to(torch.float32)
    m32.load_state_dict({k: v.float() for k, v in m64.state_dict().items()})
    gen = torch.Generator().manual_seed(seed + 1)

    bufs = {(g, b): {} for g in ("fp64", "fp32") for b in ("fp64", "fp32")}
    for _ in range(steps):
        x, y = dm.get_batch(train, block, batch, torch.device("cpu"), gen)
        grads = {}
        for tag, model in (("fp64", m64), ("fp32", m32)):
            _, loss = model(x, y)
            model.zero_grad()
            loss.backward()
            grads[tag] = {n: (p.grad.detach().clone() if p.grad is not None else None)
                          for n, p in model.named_parameters()}
        # keep both models on the identical trajectory: the fp64 weights are the
        # reference, the fp32 model is a rounded copy of them
        with torch.no_grad():
            for (n, p64), (_, p32) in zip(m64.named_parameters(), m32.named_parameters()):
                if p64.grad is not None:
                    p64 -= 0.01 * p64.grad
                    p32.copy_(p64.float())
        for g in ("fp64", "fp32"):
            for n, G in grads[g].items():
                if G is None or G.ndim != 2 or not _hidden(n):
                    continue
                X = (G.T if G.shape[0] > G.shape[1] else G).double()
                for b in ("fp64", "fp32"):
                    buf = bufs[(g, b)].setdefault(n, torch.zeros_like(X))
                    buf.mul_(beta).add_(X, alpha=1 - beta)
                    if b == "fp32":
                        buf.copy_(buf.float().double())   # round as fp32 would

    rows = []
    for g in ("fp64", "fp32"):
        for b in ("fp64", "fp32"):
            specs = [_spectrum(B) for B in bufs[(g, b)].values()]
            kap = np.array([s["kappa"] for s in specs], float)
            rows.append({"grad_precision": g, "buffer_precision": b,
                         "n_matrices": len(specs),
                         "kappa_median": float(np.nanmedian(kap)),
                         "kappa_min": float(np.nanmin(kap)),
                         "kappa_max": float(np.nanmax(kap)),
                         "rank_1e3_median": float(np.median([s["rank_1e3"] for s in specs])),
                         "rank_1e6_median": float(np.median([s["rank_1e6"] for s in specs])),
                         "stable_rank_median": float(np.median([s["stable_rank"] for s in specs])),
                         "k_median": float(np.median([s["k"] for s in specs]))})
    return rows


def _print(rows, header=""):
    if header:
        print(f"\n{header}")
    print(f"{'grad':6s} {'buffer':7s} {'n':>3s} {'median k':>10s} {'min':>10s} "
          f"{'max':>10s} {'r@1e-3':>7s} {'r@1e-6':>7s} {'stable':>7s} {'k':>4s}")
    for r in rows:
        print(f"{r['grad_precision']:6s} {r['buffer_precision']:7s} "
              f"{r['n_matrices']:3d} {r['kappa_median']:10.2e} "
              f"{r['kappa_min']:10.2e} {r['kappa_max']:10.2e} "
              f"{r['rank_1e3_median']:7.0f} {r['rank_1e6_median']:7.0f} "
              f"{r['stable_rank_median']:7.1f} {r['k_median']:4.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="results/true_conditioning.json")
    args = ap.parse_args()

    data = dm.prepare_shakespeare()
    t0 = time.time()
    base = dict(depth=args.depth, heads=args.heads, block=args.block,
                batch=args.batch, seed=args.seed)

    if not args.sweep:
        rows = run_cell(args.steps, args.width, args.depth, args.heads,
                        args.block, args.batch, args.beta, args.seed, data)
        _print(rows, f"grad/buffer 2x2  (width {args.width}, depth {args.depth}, "
                     f"steps {args.steps}, beta {args.beta})")
        out = {"config": vars(args), "beta": args.beta,
               "rows": rows, "wall_s": time.time() - t0}
        op = ROOT / args.out
        op.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {op}")
        return

    sweep = []
    grid = [(s, w, d, b)
            for s in (20, 100, 300)
            for w in (192, 384)
            for d in (2, 4)
            for b in (0.90, 0.95, 0.99)]
    for steps, width, depth, beta in grid:
        rows = run_cell(steps, width, depth, args.heads, args.block,
                        args.batch, beta, args.seed, data)
        get = {(r["grad_precision"], r["buffer_precision"]): r for r in rows}
        true = get[("fp64", "fp64")]
        meas = get[("fp32", "fp32")]
        sweep.append({"steps": steps, "width": width, "depth": depth, "beta": beta,
                      "kappa_true": true["kappa_median"],
                      "kappa_measured_fp32": meas["kappa_median"],
                      "ratio": meas["kappa_median"] / true["kappa_median"],
                      "rank_1e3_median": true["rank_1e3_median"],
                      "stable_rank_median": true["stable_rank_median"],
                      "k_median": true["k_median"]})
        print(f"  steps {steps:4d} w{width:4d} d{depth} beta {beta:.2f}  "
              f"true {true['kappa_median']:.2e}  fp32 {meas['kappa_median']:.2e}  "
              f"rank@1e-3 {true['rank_1e3_median']:.0f}/{true['k_median']:.0f}",
              flush=True)

    kt = np.array([r["kappa_true"] for r in sweep])
    print(f"\n{len(sweep)} configurations in {time.time() - t0:.0f}s")
    print(f"  true kappa: median {np.median(kt):.2e}  "
          f"range [{kt.min():.2e}, {kt.max():.2e}]  "
          f"spread {np.log10(kt.max() / kt.min()):.1f} decades")
    print(f"  fp32-measured / true: median "
          f"{np.median([r['ratio'] for r in sweep]):.2f}x  "
          f"max {max(r['ratio'] for r in sweep):.2f}x")
    out = {"config": vars(args), "sweep": sweep, "wall_s": time.time() - t0,
           "summary": {"n_configs": len(sweep),
                       "kappa_true_median": float(np.median(kt)),
                       "kappa_true_min": float(kt.min()),
                       "kappa_true_max": float(kt.max()),
                       "log10_spread": float(np.log10(kt.max() / kt.min())),
                       "fp32_over_true_median": float(np.median([r["ratio"] for r in sweep])),
                       "fp32_over_true_max": float(max(r["ratio"] for r in sweep))}}
    (RESULTS / "true_conditioning.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'true_conditioning.json'}")


if __name__ == "__main__":
    main()
