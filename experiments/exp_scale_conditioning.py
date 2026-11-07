"""
Does the momentum-matrix conditioning ratio improve with model scale?

This is the single most important open question in the paper. The paper's
premise is that real momentum matrices are ill-conditioned (sigma_min/sigma_max
~ 1e-8) while polynomial orthogonalizers are designed for ~1e-1. If the ratio
improves with width / depth / training time, then the conditioning repair is a
small-model artefact and the premise weakens. If it does not, the repair is
necessary rather than optional.

We therefore train several model sizes briefly and measure, in float64, the
conditioning ratio of the momentum buffer for every 2D hidden matrix, as a
function of width, depth and training step.

Usage:
    python exp_scale_conditioning.py            # default ladder
    python exp_scale_conditioning.py --quick     # fast smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import data as dm
import robust_orth as ro
from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def cond_ratio_f64(X: torch.Tensor) -> float:
    """sigma_min/sigma_max in float64 on CPU (the measurement of record)."""
    Xc = X.detach().float().cpu().double()
    s = torch.linalg.svdvals(Xc)
    s = s[s > 0]
    if s.numel() == 0 or float(s[0]) == 0:
        return float("nan")
    return float(s[-1] / s[0])


@torch.no_grad()
def measure(model, buf: dict, step: int, widths: list[int]) -> list[dict]:
    out = []
    for n, B in buf.items():
        rec = {"name": n, "shape": list(B.shape), "step": step,
               "cond_ratio": cond_ratio_f64(B)}
        # subsampled conditioning at rho = 0.15, row-norm selection
        idx = ro.select_rows(B, 0.15, None, mode="norm")
        rec["cond_ratio_sub015"] = cond_ratio_f64(B[idx])
        idx05 = ro.select_rows(B, 0.05, None, mode="norm")
        rec["cond_ratio_sub005"] = cond_ratio_f64(B[idx05])
        rec["k_full"] = int(B.shape[0])
        rec["k_sub015"] = int(idx.numel())
        out.append(rec)
    return out


def run_one(width: int, depth: int, head: int, steps: int, batch: int,
            block: int, data, device, momentum: float = 0.95,
            measure_every: int = 0) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    gen = torch.Generator().manual_seed(1)
    cfg = GPTConfig(vocab_size=data[2], block_size=block, n_layer=depth,
                    n_head=head, n_embd=width)
    model = GPT(cfg).to(device)
    n_params = model.num_params()

    # plain SGD-with-momentum-free buffers: we only need the momentum matrix,
    # so we accumulate exactly what a Muon-style optimizer would.
    buf: dict[str, torch.Tensor] = {}
    records = []
    t0 = time.time()
    for step in range(steps):
        x, y = dm.get_batch(data[0], block, batch, device, gen)
        _, loss = model(x, y)
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is None or p.ndim != 2:
                continue
            if "wte" in n or "wpe" in n or "lm_head" in n:
                continue
            G = p.grad
            Xg = G.T if G.shape[0] > G.shape[1] else G
            if n not in buf:
                buf[n] = torch.zeros_like(Xg)
            buf[n].mul_(momentum).add_(Xg.float(), alpha=1 - momentum)
        if measure_every and (step + 1) % measure_every == 0:
            records.extend(measure(model, buf, step + 1, [width]))
        # a cheap "training" step so the momentum sees a moving target:
        # plain SGD at a small lr is enough for conditioning statistics.
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-0.01)

    if not records:
        records = measure(model, buf, steps, [width])
    ratios = [r["cond_ratio"] for r in records if math.isfinite(r["cond_ratio"])]
    subs = [r["cond_ratio_sub015"] for r in records
            if math.isfinite(r["cond_ratio_sub015"])]
    subs05 = [r["cond_ratio_sub005"] for r in records
              if math.isfinite(r["cond_ratio_sub005"])]
    return {
        "width": width, "depth": depth, "n_head": head, "n_params": n_params,
        "steps": steps, "batch": batch, "block": block,
        "wall_s": time.time() - t0,
        "final_loss": float(loss.detach()),
        "cond_ratio_median": float(np.median(ratios)) if ratios else None,
        "cond_ratio_min": float(np.min(ratios)) if ratios else None,
        "cond_ratio_max": float(np.max(ratios)) if ratios else None,
        "cond_ratio_sub015_median": float(np.median(subs)) if subs else None,
        "cond_ratio_sub005_median": float(np.median(subs05)) if subs05 else None,
        "n_matrices": len(records),
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--measure-every", type=int, default=50)
    ap.add_argument("--out", default=str(RESULTS / "scale_conditioning.json"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    data = dm.prepare_shakespeare()
    print(f"shakespeare: {data[0].size} tokens, vocab {data[2]}")

    if args.quick:
        ladder = [(192, 4, 6), (384, 6, 6)]
    else:
        # width ladder (fixed depth 6) + depth ladder (fixed width 384)
        ladder = [(192, 6, 6), (384, 6, 6), (576, 6, 6), (768, 6, 12),
                  (1152, 6, 12), (1536, 6, 16), (1024, 12, 16)]

    results = []
    for width, depth, head in ladder:
        print(f"\n=== width={width} depth={depth} heads={head} "
              f"steps={args.steps} ===", flush=True)
        try:
            r = run_one(width, depth, head, args.steps, args.batch, args.block,
                        data, device, measure_every=args.measure_every)
        except RuntimeError as e:
            print(f"  skipped ({str(e)[:80]})")
            continue
        results.append(r)
        print(f"  params {r['n_params']/1e6:.1f}M  loss {r['final_loss']:.3f}")
        print(f"  cond ratio  full median {r['cond_ratio_median']:.3e}  "
              f"[{r['cond_ratio_min']:.2e}, {r['cond_ratio_max']:.2e}]")
        print(f"  cond ratio  sub(rho=.15) median {r['cond_ratio_sub015_median']:.3e}")
        Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\nwrote {args.out}")
    print("\nscale summary (median sigma_min/sigma_max over all hidden matrices):")
    print(f"{'width':>6} {'depth':>6} {'params':>10} {'full':>12} {'sub .15':>12}")
    for r in results:
        print(f"{r['width']:6d} {r['depth']:6d} {r['n_params']/1e6:9.1f}M "
              f"{r['cond_ratio_median']:12.3e} {r['cond_ratio_sub015_median']:12.3e}")


if __name__ == "__main__":
    main()
