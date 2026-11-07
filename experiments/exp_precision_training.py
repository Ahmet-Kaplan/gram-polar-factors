"""
H13 -- does the precision sensitivity of prop:family reach the loss?

Prop:family says the per-direction relative error of a spectral map applied
through the Gram matrix is L_q(lambda) |dlambda|/lambda.  For the exact factor
(q = lambda^{-1/2}, L_q = 1/2) every direction is retained, including the tail
where |dlambda|/lambda is worst; for a map truncated at theta*lambda_max the tail
is removed, so the sensitivity is confined to directions with
lambda >= theta*lambda_max.

Prediction, and it is the one the paper has been unable to test: at matched update
magnitude, forming the Gram matrix in float32 degrades the exact factor's loss and
leaves the truncated one's alone.

Four arms, crossed with the Gram precision and nothing else:

    exact polar,  fp64 Gram        (reference)
    exact polar,  fp32 Gram        (prediction: degrades)
    truncated polar at theta, fp64 Gram
    truncated polar at theta, fp32 Gram   (prediction: unchanged)

The momentum buffer is float64 in every arm, so the only thing that varies between
the two precisions is the formation of the Gram matrix.  theta is the same in both
truncated arms, 1e-7, which is the measured float32 formation error; the map is
therefore identical and only the arithmetic differs.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "novelty"))
import common as C  # noqa: E402

OUT = ROOT / "results" / "precision_training.json"
ARMS = {"exact_fp64": ("float64", 0.0), "exact_fp32": ("float32", 0.0),
        "trunc_fp64": ("float64", 1e-7), "trunc_fp32": ("float32", 1e-7)}
STEPS = 300
LR = 3e-4
SEEDS = [0, 1, 2]
TRAIN0 = 50
EVAL_STREAM = 900


def gram_update(B, gdt, theta):
    """q(G) B with the Gram formed in gdt, cut at theta*lambda_max."""
    G = (B.to(getattr(torch, gdt)) @ B.to(getattr(torch, gdt)).T).to(torch.float64)
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0.0)
    o = torch.argsort(lam, descending=True)
    lam, V = lam[o], V[:, o]
    if theta:
        lam = torch.where(lam < theta * lam[0], torch.zeros_like(lam), lam)
    w = torch.where(lam > 0, lam.rsqrt(), torch.zeros_like(lam))
    return (V * w) @ (V.T @ B)


@torch.no_grad()
def evaluate(model, raw, perm, stream, iters=8):
    tot = 0.0
    for x, y in C.batches(raw, perm, stream, iters):
        _, loss = model(x, y)
        tot += float(loss.detach())
    return tot / iters


def run(arm, seed, raw, perm):
    gdt, theta = ARMS[arm]
    model = C.make_model(seed=0)          # identical init in every arm
    H = C.hidden(model)
    B = {n: torch.zeros(min(p.shape), max(p.shape), dtype=torch.float64,
                        device=p.device) for n, p in H.items()}
    base = TRAIN0 + seed * STEPS
    for i in range(STEPS):
        x, y = C.batches(raw, perm, 0, 1, start=base + i)[0]
        model.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        with torch.no_grad():
            for n, p in H.items():
                tr = p.grad.shape[0] > p.grad.shape[1]
                g = C.to64(p.grad)
                if tr:
                    g = g.T
                B[n].mul_(C.BETA).add_(g, alpha=1 - C.BETA)
                U = gram_update(B[n], gdt, theta)
                U = U / U.pow(2).mean().sqrt().clamp(min=1e-300)
                if tr:
                    U = U.T
                p.add_(U.to(p.dtype), alpha=-LR)
    return evaluate(model, raw, perm, EVAL_STREAM)


def main() -> None:
    print(f"{platform.machine()}, torch {torch.__version__}")
    raw, perm = C.corpus()
    out = {"config": {"steps": STEPS, "lr": LR, "seeds": SEEDS,
                      "theta": 1e-7, "machine": platform.machine()}, "arms": {}}
    for arm in ARMS:
        losses = []
        for s in SEEDS:
            vl = run(arm, s, raw, perm)
            losses.append(vl)
            print(f"  {arm:11s} seed {s} val {vl:.4f}", flush=True)
        out["arms"][arm] = {"losses": losses,
                            "mean": float(np.mean(losses)),
                            "sd": float(np.std(losses, ddof=1))}
        a = out["arms"][arm]
        print(f"  >> {arm:11s} {a['mean']:.4f} +- {a['sd']:.4f}", flush=True)

    A = out["arms"]
    d_exact = A["exact_fp32"]["mean"] - A["exact_fp64"]["mean"]
    d_trunc = A["trunc_fp32"]["mean"] - A["trunc_fp64"]["mean"]
    pooled = float(np.sqrt(np.mean([A[k]["sd"] ** 2 for k in A])))
    out["verdict"] = {
        "cost_of_fp32_gram_exact": d_exact,
        "cost_of_fp32_gram_truncated": d_trunc,
        "pooled_seed_sd": pooled,
        "exact_degrades_more": bool(d_exact > d_trunc),
        "prediction_holds": bool(d_exact > 0 and abs(d_trunc) < d_exact),
    }
    v = out["verdict"]
    print(f"\ncost of an fp32 Gram: exact {d_exact:+.4f} nats, "
          f"truncated {d_trunc:+.4f} nats (pooled seed sd {pooled:.4f})")
    print(f"prediction holds: {v['prediction_holds']}")
    Path(OUT).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
