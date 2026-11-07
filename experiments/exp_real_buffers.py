"""
The a posteriori bound, the repairs and mixed precision, on real momentum buffers.

One run produces both real-matrix tables of the NLAA manuscript (Table "real" and
the bottom block of Table "mixed"), so that the two can no longer disagree.  The
matrices are the 16 two-dimensional block weights of a 4-layer, width-192 GPT
trained on byte-level TinyStories; the momentum buffer is accumulated in float64
from float32 gradients over 21 minibatches at fixed weights (beta = 0.95).

Conventions, each of which an earlier version of this script got wrong:
  * every matrix is oriented with m <= n before the Gram matrix is formed; a tall
    weight would otherwise give a rank-deficient m x m Gram matrix and a different
    problem;
  * kappa is the condition number of the input, from a float64 SVD -- never the
    ratio of computed Gram eigenvalues, which is censored at eta^(-1/2) and makes
    eta*kappa^2 = 1 by construction;
  * eta*kappa^2 and the bound use eta_total (formation + eigensolver residual).

Requires a machine where numpy.longdouble is wider than float64 (x86-64; Rosetta 2
on Apple silicon works).
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

OUT = ROOT / "results" / "aposteriori_real.json"
STEPS = 21
CONFIGS = [("float64", "float64"), ("float32", "float64"), ("float64", "float32"),
           ("float32", "float32"), ("bfloat16", "float32")]


def _require_extended() -> None:
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        raise SystemExit(f"longdouble is not wider than float64 on "
                         f"{platform.machine()}; run on x86-64")


def opn(A: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(A, ord=2))


def analyse(B: torch.Tensor, gdt: str, edt: str) -> dict:
    B = B.detach().cpu().to(torch.float64)
    s = torch.linalg.svdvals(B)
    kappa = float(s[0] / s[-1])
    U, _, Vh = torch.linalg.svd(B, full_matrices=False)
    P = U @ Vh

    Bd = B.to(getattr(torch, gdt))
    G = (Bd @ Bd.T).to(torch.float64)
    Bl = B.numpy().astype(np.longdouble)
    Ref = Bl @ Bl.T
    nr = float(np.linalg.norm(Ref.astype(np.float64), 2))
    form = 2.0 * float(np.linalg.norm(G.numpy() - Ref.astype(np.float64), 2)) / nr
    Ge = G.to(getattr(torch, edt))
    lam, V = torch.linalg.eigh(Ge)
    lam, V = lam.to(torch.float64), V.to(torch.float64)
    Gd = Ge.to(torch.float64)
    res = opn(Gd @ V - V * lam) / opn(Gd)
    eta = form + res
    t = eta * kappa ** 2
    lam_max = float(lam.max())

    def apply(w: torch.Tensor) -> tuple[float, float]:
        Ph = (V * w) @ (V.T @ B)
        return opn(Ph - P), float(torch.linalg.svdvals(Ph)[0])

    pos = lam.clamp(min=0.0)
    plain = torch.where(pos > 0, pos.clamp(min=1e-300).rsqrt(), torch.zeros_like(pos))
    cut = torch.where(lam < eta * lam_max, torch.zeros_like(lam), lam)
    trunc = torch.where(cut > 0, cut.clamp(min=1e-300).rsqrt(), torch.zeros_like(cut))
    shift = (lam + eta * lam_max).clamp(min=1e-300).rsqrt()
    e0, s0 = apply(plain)
    e1, s1 = apply(trunc)
    e2, s2 = apply(shift)
    return {"m": B.shape[0], "n": B.shape[1], "kappa": kappa,
            "eta_formation": form, "eta_residual": res, "eta_total": eta, "t": t,
            "void": bool(t >= 1.0),
            "bound": (1.0 - t) ** -0.5 - 1.0 if t < 1.0 else float("inf"),
            "observed": e0, "sigma_max": s0,
            "observed_truncated": e1, "sigma_max_truncated": s1,
            "truncation_fires": bool((lam < eta * lam_max).any()),
            "observed_shifted": e2, "sigma_max_shifted": s2}


def agg(rows: list[dict], key: str) -> dict:
    v = [r[key] for r in rows]
    return {"median": float(np.median(v)), "min": float(np.min(v)),
            "max": float(np.max(v))}


def main() -> None:
    _require_extended()
    torch.set_num_threads(4)
    raw, perm = C.corpus()
    model = C.make_model(seed=0)
    B, loss = C.momentum(model, raw, perm, 0, STEPS, buf_dtype=torch.float64)
    print(f"{len(B)} matrices, loss {loss:.4f}, longdouble "
          f"{np.finfo(np.longdouble).eps:.1e} on {platform.machine()}")
    shapes = {"full": {n: C.thin(b.detach().cpu().to(torch.float64))
                       for n, b in B.items()},
              "sub": {n: C.subsample(b).cpu() for n, b in B.items()}}
    out = {"config": {"machine": platform.machine(), "torch": torch.__version__,
                      "longdouble_eps": float(np.finfo(np.longdouble).eps),
                      "rho": C.RHO, "steps": STEPS, "beta": C.BETA,
                      "corpus": "TinyStoriesV2 (byte level)", "depth": 4,
                      "width": 192, "n_matrices": len(B)},
           "rows": {}, "summary": {}}
    for variant, mats in shapes.items():
        for gdt, edt in CONFIGS:
            key = f"{variant}|{gdt}|{edt}"
            rows = {n: analyse(b, gdt, edt) for n, b in mats.items()}
            out["rows"][key] = rows
            R = list(rows.values())
            summ = {k: agg(R, k) for k in ("kappa", "eta_total", "t", "observed",
                                           "observed_truncated", "observed_shifted",
                                           "sigma_max", "sigma_max_truncated",
                                           "sigma_max_shifted")}
            summ["n_void"] = sum(r["void"] for r in R)
            summ["n"] = len(R)
            adm = [r for r in R if not r["void"]]
            summ["bound_never_exceeded"] = all(r["observed"] <= r["bound"] for r in adm)
            summ["bound_median_admissible"] = (float(np.median([r["bound"] for r in adm]))
                                               if adm else None)
            summ["n_truncation_fires"] = sum(r["truncation_fires"] for r in R)
            out["summary"][key] = summ
            print(f"{key:28s} kappa {summ['kappa']['median']:.2e} "
                  f"eta {summ['eta_total']['median']:.1e} t {summ['t']['median']:.1e} "
                  f"void {summ['n_void']}/{summ['n']}  err {summ['observed']['median']:.3g} "
                  f"[{summ['observed']['min']:.3g},{summ['observed']['max']:.3g}]  "
                  f"trunc {summ['observed_truncated']['median']:.3g}  "
                  f"shift {summ['observed_shifted']['median']:.3g} "
                  f"(sigma {summ['sigma_max_shifted']['max']:.4f})")
    # what subsampling gives up: the descent efficiency <U, B>/||B||_* of the
    # subsampled update is ||B_S||_* / ||B||_*, against 1 for the full polar factor
    eff = {n: float(torch.linalg.svdvals(shapes["sub"][n]).sum()
                    / torch.linalg.svdvals(shapes["full"][n]).sum())
           for n in B}
    out["subsample_descent_efficiency"] = {"per_matrix": eff,
                                           **agg([{"v": v} for v in eff.values()], "v")}
    print(f"subsampling keeps a descent efficiency of "
          f"{out['subsample_descent_efficiency']['median']:.3f} (median)")
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
