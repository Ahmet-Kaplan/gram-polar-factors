"""
The rotation term is one power of kappa cheaper than the value term.

Round 4 of the panel noted that prop:subspace bounds the angle but not the
resulting error, and two attempts to measure a constant failed.  The reason the
attempts failed is that there is nothing to measure: the two mechanisms scale
differently, and the scaling is derivable.

Take a 2x2 block, G = diag(l1, l2) with l1 > l2 = l1/kappa^2, B = diag(s1, s2)
with s_i = sqrt(l_i), and a purely rotational perturbation E = [[0, eps],[eps, 0]]
of norm eps = eta*l1.  Then

  * the eigenvalues move only at second order,
        lhat_1 = l1 + eps^2/(l1 - l2),  lhat_2 = l2 - eps^2/(l1 - l2),
    so the rescaling error s_i - 1 = O((eta kappa)^2 / ...) is second order;
  * the eigenvectors rotate at first order, sin(theta) = eps/(l1 - l2), and the
    off-diagonal entry of Ghat^(-1/2) B is
        eps/(s1 s2) = eps/sqrt(l1 l2) = eta * sqrt(l1/l2) = eta * kappa.

So the rotation contributes eta*kappa and the value term contributes
(1/2) eta*kappa^2: for kappa >> 1 the rotation is dominated, which is why no
sharp constant was needed and why Theorem 1's bound controls the error after all.

Measured below: a purely rotational perturbation (zero diagonal, so it cannot
change an eigenvalue at first order) against a diagonal one of the same norm, on
blocks and on 192 x 576 matrices with prescribed spectra.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "subspace_order.json"
KAPPAS = [10.0, 1e2, 1e3, 1e4]
ETAS = [1e-9, 1e-8, 1e-7]


def make_matrix(kappa, m=192, n=576, seed=0):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, m, generator=g, dtype=torch.float64))
    s = torch.logspace(0, np.log10(kappa), m, dtype=torch.float64)
    return (U * s) @ V.T


def polar(B):
    U, _, Vh = torch.linalg.svd(B, full_matrices=False)
    return U @ Vh


def factor_from_G(G, B):
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0.0)
    o = torch.argsort(lam, descending=True)
    lam, V = lam[o], V[:, o]
    w = torch.where(lam > 0, lam.rsqrt(), torch.zeros_like(lam))
    return (V * w) @ (V.T @ B)


def runs() -> dict:
    rows = []
    for kappa in KAPPAS:
        for eta in ETAS:
            B = make_matrix(kappa)
            B = B if B.shape[0] <= B.shape[1] else B.T
            G = B @ B.T
            lam_max = float(torch.linalg.matrix_norm(G, ord=2))
            P0 = polar(B)
            g = torch.Generator().manual_seed(1)

            # (i) purely rotational: symmetric, zero diagonal
            R = torch.randn(G.shape, dtype=torch.float64, generator=g)
            R = 0.5 * (R + R.T)
            R.fill_diagonal_(0.0)
            E_rot = R * (eta * lam_max /
                         float(torch.linalg.matrix_norm(R, ord=2)))
            # (ii) purely rescaling: diagonal, same norm
            d = torch.randn(G.shape[0], dtype=torch.float64, generator=g)
            D = torch.diag(d)
            E_val = D * (eta * lam_max /
                         float(torch.linalg.matrix_norm(D, ord=2)))

            e_rot = float(torch.linalg.matrix_norm(
                factor_from_G(G + E_rot, B) - P0, ord=2))
            e_val = float(torch.linalg.matrix_norm(
                factor_from_G(G + E_val, B) - P0, ord=2))
            rows.append({"kappa": kappa, "eta": eta,
                         "err_rotation": e_rot, "err_value": e_val,
                         "rot_over_eta_kappa": e_rot / (eta * kappa),
                         "val_over_eta_kappa_sq": e_val / (eta * kappa ** 2),
                         "ratio_rot_to_val": e_rot / max(e_val, 1e-300)})
    # 2x2 analytic check
    blocks = []
    for kappa in KAPPAS:
        l1 = 1.0
        l2 = l1 / kappa ** 2
        B = torch.diag(torch.tensor([l1, l2], dtype=torch.float64).sqrt())
        for eta in ETAS:
            eps = eta * l1
            E = torch.tensor([[0.0, eps], [eps, 0.0]], dtype=torch.float64)
            P0 = polar(B)
            err = float(torch.linalg.matrix_norm(
                factor_from_G(B @ B.T + E, B) - P0, ord=2))
            blocks.append({"kappa": kappa, "eta": eta, "err": err,
                           "predicted": eta * kappa,
                           "ratio": err / (eta * kappa)})
    return {"rows": rows, "blocks": blocks}


def main() -> None:
    print(platform.machine())
    out = {"config": {"kappas": KAPPAS, "etas": ETAS,
                      "machine": platform.machine()}, **runs()}
    print(f"{'kappa':>8} {'eta':>8} {'rot/(eta k)':>12} {'val/(eta k^2)':>14} "
          f"{'rot/val':>9}")
    for r in out["rows"]:
        print(f"{r['kappa']:8.0e} {r['eta']:8.0e} {r['rot_over_eta_kappa']:12.4f} "
              f"{r['val_over_eta_kappa_sq']:14.4f} {r['ratio_rot_to_val']:9.2e}")
    print("\n2x2 block, analytic prediction err = eta*kappa:")
    for b in out["blocks"]:
        print(f"  kappa {b['kappa']:8.0e} eta {b['eta']:8.0e} "
              f"err {b['err']:.4e} predicted {b['predicted']:.4e} "
              f"ratio {b['ratio']:.4f}")
    rot = [r["rot_over_eta_kappa"] for r in out["rows"]]
    val = [r["val_over_eta_kappa_sq"] for r in out["rows"]]
    br = [b["ratio"] for b in out["blocks"]]
    out["verdict"] = {
        "rot_over_eta_kappa_median": float(np.median(rot)),
        "rot_over_eta_kappa_max": float(np.max(rot)),
        "val_over_eta_kappa_sq_median": float(np.median(val)),
        "rot_dominated": bool(max(rot) < np.median(val)),
        "block_ratio_median": float(np.median(br)),
        "block_ratio_max": float(np.max(br)),
        "block_prediction_holds": bool(0.2 < float(np.median(br)) < 5.0)}
    v = out["verdict"]
    print(f"\nrotation/(eta kappa): median {v['rot_over_eta_kappa_median']:.4f} "
          f"max {v['rot_over_eta_kappa_max']:.4f}")
    print(f"value/(eta kappa^2):  median {v['val_over_eta_kappa_sq_median']:.4f}")
    print(f"rotation dominated by the value term: {v['rot_dominated']}")
    print(f"2x2 blocks: measured/(eta kappa) median {v['block_ratio_median']:.4f} "
          f"max {v['block_ratio_max']:.4f}")
    Path(OUT).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
