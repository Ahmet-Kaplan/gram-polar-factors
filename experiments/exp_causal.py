"""
H14 and H15 -- what the loss actually sees.

Both need the same three objects at the same weights: the momentum buffer B, the
current gradient G, and the update a rule would apply.  From those, two
measurements follow without any training.

H14.  For a spectral update built from the singular pairs of B, the first-order
loss change is -lambda * sum_j c_j with c_j = u_j^T G v_j.  That decomposes the
loss decrease across directions exactly, and the share each direction carries can
be compared with the share of the nuclear norm, which is what the truncation
lemma charges for.

H15.  The quantity the loss sees is the functional <D, G>, not the norm of D.
Define e2 = ||D||_2 and eG = |<D,G>| / ||G||_*, so that eG/e2 <= 1 always and the
ratio measures how much of the error's size reaches the loss.  If the error sits
where the gradient is small, the ratio is far below one.
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

OUT = ROOT / "results" / "causal.json"
STEPS = 21
ETA_FP32 = 1e-7


def gram_update(B, gdt, theta, Bref):
    """The factor, in the caller's units, with the Gram formed in gdt."""
    G = (B.to(getattr(torch, gdt)) @ B.to(getattr(torch, gdt)).T).to(torch.float64)
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0.0)
    o = torch.argsort(lam, descending=True)
    lam, V = lam[o], V[:, o]
    if theta:
        lam = torch.where(lam < theta * lam[0], torch.zeros_like(lam), lam)
    w = torch.where(lam > 0, lam.rsqrt(), torch.zeros_like(lam))
    return (V * w) @ (V.T @ Bref)


def main() -> None:
    print(f"{platform.machine()}, torch {torch.__version__}")
    raw, perm = C.corpus()
    model = C.make_model(seed=0)
    B, _ = C.momentum(model, raw, perm, 0, STEPS, buf_dtype=torch.float64)
    # the current gradient at the same weights, on data the buffer has not seen
    Graw, _ = C.momentum(model, raw, perm, 400, 1, buf_dtype=torch.float64)

    h14, h15 = {}, {}
    for name, Bm in B.items():
        Bm = Bm.detach().cpu().to(torch.float64)
        Gm = Graw[name].detach().cpu().to(torch.float64)
        if Bm.shape[0] > Bm.shape[1]:
            Bm, Gm = Bm.T, Gm.T
        U, s, Vh = torch.linalg.svd(Bm, full_matrices=False)
        # H14: the per-direction attribution of the loss decrease
        c = torch.einsum("mi,ij,jk->k", U, Gm, Vh.T).abs()
        share_c = (c / c.sum()).cpu().numpy()
        share_n = (s / s.sum()).cpu().numpy()
        m = len(s)
        h14[name] = {
            "m": m, "kappa": float(s[-1] / s[0]),
            "share_top10_update_norm": float(share_n[:10].sum()),
            "share_top10_loss": float(share_c[:10].sum()),
            "share_bottom_half_loss": float(share_c[m // 2:].sum()),
            "share_bottom_half_nuclear": float(share_n[m // 2:].sum()),
            "share_first_half_loss": float(share_c[: m // 2].sum()),
        }
        # H15: how much of the error reaches the loss
        P = U @ Vh
        normG = float(torch.linalg.svdvals(Gm).sum())
        normG2 = float(torch.linalg.svdvals(Gm)[0])
        row = {}
        for key, gdt, th in (("fp32", "float32", 0.0),
                             ("trunc", "float64", ETA_FP32)):
            Ph = gram_update(Bm, gdt, th, Bm)
            D = Ph - P
            e2 = float(torch.linalg.matrix_norm(D, ord=2))
            dG = abs(float((D * Gm).sum()))
            eG = dG / max(normG, 1e-300)
            # the interpretable one: by what fraction does the error change the
            # predicted loss decrease <P, G>?
            base = abs(float((P * Gm).sum()))
            row[key] = {"e2": e2, "eG": eG, "ratio": eG / max(e2, 1e-300),
                        "rel_functional": dG / max(base, 1e-300)}
        row["kappa"] = float(s[-1] / s[0])
        row["grad_flatness"] = normG2 / max(normG, 1e-300)
        # DA-2: is the gradient's energy in the same directions as the buffer's?
        Ug, sg, _ = torch.linalg.svd(Gm, full_matrices=False)
        row["overlap"] = {
            r: float(torch.linalg.matrix_norm(U[:, :r].T @ Ug[:, :r]) ** 2 / r)
            for r in (1, 5, 10, 20)}
        # fraction of the gradient's nuclear norm carried by the buffer's tail
        # coordinates of the buffer's tail directions in the gradient's basis
        pt = Ug.T @ U[:, m // 2:]                 # (m, m/2)
        w = pt.pow(2).sum(1).sqrt().clamp(max=1.0)  # per gradient direction
        row["grad_nuclear_in_buffer_tail"] = float((sg * w).sum() / sg.sum())
        # DA-5: what truncation costs, in the two currencies
        theta = ETA_FP32
        keep = (s / s[0]) >= theta ** 0.5
        nuclear_cost = 1.0 - float(s[keep].sum() / s.sum())
        Pt = gram_update(Bm, "float64", theta, Bm)
        func_cost = abs(float(((Pt - P) * Gm).sum())) / max(abs(float((P * Gm).sum())), 1e-300)
        row["truncation_costs"] = {"nuclear": nuclear_cost,
                                   "functional": func_cost}
        h15[name] = row

    def agg(d, key):
        v = [x[key] for x in d.values() if key in x]
        return {"median": float(np.median(v)), "min": float(np.min(v)),
                "max": float(np.max(v))}

    out = {"config": {"steps": STEPS, "eta_fp32": ETA_FP32,
                      "machine": platform.machine()},
           "H14": {"per_matrix": h14,
                   "share_top10_loss": agg(h14, "share_top10_loss"),
                   "share_top10_update_norm": agg(h14, "share_top10_update_norm"),
                   "share_bottom_half_loss": agg(h14, "share_bottom_half_loss"),
                   "share_bottom_half_nuclear":
                       agg(h14, "share_bottom_half_nuclear")},
           "H15": {"per_matrix": h15,
                   "ratio_fp32": agg({k: v["fp32"] for k, v in h15.items()}, "ratio"),
                   "ratio_trunc": agg({k: v["trunc"] for k, v in h15.items()}, "ratio"),
                   "rel_functional_fp32":
                       agg({k: v["fp32"] for k, v in h15.items()}, "rel_functional"),
                   "rel_functional_trunc":
                       agg({k: v["trunc"] for k, v in h15.items()}, "rel_functional"),
                   "grad_flatness": agg(h15, "grad_flatness"),
                   "overlap_r10": agg({k: {"v": v["overlap"][10]} for k, v in h15.items()}, "v"),
                   "overlap_r1": agg({k: {"v": v["overlap"][1]} for k, v in h15.items()}, "v"),
                   "grad_nuclear_in_buffer_tail":
                       agg(h15, "grad_nuclear_in_buffer_tail"),
                   "truncation_nuclear_cost":
                       agg({k: {"v": v["truncation_costs"]["nuclear"]} for k, v in h15.items()}, "v"),
                   "truncation_functional_cost":
                       agg({k: {"v": v["truncation_costs"]["functional"]} for k, v in h15.items()}, "v")}}

    h = out["H14"]
    print(f"\nH14  share of the loss decrease carried by the top ten directions: "
          f"{h['share_top10_loss']['median']:.3f} "
          f"[{h['share_top10_loss']['min']:.3f},{h['share_top10_loss']['max']:.3f}]")
    print(f"     their share of the nuclear norm:                            "
          f"{h['share_top10_update_norm']['median']:.3f}")
    print(f"     share of the loss carried by the bottom half of the spectrum: "
          f"{h['share_bottom_half_loss']['median']:.3f}")
    print(f"     their share of the nuclear norm:                            "
          f"{h['share_bottom_half_nuclear']['median']:.3f}")
    v = out["H15"]
    print(f"\nH15  eG/e2 for an fp32 Gram: {v['ratio_fp32']['median']:.3e} "
          f"[{v['ratio_fp32']['min']:.1e},{v['ratio_fp32']['max']:.1e}]")
    print(f"     eG/e2 for the truncated factor: {v['ratio_trunc']['median']:.3e} "
          f"[{v['ratio_trunc']['min']:.1e},{v['ratio_trunc']['max']:.1e}]")
    print(f"     relative change in the predicted loss decrease, fp32 Gram: "
          f"{v['rel_functional_fp32']['median']:.3e} "
          f"[{v['rel_functional_fp32']['min']:.1e},{v['rel_functional_fp32']['max']:.1e}]")
    print(f"     ... truncated: {v['rel_functional_trunc']['median']:.3e}")
    print(f"\nDA-2  ||G||_2/||G||_* = {v['grad_flatness']['median']:.3f} "
          f"[{v['grad_flatness']['min']:.2f},{v['grad_flatness']['max']:.2f}]")
    print(f"      overlap of the leading direction, cos^2 = {v['overlap_r1']['median']:.3f}")
    print(f"      overlap of the leading ten, F^2/10      = {v['overlap_r10']['median']:.3f}")
    print(f"      gradient's nuclear norm in the buffer's tail: "
          f"{v['grad_nuclear_in_buffer_tail']['median']:.3f}")
    print(f"\nDA-5  truncation at eta: nuclear cost {v['truncation_nuclear_cost']['median']:.2e}, "
          f"functional cost {v['truncation_functional_cost']['median']:.2e}")
    out["verdict"] = {
        "H14_loss_more_concentrated_than_nuclear":
            h["share_top10_loss"]["median"] > h["share_top10_update_norm"]["median"],
        "H14_tail_carries_little_loss":
            h["share_bottom_half_loss"]["median"] < 0.2,
        "H15_error_much_smaller_in_the_functional":
            v["ratio_fp32"]["median"] < 0.1,
    }
    print(f"\nverdict: {out['verdict']}")
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
