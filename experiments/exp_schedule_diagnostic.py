"""
Proposition 3 as a practical diagnostic: score a published schedule on a
spectrum, with no training run and no matrix product.

The appeal of the identity

    eta(P(B), B) = sum_i s_i p_i / ( max_i |p_i| * sum_i s_i )

is that p is the schedule's scalar map and s is a measured spectrum.  "Which
orthogonalizer should I use on these matrices?" becomes a quadrature over
singular values, answerable before spending a GPU hour.

Two comparisons are reported, because they disagree and the disagreement is the
point:

  eta   = sum s_i p_i / (max|p| sum s_i)      per unit spectral norm
  gain  = sum s_i p_i / sum s_i = eta * max|p|  per unit learning rate

eta is the metric the paper audits.  gain is what the first-order model gets per
step at a fixed learning rate when the update is used at the scale the
implementation emits, and it is not normalised: an update whose top singular
value overshoots 1 buys more descent per step, at the cost of the Lipschitz
budget.  Schedules that look good on one look bad on the other.

A third column guards against reading either as canonical.  The training runs in
Section 9.3 renormalise every update to unit elementwise RMS, which multiplies
gain by sqrt(mn)/||U||_F, a factor that spans an order of magnitude across these
schedules.  gain_rms reports that quantity, and its per-matrix rank is what the
paper quotes when it says the matched-RMS ordering differs again.

The three columns are not three bookkeepings; they are the three curvature
models a first-order analysis admits.  Writing U for the emitted update and
c_1 = <U,B>, c_2 = <U,H[U]> for the linear and quadratic coefficients of the
per-step decrease, the unit-step linear term ranks by c_1 (= gain), the
step-optimal decrease under operator-norm smoothness ranks by c_1/||U||_2
(= eta), and the step-optimal decrease under Frobenius smoothness ranks by
c_1/||U||_F.  The last two differ by exactly

    sqrt(er),  er := ||U||_F^2 / ||U||_2^2,

the update's effective rank (participation ratio), so

    gain_rms = eta * sqrt(mn) / sqrt(er).

That single number organises everything the two step-optimal columns disagree
about, and the script reports it per cell, together with the pairwise reversal
counts and the check that the analytic reversal criterion

    er_A/er_B > (eta_A/eta_B)^2

reproduces every observed disagreement.

Every schedule is fed the same input convention: the buffer normalised by its
Frobenius norm, which is what Muon, Polar Express and Dion3 all do internally.
The raw-momentum arm therefore divides by ||B||_F too; an unnormalised arm would
have a gain in the units of B, and its median would mix four different scales.

The schedules are scored as implemented (float32 code path, as in Table 8), and
the exact-spectrum evaluation is reported beside the matrix evaluation so that
the arithmetic is not mistaken for the model.  The fit-window sweep is the
parameter the literature actually tunes: a minimax composition fitted for
kappa <= 1/l is being run on matrices with kappa up to 1e9.

Run:  python exp_schedule_diagnostic.py
Out:  ../results/schedule_diagnostic.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
import robust_orth as ro
from baselines import (gram_newton_schulz, optimal_composition, polar_express)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RHO = 0.15
TAIL = 1e-2          # sigma_i/sigma_max below this counts as "tail"


@torch.no_grad()
def eta_direct(U: torch.Tensor, B: torch.Tensor) -> float:
    """The manuscript's definition, evaluated on the matrices."""
    Uf, Bf = U.double(), B.double()
    return float((Uf * Bf).sum()
                 / (torch.linalg.matrix_norm(Uf, ord=2)
                    * torch.linalg.svdvals(Bf).sum()))


@torch.no_grad()
def p_of(s: torch.Tensor, fn) -> torch.Tensor:
    """The schedule's scalar map, read off diag(s).

    Every schedule in the table is a function of Z Z^T, hence of the singular
    values, so acting on a diagonal matrix returns p_i on the diagonal exactly
    as the schedule produces it on the real matrix.
    """
    return torch.diag(fn(torch.diag(s.double()))).double()


@torch.no_grad()
def metrics(p: torch.Tensor, s: torch.Tensor) -> dict:
    """eta (per unit spectral norm), gain (per unit lr), and where mass is lost."""
    pmax = float(p.abs().max())
    pm = p / pmax
    num = float((s * p).sum())
    tot = float(s.sum())
    w = s / tot
    short = w * (1.0 - pm)
    tail = s / s.max() < TAIL
    sh = float(short.sum())
    return {
        "eta": num / (pmax * tot),
        "gain": num / tot,
        "p_max": pmax,
        "p_min": float(p.min()),
        "shortfall": sh,
        "shortfall_from_tail": float(short[tail].sum()),
        "tail_mass": float(w[tail].sum()),
        "top_shortfall_share": float(short[0] / max(sh, 1e-12)),
        "w1": float(w[0]),
    }


@torch.no_grad()
def tail_share_at(p: torch.Tensor, s: torch.Tensor, thresh: float) -> float:
    """Share of the eta shortfall carried by directions below thresh*sigma_max."""
    pm = p / p.abs().max()
    w = s / s.sum()
    short = w * (1.0 - pm)
    return float(short[s / s.max() < thresh].sum() / max(float(short.sum()), 1e-12))


def compose64(coeffs, steps: int):
    """Polar Express's arithmetic, in float64, with a supplied coefficient table."""
    @torch.no_grad()
    def fn(X: torch.Tensor) -> torch.Tensor:
        X = X.double()
        Z = X / (X.norm() + 1e-7)
        for t in range(steps):
            a, b, c = coeffs[t]
            A = Z @ Z.T
            Z = a * Z + (b * A + c * (A @ A)) @ Z
        return Z
    return fn


def schedules() -> dict:
    return {
        # The reference: p = 1 on every observed direction, so eta = gain = 1
        # and the emitted norms are those of an orthonormal factor.
        "exact polar factor": lambda X: ro.polar_svd(X),
        # Every other arm divides by ||B||_F before iterating (Muon, Polar
        # Express and Dion3 all do).  The raw arm must use the same input
        # convention, or its gain is measured in the units of B and its median
        # over matrices mixes scales that differ by 4x in the saved buffers.
        "raw momentum": lambda X: X / (X.norm() + 1e-30),        "Muon quintic, 1 step": lambda X: ro.muon_reference(X, 1),
        "Muon quintic, 3 steps": lambda X: ro.muon_reference(X, 3),
        "Muon quintic, 5 steps": lambda X: ro.muon_reference(X, 5),
        "Muon quintic, 10 steps": lambda X: ro.muon_reference(X, 10),
        "Polar Express, 5 steps": lambda X: polar_express(X, steps=5),
        "Polar Express, 8 steps": lambda X: polar_express(X, steps=8),
        "Gram-NS (Dion3), 5 steps": lambda X: gram_newton_schulz(X, steps=5),
    }


def main() -> None:
    bufs = torch.load(RESULTS / "momentum_matrices.pt", map_location="cpu")
    rows = []

    for name, B in bufs.items():
        idx = ro.select_rows(B.float(), RHO, None, mode="norm")
        for regime, X in (("full", B.float()), ("sub", B[idx].float())):
            s = torch.linalg.svdvals(X.double())
            for sched, fn in schedules().items():
                p = p_of(s, fn)
                m = metrics(p, s)
                # The schedule as implemented: eta on the matrix, and the
                # gain that follows from it exactly, gain = eta * ||U||_2.
                U = fn(X)
                eta_mat = eta_direct(U, X)
                norm2 = float(torch.linalg.matrix_norm(U.double(), ord=2))
                normf = float(torch.linalg.matrix_norm(U.double(), ord="fro"))
                mrows, ncols = X.shape
                rows.append({
                    "param": name, "regime": regime, "schedule": sched,
                    "eta": eta_mat, "eta_spectral": m["eta"],
                    "absdiff": abs(m["eta"] - eta_mat),
                    "gain": eta_mat * norm2,
                    "p_max": norm2, "p_min": m["p_min"],
                    # first-order gain at the unit-elementwise-RMS scale the
                    # training runs use: gain * sqrt(mn) / ||U||_F
                    "norm_fro": normf,
                    "eff_rank": (normf ** 2 / max(norm2 ** 2, 1e-300)),
                    "rho": norm2 / max(normf, 1e-30),
                    "rank_bound": min(mrows, ncols),
                    "gain_rms": (eta_mat * norm2
                                 * (mrows * ncols) ** 0.5 / max(normf, 1e-30)),
                    "shortfall": m["shortfall"],
                    "shortfall_from_tail": m["shortfall_from_tail"],
                    "tail_mass": m["tail_mass"],
                    "w1": m["w1"],
                    "top_shortfall_share": m["top_shortfall_share"],
                })

    # ---- fit-window sweep: what the conditioning result predicts ----------
    # The scalar map is applied in float64 to the exact spectrum, so this sweep
    # is the schedule's ideal behaviour, not a float32 artefact.
    sweep = []
    for l in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        coeffs = optimal_composition(l=l, num_iters=5,
                                     safety_factor_eps=1e-3, cushion=1e-3)
        fn = compose64(coeffs, 5)
        for regime in ("sub", "full"):
            vals = []
            for B in bufs.values():
                X = B.float() if regime == "full" else B.float()[
                    ro.select_rows(B.float(), RHO, None, mode="norm")]
                s = torch.linalg.svdvals(X.double())
                vals.append(metrics(p_of(s, fn), s))
            sweep.append({
                "fit_floor": l, "regime": regime,
                "eta_median": float(np.median([v["eta"] for v in vals])),
                "gain_median": float(np.median([v["gain"] for v in vals])),
                "p_max_median": float(np.median([v["p_max"] for v in vals])),
                "tail_mass_median": float(np.median([v["tail_mass"] for v in vals])),
            })

    def summ(regime: str) -> list[dict]:
        out = []
        reg = [r for r in rows if r["regime"] == regime]
        # median rank of each schedule on the matched-RMS gain, within matrix
        rank_rows = {}
        for param in {r["param"] for r in reg}:
            grp = sorted([r for r in reg if r["param"] == param],
                         key=lambda r: -r["gain_rms"])
            for i, r in enumerate(grp):
                rank_rows.setdefault(r["schedule"], []).append(i + 1)
        for sched in schedules():
            v = [r for r in reg if r["schedule"] == sched]
            out.append({
                "schedule": sched,
                "eta_median": float(np.median([r["eta"] for r in v])),
                "eta_min": float(np.min([r["eta"] for r in v])),
                "eta_spectral_median": float(np.median([r["eta_spectral"] for r in v])),
                "gain_median": float(np.median([r["gain"] for r in v])),
                "gain_rms_median": float(np.median([r["gain_rms"] for r in v])),
                "gain_rms_rank_median": float(np.median(rank_rows[sched])),
                "rho_median": float(np.median([
                    r["p_max"] / max(r["norm_fro"], 1e-30) for r in v])),
                "eff_rank_median": float(np.median([r["eff_rank"] for r in v])),
                "sqrt_eff_rank_median": float(np.median([
                    r["eff_rank"] ** 0.5 for r in v])),
                "p_max_median": float(np.median([r["p_max"] for r in v])),
                "p_min_median": float(np.median([r["p_min"] for r in v])),
                "shortfall_median": float(np.median([r["shortfall"] for r in v])),
                "tail_share_of_shortfall": float(np.median([
                    r["shortfall_from_tail"] / max(r["shortfall"], 1e-12)
                    for r in v])),
                "tail_mass_median": float(np.median([r["tail_mass"] for r in v])),
                "w1_median": float(np.median([r["w1"] for r in v])),
                "top_shortfall_share_median": float(np.median([
                    r["top_shortfall_share"] for r in v])),
            })
        return out

    per_sched = {}
    for sched in schedules():
        v = [r["absdiff"] for r in rows if r["schedule"] == sched]
        per_sched[sched] = {"max": float(np.max(v)),
                            "median": float(np.median(v))}

    # ---- the 1e-2 tail cut is a choice; report the two neighbours ----------
    tail_sweep = {}
    for regime in ("sub", "full"):
        tail_sweep[regime] = {}
        for sched, fn in schedules().items():
            shares = {1e-3: [], 1e-2: [], 1e-1: []}
            for B in bufs.values():
                X = (B.float() if regime == "full" else B.float()[
                    ro.select_rows(B.float(), RHO, None, mode="norm")])
                s = torch.linalg.svdvals(X.double())
                p = p_of(s, fn)
                for t in shares:
                    shares[t].append(tail_share_at(p, s, t))
            tail_sweep[regime][sched] = {
                f"{t:.0e}": float(np.median(v)) for t, v in shares.items()}

    # ---- how many cells the eight-step arithmetic actually moves ----------
    pe8 = [r for r in rows if r["schedule"] == "Polar Express, 8 steps"]
    pe8_bad = [r for r in pe8 if r["absdiff"] > 0.1]

    # ---- the gauge block: the effective-rank law on every pairwise ordering --
    # For each matrix, every one of the 36 pairs is scored under all three
    # conventions.  The analytic criterion is checked against the observed sign
    # of the disagreement; it must agree everywhere, which makes this a test of
    # the implementation rather than of the model.
    def sgn(a: float, b: float) -> int:
        return (a > b) - (a < b)

    gauge = {}
    for regime in ("sub", "full"):
        params = sorted({r["param"] for r in rows if r["regime"] == regime})
        pairs = rev_rms = rev_gain = agree = 0
        disp_rms, disp_gain, distinct, er_max = [], [], 0, []
        for param in params:
            rs = [r for r in rows
                  if r["regime"] == regime and r["param"] == param]
            er_max.append(max(r["eff_rank"] for r in rs))
            for a, b in itertools.combinations(rs, 2):
                pairs += 1
                s_eta = sgn(a["eta"], b["eta"])
                s_rms = sgn(a["gain_rms"], b["gain_rms"])
                s_g = sgn(a["gain"], b["gain"])
                if s_eta * s_rms < 0:
                    rev_rms += 1
                if s_eta * s_g < 0:
                    rev_gain += 1
                if s_eta > 0:
                    pred = (a["eff_rank"] / b["eff_rank"]) > (a["eta"] / b["eta"]) ** 2
                elif s_eta < 0:
                    pred = (b["eff_rank"] / a["eff_rank"]) > (b["eta"] / a["eta"]) ** 2
                else:
                    pred = None
                if pred is not None and pred == (s_eta * s_rms < 0):
                    agree += 1
            o_eta = sorted(rs, key=lambda r: -r["eta"])
            o_rms = sorted(rs, key=lambda r: -r["gain_rms"])
            o_g = sorted(rs, key=lambda r: -r["gain"])
            disp_rms.append(max(abs(o_eta.index(x) - o_rms.index(x)) for x in rs))
            disp_gain.append(max(abs(o_eta.index(x) - o_g.index(x)) for x in rs))
            if len({tuple(x["schedule"] for x in o_eta),
                    tuple(x["schedule"] for x in o_rms),
                    tuple(x["schedule"] for x in o_g)}) == 3:
                distinct += 1
        gauge[regime] = {
            "n_pairs_checked": pairs,
            "reversals_eta_vs_rms": rev_rms,
            "reversals_eta_vs_gain": rev_gain,
            "criterion_agrees": agree,
            "rank_displacement_median": float(np.median(disp_rms)),
            "rank_displacement_gain_median": float(np.median(disp_gain)),
            "all_three_orders_distinct": distinct,
            "n_matrices": len(params),
            "eff_rank_max_over_matrices_median": float(np.median(er_max)),
            "rank_bound": int(min(r["rank_bound"] for r in rows
                                  if r["regime"] == regime)),
        }

    out = {
        "rho": RHO,
        "tail_threshold": TAIL,
        "n_matrices": len(bufs),
        "spectral_vs_matrix": per_sched,
        "max_abs_eta_discrepancy": float(max(r["absdiff"] for r in rows)),
        "pe8_cells_absdiff_gt_0_1": {
            "total": len(pe8_bad), "of": len(pe8),
            "sub": sum(1 for r in pe8_bad if r["regime"] == "sub"),
            "full": sum(1 for r in pe8_bad if r["regime"] == "full"),
            "worst": float(max(r["absdiff"] for r in pe8_bad)) if pe8_bad else 0.0,
        },
        "tail_share_by_threshold": tail_sweep,
        "gauge": gauge,
        "summary": {"sub": summ("sub"), "full": summ("full")},
        "fit_window_sweep": sweep,
        "rows": rows,
    }
    (RESULTS / "schedule_diagnostic.json").write_text(json.dumps(out, indent=2))

    for regime in ("sub", "full"):
        print(f"== {regime} (medians over {len(bufs)}) ==")
        print(f"{'schedule':28s} {'eta':>6s} {'gain':>6s} {'|p|max':>7s} "
              f"{'rank_rms':>8s} {'tail%':>6s} {'|d eta|':>8s}")
        for r in out["summary"][regime]:
            print(f"{r['schedule']:28s} {r['eta_median']:6.3f} "
                  f"{r['gain_median']:6.3f} {r['p_max_median']:7.3f} "
                  f"{r['gain_rms_rank_median']:8.1f} "
                  f"{r['eff_rank_median']:7.2f} "
                  f"{100 * r['tail_share_of_shortfall']:6.1f} "
                  f"{per_sched[r['schedule']]['median']:8.1e}")
        print(f"  gauge: {gauge[regime]}")
        print()

    print("== fit window sweep (minimax composition, 5 steps, float64 map) ==")
    print(f"{'fit floor':>10s} {'sub eta':>8s} {'sub gain':>9s} "
          f"{'full eta':>9s} {'full gain':>10s}")
    for l in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        sub = [x for x in sweep if x["fit_floor"] == l and x["regime"] == "sub"][0]
        full = [x for x in sweep if x["fit_floor"] == l and x["regime"] == "full"][0]
        print(f"{l:10.0e} {sub['eta_median']:8.3f} {sub['gain_median']:9.3f} "
              f"{full['eta_median']:9.3f} {full['gain_median']:10.3f}")
    print(f"\nwrote {RESULTS / 'schedule_diagnostic.json'}")


if __name__ == "__main__":
    main()
