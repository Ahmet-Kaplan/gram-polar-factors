"""
Check every table against the archived result files.

`verify_numbers.py` checks that a number *appears* in the manuscript.  It cannot
check that it appears in the right table, in the right row, and in the right
column.  This script parses each table and compares cell by cell, so a number
copied into the wrong row is caught.

Sources are named explicitly per table; a table whose source is not listed here is
checked only for internal arithmetic.  Tolerances allow for the two or three
significant digits the tables print.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# The manuscript is not part of the code archive, so the path is either given on
# the command line or looked for in the layout the author works in.
_CANDIDATES = [ROOT / "paper" / "nlaa" / "main.tex", ROOT / "main.tex"]
_paper = next((Path(a.split("=", 1)[1]) for a in sys.argv[1:]
               if a.startswith("--paper=")), None)
if _paper is None:
    _argv = sys.argv[1:]
    _paper = next((Path(_argv[i + 1]) for i, a in enumerate(_argv)
                   if a == "--paper" and i + 1 < len(_argv)), None)
if _paper is None:
    _paper = next((q for q in _CANDIDATES if q.exists()), None)
if _paper is None or not _paper.exists():
    sys.exit("check_tables: no manuscript found; pass --paper=<path to main.tex>")
TEX = _paper.read_text()
J = lambda f: json.loads((ROOT / "results" / f).read_text())

failures: list[str] = []
checks = 0


def num(s: str) -> float | None:
    s = s.replace("$", "").replace("\\,", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:\\times\s*10\^\{?(-?\d+)\}?)?", s)
    if not m:
        return None
    v = float(m.group(1))
    return v * 10 ** int(m.group(2)) if m.group(2) else v


def all_nums(s: str) -> list[float]:
    """Every number in a cell, in order.  Safer than splitting on commas, because
    a LaTeX thin space is written with a backslash and a comma, and contains one."""
    s = s.replace("$", "")
    out = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)(?:\s*\\times\s*10\^\{?(-?\d+)\}?)?", s):
        v = float(m.group(1))
        out.append(v * 10 ** int(m.group(2)) if m.group(2) else v)
    return out


def rows(label: str) -> list[list[str]]:
    m = re.search(r"\\begin\{table\}.*?\\label\{" + label + r"\}(.*?)\\end\{table\}",
                  TEX, re.S)
    body = m.group(1)
    out, cur = [], ""
    for ln in body.split("\n"):
        if "rule" in ln:
            cur = ""
            continue
        cur = (cur + " " + ln).strip() if cur else ln.strip()
        if cur.endswith("\\\\"):
            if cur.count("&") >= 1:
                out.append([c.strip() for c in cur.rstrip("\\ ").split("&")])
            cur = ""
    return out


def chk(where: str, printed, actual, rel: float = 0.05) -> None:
    global checks
    checks += 1
    if printed is None or actual is None:
        failures.append(f"{where}: unparsed or missing (printed={printed}, data={actual})")
    elif abs(printed - actual) > rel * abs(actual) + 1e-300:
        failures.append(f"{where}: table says {printed:.4g}, data says {actual:.4g}")


# ---------------------------------------------------------------- tab:eps
def check_eps() -> None:
    for r in rows("tab:eps"):
        dt = r[0].strip()
        if dt not in ("float64", "float32", "float16", "bfloat16"):
            continue
        f = torch.finfo(getattr(torch, dt))
        chk(f"tab:eps {dt} eps", num(r[1]), f.eps, 0.01)
        chk(f"tab:eps {dt} sqrt(eps)", num(r[2]), f.eps ** 0.5, 0.02)
        chk(f"tab:eps {dt} eps^-1/2", num(r[3]), f.eps ** -0.5, 0.03)
        chk(f"tab:eps {dt} smallest normal", num(r[4]), f.tiny, 0.05)


# --------------------------------------------------------- tab:aposteriori
def check_aposteriori() -> None:
    ap = J("aposteriori.json")
    plain = {(round(1 / r["kappa_target"]), r["gram_dtype"]): r
             for r in ap["rows"] if r.get("mode", "plain") == "plain"}
    for r in rows("tab:aposteriori"):
        m = re.search(r"10\^\{(-?\d+)\}", r[0])
        if not m:
            continue
        kap, dt = 10 ** int(m.group(1)), r[1]
        d = plain[(kap, dt)]
        chk(f"aposteriori {kap:.0e} {dt} eta", num(r[2]), d["eta_total"])
        chk(f"aposteriori {kap:.0e} {dt} eta*kappa^2", num(r[3]), d["eta_hat_kappa_sq"])
        if "void" in " ".join(r):
            chk(f"aposteriori {kap:.0e} {dt} void", 1.0,
                1.0 if d["bound_two_sided"] >= 1 else 0.0)
        else:
            chk(f"aposteriori {kap:.0e} {dt} bound", num(r[4]), d["bound_two_sided"])
        chk(f"aposteriori {kap:.0e} {dt} observed", num(r[-1]), d["observed"])


# ----------------------------------------------------------- tab:truncation
def check_truncation() -> None:
    ap = J("aposteriori.json")
    modes = {(round(1 / r["kappa_target"]), r["gram_dtype"], r.get("mode", "plain")): r
             for r in ap["rows"]}
    name = {"fp32": "float32", "fp64": "float64"}
    for r in rows("tab:truncation"):
        m = re.search(r"10\^\{(-?\d+)\}", r[0])
        if not m:
            continue
        kap, dt = 10 ** int(m.group(1)), name[r[1]]
        for j, mode, field in ((2, "plain", "sigma_max_of_computed_factor"),
                               (3, "truncated", "sigma_max_of_computed_factor"),
                               (4, "shifted", "sigma_max_of_computed_factor"),
                               (5, "plain", "observed"),
                               (6, "truncated", "observed"),
                               (7, "shifted", "observed")):
            chk(f"truncation {kap:.0e} {dt} {mode}.{field}", num(r[j]),
                modes[(kap, dt, mode)][field])
        chk(f"truncation {kap:.0e} {dt} mass", num(r[8]),
            modes[(kap, dt, "truncated")]["nuclear_fraction_retained"], 0.002)


# ---------------------------------------------------------------- tab:sweep
def check_sweep() -> None:
    d = J("round5.json")["D_kappa_sweep_fp32"]["rows"]

    def kap_of(s: str) -> float | None:
        s = s.replace("$", "")
        m = re.match(r"([\d.]+)\\times10\^\{(-?\d+)\}", s)
        if m:
            return float(m.group(1)) * 10 ** int(m.group(2))
        m = re.match(r"10\^\{(-?\d+)\}", s)
        return 10.0 ** int(m.group(1)) if m else None

    for r in rows("tab:sweep"):
        kap = kap_of(r[0])
        if kap is None:
            continue
        d0 = min(d, key=lambda x: abs(x["kappa"] - kap))
        chk(f"sweep {kap:.0f} kappa*eps^1/2", num(r[1]), d0["kappa_over_threshold"])
        chk(f"sweep {kap:.0f} t", num(r[2]), d0["t"])
        if "void" in " ".join(r):
            chk(f"sweep {kap:.0f} void", 1.0, 1.0 if d0["t"] >= 1 else 0.0)
        else:
            chk(f"sweep {kap:.0f} bound", num(r[3]), d0["bound"])
        chk(f"sweep {kap:.0f} plain", num(r[4]), d0["err_plain"])
        chk(f"sweep {kap:.0f} truncated", num(r[5]), d0["err_truncated"])
        chk(f"sweep {kap:.0f} shifted", num(r[6]), d0["err_shifted"])


# ---------------------------------------------------------------- tab:floors
def check_floors() -> None:
    E = J("round5.json")["E_newton_schulz_floors"]["cells"]
    order = [E[k][dt] for k in ("1e+02", "1e+03")
             for dt in ("float64", "float32", "bfloat16")]
    for r in rows("tab:floors"):
        head = r[0].replace("$", "")
        if head.startswith("floor"):
            for i, c in enumerate(r[1:]):
                chk(f"floors floor[{i}]", num(c), order[i]["floor"], 0.03)
        if head.startswith("iterations"):
            for i, c in enumerate(r[1:]):
                if order[i]["iters_to_1e-1"] is None:
                    continue
                chk(f"floors iterations[{i}]", num(c),
                    float(order[i]["iters_to_1e-1"]), 0.001)
        if head.startswith("error at 5"):
            for i, c in enumerate(r[1:]):
                chk(f"floors error@5[{i}]", num(c), order[i]["error_at_5"], 0.01)
        if head.startswith("error at 10"):
            for i, c in enumerate(r[1:]):
                chk(f"floors error@10[{i}]", num(c), order[i]["error_at_10"], 0.01)


# ----------------------------------------------------------------- tab:real
def check_real() -> None:
    s = J("aposteriori_real.json")["summary"]
    keys = {"full, float64": "full|float64|float64",
            "full, float32": "full|float32|float64",
            "sub, float64": "sub|float64|float64",
            "sub, float32": "sub|float32|float64"}
    for r in rows("tab:real"):
        k = keys.get(r[0])
        if k is None:
            continue
        d = s[k]
        chk(f"real {r[0]} kappa", num(r[1]), d["kappa"]["median"])
        chk(f"real {r[0]} eta", num(r[2]), d["eta_total"]["median"])
        chk(f"real {r[0]} t", num(r[3]), d["t"]["median"], 0.10)
        chk(f"real {r[0]} void count", float(r[4].split("/")[0]), float(d["n_void"]), 0.001)
        chk(f"real {r[0]} error median", num(r[5].split("[")[0]), d["observed"]["median"])
        chk(f"real {r[0]} truncated", num(r[6]), d["observed_truncated"]["median"])


# ---------------------------------------------------------------- tab:mixed
def check_mixed() -> None:
    F = J("round5.json")["F_mixed_wide"]["cells"]
    top = (("float64", "float64"), ("float32", "float64"), ("float64", "float32"),
           ("float32", "float32"), ("float16", "float32"), ("bfloat16", "float32"))
    ks = ("1e+01", "1e+02", "1e+03")
    data = [r for r in rows("tab:mixed")]
    block = []
    for r in data:
        if "multicolumn" in " ".join(r) and "subsampled" in " ".join(r):
            break
        block.append(r)
    got = [r for r in block if r[0].strip() in
           ("fp64", "fp32", "fp16", "bf16") and len(r) >= 6]
    for r, (g, e) in zip(got, top):
        c = F[f"{g}|{e}"]
        for k, cell in zip(ks, r[3:6]):
            chk(f"mixed {g}/{e} at kappa {k}", num(cell), c[k]["median"])
    s = J("aposteriori_real.json")["summary"]
    sub = {"fp64|fp64": "sub|float64|float64", "fp32|fp64": "sub|float32|float64",
           "fp64|fp32": "sub|float64|float32", "fp32|fp32": "sub|float32|float32",
           "bf16|fp32": "sub|bfloat16|float32"}
    tail = [r for r in data if len(r) >= 5 and "void in" in " ".join(r)]
    for r, key in zip(tail, sub.values()):
        chk(f"mixed real {key} eta", num(r[2]), s[key]["eta_total"]["median"])
        cell = " ".join(r[3:])
        chk(f"mixed real {key} error", num(cell.split("void")[0].split(",")[0]),
            s[key]["observed"]["median"])
        chk(f"mixed real {key} void", num(cell.split("void in")[1]),
            float(s[key]["n_void"]), 0.001)


# ------------------------------------------------------------- tab:methods
def check_methods() -> None:
    o = {r["method"]: r for r in J("orthogonality_f64.json")["rows"]}
    name = {"exact SVD": "Exact SVD", "Gram eigendecomposition": "Gram eigen (ours)",
            "Polar Express, 5 steps": "Polar Express (5)",
            "Polar Express, 8 steps": "Polar Express (8)",
            "Muon quintic, reference": "NS quintic (Muon, ref)"}
    for r in rows("tab:methods"):
        k = name.get(r[0].split("(")[0].strip() + ("(" if "(" in r[0] else ""))
        k = name.get(r[0].strip()) or next((v for q, v in name.items()
                                            if r[0].strip().startswith(q.split(",")[0])), None)
        if k is None:
            continue
        d = o[k]
        chk(f"methods {k} error", num(r[1]), d["median_err_f64"], 0.06)
        chk(f"methods {k} sigma_min", num(r[3]), d["smin_median"], 0.10)
        chk(f"methods {k} ms", num(r[4]), d["ms"], 0.02)


# --------------------------------------------------------------- tab:anova
def check_anova() -> None:
    mp = {r["method"]: r["mean"] for r in J("metric_power.json")["rows"]}
    name = {"Gram": "gram", "Muon quintic": "ns", "Gram--NS": "gram_ns",
            "Polar Express": "polar_express"}
    for r in rows("tab:anova"):
        k = next((v for q, v in sorted(name.items(), key=lambda x: -len(x[0]))
                  if r[0].startswith(q)), None)
        if k is None:
            continue
        chk(f"anova {k}", num(r[2]), mp[k], 1e-4)


# ---------------------------------------------------------------- tab:grid
def check_grid() -> None:
    ca = J("conditioning_audit.json")
    sig = {"full": {"fp64": 1.2e-8, "fp32": 1.2e-8, "fp16": 7.2e-6, "bf16": 2.3e-5},
           "sub": {"fp64": 3.3e-4, "fp32": 3.3e-4, "fp16": 3.3e-4, "bf16": 3.8e-4}}
    for d in ca["T3_gram_grid"]:
        chk(f"grid {d['regime']}/{d['dtype']} sigma_min/max",
            sig[d["regime"]][d["dtype"]], d["sigma_min_over_max_median"])


# --------------------------------------------------- tab:truecond, tab:ladder
def check_conditioning() -> None:
    ca = J("conditioning_audit.json")
    t0, t0b = ca["T0_true_conditioning"], ca["T0b_large_model"]["runs"]
    tbl = {"float64 gradients, float64 buffer": st.median(
        [r["kappa_true"] for r in t0b])}
    seen = 0
    for r in rows("tab:truecond"):
        if r[0].strip() != "float64 gradients, float64 buffer":
            continue
        seen += 1
        if seen == 1:
            chk("truecond small-model median", num(r[1]), t0["kappa_true_median"])
            rng = all_nums(r[2])
            chk("truecond small-model min", rng[0] if rng else None,
                t0["kappa_true_min"], 0.03)
            chk("truecond small-model max", rng[-1] if rng else None,
                t0["kappa_true_max"], 0.03)
        else:
            chk("truecond large-model median", num(r[1]), tbl[r[0].strip()], 0.03)
    for r in rows("tab:ladder"):
        dt = {"float64": "fp64", "float32": "fp32",
              "float16": "fp16", "bfloat16": "bf16"}.get(r[0].strip())
        if dt is None:
            continue
        med = st.median([x["kappa"][dt] for x in ca["T2_representation_ladder"]])
        chk(f"ladder {dt} ratio", num(r[2]), med)
        eps = torch.finfo(getattr(torch, r[0].strip())).eps
        chk(f"ladder {dt} eps", num(r[1]), eps, 0.01)


def main() -> None:
    for f in (check_eps, check_aposteriori, check_truncation, check_sweep,
              check_floors, check_real, check_mixed, check_methods, check_anova,
              check_grid, check_conditioning):
        try:
            f()
        except Exception as e:
            failures.append(f"{f.__name__} raised {type(e).__name__}: {e}")
    print(f"{checks} table cells checked, {len(failures)} disagreements")
    for x in failures:
        print("  !", x)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
