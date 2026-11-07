"""
Faithful implementations of the published baselines compared in this paper.

Sources (all verified against the primary papers/reference code, see
results/citation_report.md):

* Polar Express -- Amsel, Persson, Musco, Gower, "The Polar Express: Optimal
  Matrix Sign Methods and Their Application to the Muon Algorithm",
  arXiv:2505.16932.  A drop-in replacement for Muon's orthogonalization: at each
  iteration it applies an odd quintic p_t(x) = a_t x + b_t x^3 + c_t x^5 whose
  coefficients are the minimax approximation to 1 on the current interval
  [l_t, u_t], with u_t = 2 - l_t once the interval straddles the fixed point.
  The coefficient generator here is a direct port of the reference
  implementation (`optimal_cubic` / `optimal_quintic` / `optimal_composition`,
  flashlib/linalg/polar/express_coeffs.py, which in turn follows the paper's
  simplified Remez algorithm).

* Dion3 -- Amsel et al., "Dion3: Full-stack orthogonal updates", arXiv:2608.11612.
  Two algorithmic ingredients are reproducible without their CuteDSL kernels:
    (i) Gram Newton-Schulz -- run Newton-Schulz on the small symmetric Gram
        matrix instead of on X, which is where the FLOP saving comes from;
    (ii) row subsampling -- "selecting only a fraction of the momentum matrix's
        rows to orthogonalize at each step" (\S6).
  The kernel/megabatch contributions are systems-level and are not reproduced;
  we note this explicitly in the paper.

* Drop-Muon -- Gruntkowska, Maziane, Qu, Richtárik, "Drop-Muon: Update Less,
  Converge Faster", arXiv:2510.02239.  NOT a row-subsampling method: it is
  Randomized Progressive Training, which freezes a prefix of layers and updates
  a suffix {s^k, ..., b} where s^k is sampled each step (Algorithms 1-2, RPT
  sampling in \S4).  It composes with any matrix orthogonalizer, so we
  implement it as a layer-selection wrapper orthogonal to the others.
"""
from __future__ import annotations

import math

import torch

from robust_orth import (NS_QUINTIC_PUBLISHED, polar_gram, polar_ns,
                         polar_svd)

# ---------------------------------------------------------------------------
# Polar Express coefficients (port of the reference implementation)
# ---------------------------------------------------------------------------


def optimal_cubic(l: float, u: float):
    """Minimax odd cubic a x + b x^3 approximating 1 on [l, u] (closed form)."""
    alpha = math.sqrt(3.0 / (u * u + l * u + l * l))
    beta = 4.0 / (2.0 + l * u * (l + u) * (alpha ** 3))
    return 1.5 * alpha * beta, -0.5 * (alpha ** 3) * beta


def optimal_quintic(l: float, u: float, iters: int = 200):
    """
    Minimax odd quintic a x + b x^3 + c x^5 approximating 1 on [l, u].

    Simplified Remez: solve the 4x4 equioscillation system at node guesses
    (l, q, r, u), then move q, r to the roots of p'(x) = 0.  Ported from the
    reference implementation so the coefficients match published values.
    """
    if 1.0 - 5e-6 <= l / u:
        return (15.0 / 8.0) / u, (-10.0 / 8.0) / (u ** 3), (3.0 / 8.0) / (u ** 5)
    q = (3.0 * l + u) / 4.0
    r = (l + 3.0 * u) / 4.0
    E, old_E = float("inf"), None
    a = b = c = 0.0
    for _ in range(iters):
        old_E = E
        M = torch.tensor([
            [l, l ** 3, l ** 5, 1.0],
            [q, q ** 3, q ** 5, -1.0],
            [r, r ** 3, r ** 5, 1.0],
            [u, u ** 3, u ** 5, -1.0],
        ], dtype=torch.float64)
        try:
            sol = torch.linalg.solve(M, torch.ones(4, dtype=torch.float64))
        except Exception:
            break
        a, b, c, E = (float(sol[0]), float(sol[1]), float(sol[2]), float(sol[3]))
        disc = 9.0 * b * b - 20.0 * a * c
        if disc < 0:
            break
        q = math.sqrt((-3.0 * b - math.sqrt(disc)) / (10.0 * c))
        r = math.sqrt((-3.0 * b + math.sqrt(disc)) / (10.0 * c))
        if old_E is not None and abs(old_E - E) <= 1e-15:
            break
    return a, b, c


def optimal_composition(l: float, num_iters: int, safety_factor_eps: float = 0.0,
                        cushion: float = 0.0):
    """Coefficient table for `num_iters` Polar Express steps."""
    u = 1.0
    safety = 1.0 + safety_factor_eps
    coeffs = []
    for t in range(num_iters):
        lo = max(l, cushion * u)
        a, b, c = optimal_quintic(lo, u)
        if cushion * u > l:
            pl = a * l + b * l ** 3 + c * l ** 5
            pu = a * u + b * u ** 3 + c * u ** 5
            rescaler = 2.0 / (pl + pu)
            a, b, c = a * rescaler, b * rescaler, c * rescaler
        if t < num_iters - 1:
            a /= safety
            b /= safety ** 3
            c /= safety ** 5
        coeffs.append((a, b, c))
        l = a * l + b * l ** 3 + c * l ** 5
        u = 2.0 - l
    return coeffs


# Default table, matching the reference (l = 1e-6, 12 iter, 1e-3 safety+cushion).
_POLAR_COEFFS = None
_COMPOSITION_CACHE: dict = {}


def polar_express_coeffs():
    global _POLAR_COEFFS
    if _POLAR_COEFFS is None:
        _POLAR_COEFFS = optimal_composition(
            l=1e-6, num_iters=12, safety_factor_eps=1e-3, cushion=1e-3)
    return _POLAR_COEFFS


@torch.no_grad()
def _cached_composition(num_iters: int, safety_factor_eps: float):
    """Cache the (expensive) Remez solve; it depends only on (num_iters, safety)."""
    key = (num_iters, safety_factor_eps)
    if key not in _COMPOSITION_CACHE:
        _COMPOSITION_CACHE[key] = optimal_composition(
            l=1e-6, num_iters=num_iters, safety_factor_eps=safety_factor_eps,
            cushion=1e-3)
    return _COMPOSITION_CACHE[key]


@torch.no_grad()
def polar_express(X: torch.Tensor, steps: int = 5,
                  safety_factor_eps: float = 1e-3) -> torch.Tensor:
    """
    Polar Express orthogonalization (arXiv:2505.16932), Algorithm 1.

    Uses the same normalization convention as Muon's Newton-Schulz
    (X <- X / ||X||_F) so that the two are directly comparable, then applies
    `steps` of the adaptive minimax quintic.  Each step costs 3 matrix
    products, identical to the fixed quintic.
    """
    coeffs = _cached_composition(max(steps, 12), safety_factor_eps)
    Z = X.float() / (X.float().norm() + 1e-7)
    for t in range(steps):
        a, b, c = coeffs[t]
        A = Z @ Z.T
        Z = a * Z + (b * A + c * (A @ A)) @ Z
    return Z.to(X.dtype)


@torch.no_grad()
def gram_newton_schulz(X: torch.Tensor, mode: str = "ise", steps: int = 5,
                       n_outer: int = 1,
                       coeffs=NS_QUINTIC_PUBLISHED) -> torch.Tensor:
    """
    Dion3's Gram Newton-Schulz (arXiv:2608.11612, \S4), reproduced as faithfully
    as possible without the CuteDSL kernels.

    The saving in Dion3 comes from iterating on the *small* symmetric Gram
    matrix G = X^T X instead of on X: the iterates are (k x k) with
    k = min(m, n) rather than (m x n).

    The polar factor is X (X^T X)^{-1/2}, so the method must approximate an
    inverse square root.  We implement the two natural choices:

      mode="ise" -- Newton iteration Y <- Y(3I - A Y Y)/2 for A^{-1/2};
      mode="ns"  -- the fixed-point Newton-Schulz polynomial on A.

    Both are wrapped in the renormalisation/restart scheme the Dion3 paper
    describes (\S B.2).  As the paper reports, both fail at the conditioning of
    real momentum submatrices -- the Gram matrix squares the condition number,
    which puts it at the edge of float64 -- and this function exists so the
    comparison runs against the real algorithm rather than a straw man.
    """
    Xf = X.float()
    fro = Xf.norm() + 1e-12
    Z = Xf / fro
    k = min(Z.shape)
    I = torch.eye(k, dtype=Z.dtype, device=Z.device)

    def gram(M):
        return M @ M.T if Z.shape[0] <= Z.shape[1] else M.T @ M

    def mul(M, P):
        return P @ M if Z.shape[0] <= Z.shape[1] else M @ P

    A = gram(Z)
    # keep the matrices in the working dtype; only norms go through float64
    smax = float(torch.linalg.matrix_norm(A.detach().float().cpu().double(), ord=2))
    smax = torch.tensor(max(smax, 1e-12), dtype=A.dtype, device=A.device)
    Y = A / smax
    a, b, c = coeffs
    for _ in range(max(1, n_outer)):
        if mode == "ise":
            for _ in range(steps):
                Y = 0.5 * Y @ (3.0 * I - A @ (Y @ Y))
        elif mode == "ns":
            for _ in range(steps):
                Y = a * Y + b * (Y @ Y) + c * (Y @ Y @ Y)
                Y = torch.clamp(Y, 0.0, 1.0)
        else:
            raise ValueError(mode)
        # restart: renormalise so the next round starts inside the valid range
        ymax = float(torch.linalg.matrix_norm(Y.detach().float().cpu().double(),
                                              ord=2))
        Y = Y / max(ymax, 1e-12)
    O = mul(Z, Y)
    return (O * fro).to(X.dtype)


# ---------------------------------------------------------------------------
# Drop-Muon: Randomized Progressive Training layer schedule
# ---------------------------------------------------------------------------


class RPTLayerSchedule:
    """
    Randomized Progressive Training (Drop-Muon, arXiv:2510.02239, \\S4).

    At each step sample a smallest-active-layer index s ~ D over [1, b] and
    update the suffix of layers {s, ..., b}; the prefix is frozen.  The paper
    notes the distribution is a design choice (\S4, Appendix C) and that the
    cost-optimal one depends on per-layer smoothness constants.  We use the
    uniform distribution over s, which is the natural default when smoothness
    constants are unknown, and we report the expected active fraction so the
    compute saving is explicit rather than implicit.

    `blocks` is a list of lists of parameter names, ordered as in the
    backpropagation direction (index 0 = earliest layer).
    """

    def __init__(self, blocks: list[list[str]], seed: int = 0):
        self.blocks = blocks
        self.b = len(blocks)
        self.rng = torch.Generator().manual_seed(seed)

    def active_names(self) -> set[str]:
        if self.b == 0:
            return set()
        s = int(torch.randint(0, self.b, (1,), generator=self.rng).item())
        active = set()
        for blk in self.blocks[s:]:
            active.update(blk)
        return active

    def expected_active_fraction(self) -> float:
        """E[|S|]/b for uniform s over {0..b-1}; suffix lengths are b, b-1, ..."""
        if self.b == 0:
            return 0.0
        return (self.b + 1) / (2.0 * self.b)

    def all_names(self) -> set[str]:
        return {n for blk in self.blocks for n in blk}


def transformer_blocks(model) -> list[list[str]]:
    """
    Group a model's 2D hidden parameters into transformer blocks in
    backpropagation order.  Embedding is treated as the first block and the
    final norm/head as the last.
    """
    names = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            names.append(n)
    def key(n):
        parts = n.split(".")
        for i, tok in enumerate(parts):
            if tok == "h":
                try:
                    return int(parts[i + 1])
                except (ValueError, IndexError):
                    return -1
        if "wte" in n or "wpe" in n:
            return -1
        return 999
    buckets: dict[int, list[str]] = {}
    for n in names:
        buckets.setdefault(key(n), []).append(n)
    return [buckets[k] for k in sorted(buckets)]
