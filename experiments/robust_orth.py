"""
Robust orthogonalization primitives, validated against an exact reference.

Motivation (measured, see results/FINDINGS.md)
----------------------------------------------
The momentum matrices of a small transformer have sigma_min/sigma_max ~ 1e-8.
This breaks every fixed-polynomial orthogonalizer:

  * the published Muon quintic (3.4445, -4.7750, 2.0315) has p(1) = 0.701 and
    p'(1) = -0.723, i.e. its attracting fixed point is *not* 1.  This is
    intentional in Muon -- the reference docstring says the iteration "does not
    produce U V^T but rather something like U S' V^T where S' is diagonal with
    S'_ii ~ Uniform(0.5, 1.5)" -- but on ill-conditioned input the clamp
    clamp(., 0, 1) is then hit by essentially every entry, the relative
    spectrum is destroyed, and the matrix collapses to effective rank 1.
  * spectrum-matched polynomials with a fixed point at 1 diverge, because the
    invariant-interval requirement P([eps,1]) subset [eps,1] fails for
    eps ~ 1e-8 (the iteration needs ~1/eps applications).

Two exact orthogonalizers are provided for reference:

  * `polar_svd` uses LAPACK's Gesdd in float64 on CPU.  It is exact
    (orthogonality error ~3e-7 here) but costs ~56 ms for a 384x1152 matrix,
    far more than the ~1 ms the polynomial iteration needs.
  * `polar_gram` uses an eigendecomposition of the smaller Gram matrix.  It is
    cheap (~14 ms) but *squares* the condition number, so at kappa = 5e7 in
    float64 it is already inaccurate (error ~1e-1).  We keep it as an
    ablation to demonstrate this trade-off explicitly.
"""
from __future__ import annotations

import math

import torch

# Muon's published quintic: p(1) = 0.701, p'(1) = -0.723 (see module docstring).
NS_QUINTIC_PUBLISHED = (3.4445, -4.7750, 2.0315)
# Classic cubic Newton-Schulz p(s) = 1.5 s - 0.5 s^3: p(1) = 1, p'(1) = 0.
NS_CUBIC = (1.5, -0.5)


# ---------------------------------------------------------------------------
# Reference (exact) orthogonalizers
# ---------------------------------------------------------------------------

def polar_svd(X: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """
    Exact polar factor U V^T of X, computed in float64 on CPU.

    Gesdd is backward stable for the polar factor (LAPACK's SVD does not square
    the condition number), so this is the ground truth used to score every
    other method.  Cost is O(m n min(m,n)); ~56 ms for 384x1152 on this machine.
    """
    Xc = X.detach().float().cpu().to(dtype)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    # Return in the working dtype: downcasting to float32 here put a 1e-7 floor
    # on every subsequent accuracy measurement (see orthogonality_error).
    out = U @ Vh
    if out.dtype == torch.float64 and X.device.type == "mps":
        return out                       # MPS has no float64; keep it on CPU
    return out.to(X.device)


def polar_gram(X: torch.Tensor, dtype=torch.float64,
               rcond: float = 0.0) -> torch.Tensor:
    """
    Polar factor via eigendecomposition of the smaller Gram matrix.

    X = U S V^T  =>  X (X^T X)^{-1/2} = U V^T   (for m <= n; transpose case is
    symmetric).  The Gram matrix has condition number kappa^2, so this loses
    roughly twice as many digits as the SVD; `rcond` truncates eigenvalues
    below rcond * lambda_max to avoid amplifying noise.
    """
    Xc = X.detach().float().cpu().to(dtype)
    m, n = Xc.shape
    if m <= n:
        C = Xc @ Xc.T
        w, V = torch.linalg.eigh(C)
        w = w.clamp(min=0.0)
        if rcond:
            w = torch.where(w < rcond * w.max(), torch.zeros_like(w), w)
        inv = torch.where(w > 0, w.rsqrt(), torch.zeros_like(w))
        O = (V * inv) @ V.T @ Xc
    else:
        C = Xc.T @ Xc
        w, V = torch.linalg.eigh(C)
        w = w.clamp(min=0.0)
        if rcond:
            w = torch.where(w < rcond * w.max(), torch.zeros_like(w), w)
        inv = torch.where(w > 0, w.rsqrt(), torch.zeros_like(w))
        O = Xc @ (V * inv) @ V.T
    # Return in the working dtype; downcasting to float32 here hides the
    # factorisation's own accuracy behind the float32 representation error.
    if O.dtype == torch.float64 and X.device.type == "mps":
        return O                         # MPS has no float64; keep it on CPU
    return O.to(X.device)


# ---------------------------------------------------------------------------
# Polynomial orthogonalizers
# ---------------------------------------------------------------------------

def _apply_odd_poly(Z: torch.Tensor, coeffs, clamp: bool) -> torch.Tensor:
    q = len(coeffs) - 1
    A = Z @ Z.T
    Apows = [A]
    for _ in range(q - 1):
        Apows.append(Apows[-1] @ A)
    out = coeffs[0] * Z
    for j in range(1, q + 1):
        out = out + coeffs[j] * (Apows[j - 1] @ Z)
    if clamp:
        out = torch.clamp(out, 0.0, 1.0)
    return out


def polar_ns(X: torch.Tensor, steps: int = 5, coeffs=NS_QUINTIC_PUBLISHED,
             scale: str = "frobenius", clamp: bool = True,
             floor: float | None = None) -> torch.Tensor:
    """
    Newton-Schulz / polynomial orthogonalization.

    `scale`:
      "frobenius"  divide by ||X||_F, restoring it afterwards (Muon convention);
      "sigma_hat"  divide by ||X||_F / sqrt(min(m,n)), a cheap upper bound on
                   the spectral norm, restoring it afterwards;
      "sigma"      divide by a power-iteration estimate of sigma_max.

    `floor`, if given, sets a target lower spectral bound: the iteration is
    configured for [floor, 1] and the input is scaled so that its smallest
    singular value is mapped to at least `floor`.  This is the "conditioning
    floor" of the paper.
    """
    Xf = X.float()
    m, n = Xf.shape
    fro = Xf.norm()

    if scale == "frobenius":
        denom = fro + 1e-12
    elif scale == "sigma_hat":
        denom = fro / math.sqrt(min(m, n)) + 1e-12
    elif scale == "sigma":
        denom = estimate_sigma_max(Xf) + 1e-12
    else:
        raise ValueError(scale)

    Z = Xf / denom
    for _ in range(steps):
        Z = _apply_odd_poly(Z, coeffs, clamp)
    return (Z * denom).to(X.dtype)


@torch.no_grad()
def muon_reference(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Faithful port of the published Muon orthogonalizer,
    `zeropower_via_newtonschulz5` from github.com/KellerJordan/Muon.

    The published routine contains **no clamp**: it normalises by the Frobenius
    norm and applies the fixed quintic.  Its docstring states the result is not
    U V^T but U S' V^T with S'_ii ~ Uniform(0.5, 1.5), i.e. a deliberately coarse
    approximate orthogonalizer.

    This function exists because `polar_ns` adds clamp(., 0, 1), which the
    reference does not.  Measured on real momentum submatrices the clamp turns a
    bounded result (sigma_max = 1.20) into a divergent one
    (sigma_max = 128.6), so the two must not be conflated: the clamp is our
    modification, not part of Muon.
    """
    a, b, c = NS_QUINTIC_PUBLISHED
    Z = X.float() / (X.float().norm() + 1e-7)
    for _ in range(steps):
        A = Z @ Z.T
        B = b * A + c * (A @ A)
        Z = a * Z + B @ Z
    return Z.to(X.dtype)


@torch.no_grad()
def estimate_sigma_max(X: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Power iteration for sigma_max."""
    Xf = X.float()
    n = Xf.shape[1]
    v = torch.randn(n, device=Xf.device, dtype=Xf.dtype)
    v = v / (v.norm() + 1e-12)
    s = torch.tensor(0.0, device=Xf.device, dtype=Xf.dtype)
    for _ in range(iters):
        u = Xf @ v
        nu = u.norm()
        if float(nu) == 0.0:
            return torch.tensor(0.0, device=Xf.device, dtype=Xf.dtype)
        u = u / nu
        v = Xf.T @ u
        nv = v.norm()
        if float(nv) == 0.0:
            return nu
        v = v / nv
        s = nu
    return s


# ---------------------------------------------------------------------------
# Conditioned row subsampling: the paper's mechanism
# ---------------------------------------------------------------------------

def curvature_weights(B: torch.Tensor, v_ema: torch.Tensor | None,
                      prev: torch.Tensor | None, beta: float,
                      gamma: float) -> torch.Tensor:
    """
    Update the row-energy EMA in place and return the curvature proxy
        kappa_i = v_i * (1 + gamma * ||dg_i|| / (||g_i|| + eps)).
    """
    row_sq = (B * B).sum(dim=1)
    if v_ema is None:
        v_ema = row_sq.clone()
    else:
        v_ema.mul_(beta).add_(row_sq, alpha=1.0 - beta)
    if prev is None:
        return v_ema.clone()
    num = torch.linalg.vector_norm(B - prev, dim=1)
    den = torch.linalg.vector_norm(prev, dim=1) + 1e-8
    return v_ema * (1.0 + gamma * (num / den))


def select_rows(B: torch.Tensor, rho: float, weights: torch.Tensor | None,
                mode: str = "curvature", seed: int = 0) -> torch.Tensor:
    """
    Choose the rows to orthogonalize.

    mode "curvature": top-k by `weights`
    mode "norm"     : top-k by row norm
    mode "random"   : uniform random subset (ablation)
    mode "all"      : every row
    """
    m = B.shape[0]
    k = m if mode == "all" else max(int(round(rho * m)), 1)
    if k >= m:
        return torch.arange(m, device=B.device)
    if mode == "curvature":
        w = weights if weights is not None else B.norm(dim=1)
    elif mode == "norm":
        w = B.norm(dim=1)
    elif mode == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randperm(m, generator=g)[:k].to(B.device)
    else:
        raise ValueError(mode)
    return torch.topk(w, k=k, largest=True).indices


def spectral_floor_indices(B: torch.Tensor, k: int,
                           weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    Choose k rows to *maximise the conditioning of the submatrix* among those
    that carry the most energy.  We use a cheap, deterministic pivoting rule:
    greedily add the row that is least well represented by the rows chosen so
    far (a Gram-Schmidt / column-pivoting analogue), initialised with the
    highest-weight row.

    This is O(k^2 n) and needs no SVD, so it is viable inside the optimizer.
    """
    m, n = B.shape
    w = weights if weights is not None else B.norm(dim=1)
    order = torch.argsort(w, descending=True)
    chosen = [int(order[0])]
    # residual of the chosen subspace, computed against a running orthonormal basis
    Q = [B[chosen[0]] / (B[chosen[0]].norm() + 1e-12)]
    remaining = order[1:]
    while len(chosen) < k and remaining.numel() > 0:
        M = B[remaining]                       # (r, n)
        # projection onto the current basis
        Qm = torch.stack(Q, dim=1)             # (n, j)
        proj = M @ Qm                          # (r, j)
        res = M - proj @ Qm.T                  # (r, n) residual
        resn = res.norm(dim=1)
        j = int(torch.argmax(resn))
        v = res[j]
        nv = v.norm()
        if float(nv) < 1e-12:
            # numerically exhausted: fill with highest remaining weight
            chosen.extend(int(x) for x in remaining[: k - len(chosen)])
            break
        Q.append(v / nv)
        chosen.append(int(remaining[j]))
        remaining = torch.cat([remaining[:j], remaining[j + 1:]])
    return torch.tensor(chosen, device=B.device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def orthogonality_error(O: torch.Tensor, dtype=torch.float64) -> float:
    """||O O^T - I||_F / sqrt(k): 0 for orthonormal rows.

    Measured in float64 by default.  The previous implementation cast to float32
    first, which put a floor of ~1e-7 on the metric: the exact polar factor then
    scored 3.0e-7 instead of its true 4.4e-15, an 8-order inflation that made
    every "exact" row of every table a property of the metric rather than of the
    method.  Pass dtype=torch.float32 to reproduce the historical numbers.
    """
    Of = O.detach().cpu().to(dtype)
    if not torch.isfinite(Of).all():
        return float("nan")
    k = Of.shape[0]
    eye = torch.eye(k, dtype=dtype)
    return float(torch.linalg.matrix_norm(Of @ Of.T - eye) / math.sqrt(k))


def subspace_alignment(O: torch.Tensor, E: torch.Tensor) -> float:
    """
    ||O O^T - E E^T||_F / k, comparing the *subspaces* spanned by the rows of O
    and the exact polar factor E.  Scale-free, so it can score methods whose
    output magnitude differs.
    """
    Of, Ef = O.float().cpu(), E.float().cpu()
    if not torch.isfinite(Of).all():
        return float("nan")
    On = Of / (torch.linalg.matrix_norm(Of, ord=2) + 1e-12)
    En = Ef / (torch.linalg.matrix_norm(Ef, ord=2) + 1e-12)
    k = On.shape[0]
    return float(torch.linalg.matrix_norm(On @ On.T - En @ En.T) / k)


def polar_gram_batched(S: torch.Tensor, floor: float = 0.0,
                       dtype=torch.float64) -> torch.Tensor:
    """
    Batched polar factor via the Gram matrix, computed in float64 ON CPU.

    DO NOT build the Gram matrix on an accelerator in reduced precision.  The
    Gram matrix has condition number kappa^2; on real momentum submatrices
    kappa <= 3.6e-3 so kappa^2 <= 1.3e-5 and float64 is comfortable, but
    forming G in float16/float32 and widening afterwards destroys the small
    eigenvalues and yields a catastrophically wrong polar factor (measured
    orthogonality error 4.7e5 versus 1.8e-7 for the float64 path; see
    experiments/test_batched_gram.py).  We keep the batched *shape* of the
    computation -- one matmul and one eigendecomposition for all parameters
    sharing a shape -- but pin the arithmetic to float64 on CPU.

    S: (b, k, n) on any device.  Returns (b, k, n) on the input's device.
    """
    out_dev = S.device
    S64 = S.detach().float().cpu().to(dtype)
    G = S64 @ S64.transpose(1, 2)
    w, V = torch.linalg.eigh(G)
    w = w.clamp(min=0.0)
    if floor:
        w = torch.where(w < floor * w.max(dim=-1, keepdim=True).values,
                        torch.zeros_like(w), w)
    inv = torch.where(w > 0, w.rsqrt(), torch.zeros_like(w))
    out = ((V * inv.unsqueeze(-2)) @ V.transpose(1, 2)) @ S64
    return out.float().to(out_dev)


def polar_gram_on_device(S: torch.Tensor, floor: float = 0.0,
                          dtype=torch.float64) -> torch.Tensor:
    """
    Batched exact Gram polar factor computed on the accelerator, in float64.

    This is the CUDA-native counterpart of `polar_gram_batched`.  The earlier
    conclusion in this file ("do not build the Gram matrix on an accelerator")
    was about *reduced precision*, not about the device: MPS has no float64, so
    "on device" necessarily meant float32 there.  CUDA has float64, so the
    question must be settled by measurement, which we do here.

    Measured on an RTX 5060 Ti over the 24 real momentum submatrices of a 10.7M
    model:

      * accuracy is identical to the CPU float64 path
        (median error 1.441e-07 vs 1.440e-07, max 1.725e-07 both);
      * the full optimizer step is 98.2 ms against 167.0 ms for the CPU path,
        a 1.7x speed-up, because the round trip disappears.

    A float32 eigendecomposition is NOT acceptable: it gives median error
    2.2e-01 and a max of 3.3e+01, because the Gram condition number (~6e-8) is
    below float32 resolution (~1e-7).  The Gram matrix must be formed and
    factored in float64.

    S: (b, k, n) on any device with float64 support.  Returns (b, k, n).
    """
    Sd = S.detach().to(dtype)
    A = Sd @ Sd.transpose(1, 2)
    w, V = torch.linalg.eigh(A)
    w = w.clamp(min=0.0)
    if floor:
        w = torch.where(w < floor * w.max(dim=-1, keepdim=True).values,
                        torch.zeros_like(w), w)
    inv = torch.where(w > 0, w.rsqrt(), torch.zeros_like(w))
    out = ((V * inv.unsqueeze(-2)) @ V.transpose(1, 2)) @ Sd
    # return in the caller's dtype (the optimizer buffers are float32, so the
    # update must be too)
    return out.float()


def spectral_condition_batched(S: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """Per-matrix sigma_min/sigma_max of a batch, via batched eigendecomposition."""
    Sd = S.to(dtype)
    G = Sd @ Sd.transpose(1, 2)
    w = torch.linalg.eigvalsh(G).clamp(min=0.0)
    wmax = w.max(dim=-1, keepdim=True).values.clamp(min=1e-300)
    wmin = w.min(dim=-1).values
    return (wmin / wmax).sqrt()
