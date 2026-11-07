"""
The optimizer under test: Conditioned-Subsampled Muon (CS-Muon).

Mechanism (all measured, see results/FINDINGS.md and figures/):

  1. Real momentum matrices have sigma_min/sigma_max ~ 1e-8.  Every fixed
     polynomial orthogonalizer fails there: Muon's published quintic has
     p(1) = 0.701 and, once its clamp(., 0, 1) guard is hit by the whole
     matrix, the relative spectrum is destroyed and the update collapses.
  2. Orthogonalizing a *row-subsampled* submatrix repairs the conditioning by
     ~4 orders of magnitude (1e-9 -> 1e-4 for rho = 0.15 on measured matrices).
  3. That repair is what makes a Gram-based exact polar factor viable: the Gram
     matrix has condition number kappa^2, so in float64 it needs kappa <~ 1e8.
     At full scale (kappa = 5e7 -> kappa^2 = 2.5e15) it fails; at rho = 0.05 it
     is exact to 1e-7 and takes 1.0 ms instead of 55 ms for the SVD.
  4. Rows outside the selected set receive a scale-matched coordinate-wise
     step, so no parameter is left without an update.

The selection rule is a curvature proxy (EMA of row energy times relative row
change).  We test it against plain row-norm and uniform-random selection so the
contribution's actual source is identifiable rather than asserted.
"""
from __future__ import annotations

import math

import torch

import robust_orth as ro


class CSMuon:
    """
    Hybrid optimizer:
      * 1D parameters (bias, LayerNorm) and embeddings -> AdamW;
      * 2D hidden matrices -> subsampled exact-polar orthogonalization;
      * rows outside the selected set -> coordinate-wise normalised step.
    """

    def __init__(self, model, *, lr_adam: float = 3e-4, lr_muon: float = 0.02,
                 weight_decay: float = 0.1, momentum: float = 0.95,
                 rho: float = 0.15, select: str = "curvature",
                 beta: float = 0.98, gamma: float = 0.5,
                 orth: str = "gram", seed: int = 0,
                 scale_lr_by_shape: bool = True,
                 on_device_gram: bool = True, gram_floor: float = 0.0,
                 update_rms_multiplier: float = 1.0):
        self.lr_adam = lr_adam
        self.lr_muon = lr_muon
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.rho = rho
        self.select = select
        self.beta = beta
        self.gamma = gamma
        self.orth = orth
        self.seed = seed
        self.scale_lr_by_shape = scale_lr_by_shape
        self.on_device_gram = on_device_gram
        self.gram_floor = gram_floor
        self.update_rms_multiplier = update_rms_multiplier

        if hasattr(model, "param_groups_by_shape"):
            matrix, other = model.param_groups_by_shape()
        else:
            # Generic fallback: orthogonalize 2D weights that are neither
            # embeddings nor the (tied) output projection.
            matrix, other = [], []
            skip = ("wte", "wpe", "lm_head")
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 2 and not any(k in name for k in skip):
                    matrix.append((name, p))
                else:
                    other.append((name, p))
        self.matrix = matrix
        self.other = other

        self._adam: dict[int, dict] = {}
        self._muon: dict[int, dict] = {}
        self.orth_stats: list[dict] = []
        # running estimate of the update RMS actually applied, used to
        # build an update-norm-matched control for other optimizers
        self.update_rms: float = 0.0
        self._rms_count: int = 0

    # ------------------------------------------------------------------
    def zero_grad(self):
        for _, p in self.matrix:
            p.grad = None
        for _, p in self.other:
            p.grad = None

    # ------------------------------------------------------------------
    def _adam_step(self, p, lr, state_key):
        g = p.grad
        st = self._adam.setdefault(state_key, {
            "m": torch.zeros_like(g), "v": torch.zeros_like(g), "t": 0})
        st["t"] += 1
        t = st["t"]
        b1, b2, eps = 0.9, 0.95, 1e-8
        st["m"].mul_(b1).add_(g, alpha=1 - b1)
        st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
        mhat = st["m"] / (1 - b1 ** t)
        vhat = st["v"] / (1 - b2 ** t)
        upd = mhat / (vhat.sqrt() + eps)
        p.add_(upd, alpha=-lr)
        if self.weight_decay:
            p.add_(p, alpha=-lr * self.weight_decay)
        return upd

    # ------------------------------------------------------------------
    def _orthogonalize(self, sub: torch.Tensor) -> torch.Tensor:
        import baselines as bl
        fn = {"gram": ro.polar_gram,                 # ours: exact Gram eigen
              "svd": ro.polar_svd,                   # exact SVD reference
              "ns": lambda z: ro.polar_ns(z, 5, ro.NS_QUINTIC_PUBLISHED,
                                          "frobenius", True),
              "ns_cubic": lambda z: ro.polar_ns(z, 5, ro.NS_CUBIC,
                                                "frobenius", True),
              # published baselines
              "polar_express": lambda z: bl.polar_express(z, steps=5),
              "gram_ns": lambda z: bl.gram_newton_schulz(z, mode="ns", steps=5),
              "gram_ise": lambda z: bl.gram_newton_schulz(z, mode="ise",
                                                          steps=10),
              }.get(self.orth)
        if fn is None and self.orth == "dion3":
            # Dion3's update rule = row subsampling + Gram Newton-Schulz
            # (arXiv:2608.11612, §6).  Row subsampling is already handled by the
            # caller (rho/select), so this branch only fixes the orthogonalizer.
            fn = lambda z: bl.gram_newton_schulz(z, mode="ns", steps=5)
        if fn is None:
            raise ValueError(self.orth)
        return fn(sub)

    def _orthogonalize_batch(self, subs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Apply the orthogonalizer to a list of submatrices that share a shape.

        `polar_gram` admits a batched form: stack (b, k, n), form the b Gram
        matrices with one batched matmul, run a single batched eigendecomposition,
        and reassemble.  On this hardware that turns ~24 sequential device
        round trips per optimizer step into 3 batched ones.
        """
        if self.orth not in ("gram",):
            return [self._orthogonalize(s) for s in subs]
        S = torch.stack([s.detach().float() for s in subs], dim=0)   # (b,k,n)
        # Precision: see robust_orth.polar_gram_batched.  The Gram eigen-solve
        # must be done in float64, and doing it on-device in reduced precision
        # silently produces a wrong update.  We batch across all parameters that
        # share a shape so this costs one round trip per shape, not per
        # parameter.
        # Prefer the device-native float64 path when available: it is
        # numerically identical to the CPU path and ~1.7x faster per step
        # (see robust_orth.polar_gram_on_device).
        if self.on_device_gram and torch.cuda.is_available():
            out = ro.polar_gram_on_device(S, floor=self.gram_floor,
                                          dtype=torch.float64)
        else:
            out = ro.polar_gram_batched(S, floor=self.gram_floor,
                                        dtype=torch.float64)
        return [out[i] for i in range(out.shape[0])]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, lr_scale: float = 1.0, collect_stats: bool = False):
        for name, p in self.other:
            if p.grad is None:
                continue
            self._adam_step(p, self.lr_adam * lr_scale, ("other", id(p)))

        # Pass 1: momentum, selection weights, and row selection.  The
        # orthogonalization is deferred so that submatrices sharing a shape can
        # be processed as one batched call.
        jobs = []
        for name, p in self.matrix:
            if p.grad is None:
                continue
            G = p.grad
            m, n = G.shape
            transposed = m > n
            X = G.T if transposed else G
            krows = X.shape[0]

            st = self._muon.setdefault(id(p), {
                "buf": torch.zeros_like(X), "prev": None, "vema": None})
            st["buf"].mul_(self.momentum).add_(X, alpha=1 - self.momentum)
            B = st["buf"]

            # Row-selection weights.  The curvature proxy is an EMA of row
            # energy times the relative row change; norm/random are ablations.
            if self.select == "curvature":
                row_sq = (B * B).sum(dim=1)
                if st["vema"] is None:
                    st["vema"] = row_sq.clone()
                else:
                    st["vema"].mul_(self.beta).add_(row_sq, alpha=1 - self.beta)
                if st["prev"] is None:
                    w = st["vema"].clone()
                else:
                    num = torch.linalg.vector_norm(B - st["prev"], dim=1)
                    den = torch.linalg.vector_norm(st["prev"], dim=1) + 1e-8
                    w = st["vema"] * (1.0 + self.gamma * (num / den))
            else:
                w = None

            idx = ro.select_rows(B, self.rho, w, mode=self.select, seed=self.seed)
            jobs.append({"name": name, "p": p, "B": B, "idx": idx, "st": st,
                         "m": m, "n": n, "transposed": transposed,
                         "krows": krows, "sub": B[idx]})

        # Pass 2: batched orthogonalization, grouped by submatrix shape.
        groups: dict[tuple[int, int], list[int]] = {}
        for j, job in enumerate(jobs):
            groups.setdefault(tuple(job["sub"].shape), []).append(j)
        orth: dict[int, torch.Tensor] = {}
        for shape, js in groups.items():
            outs = self._orthogonalize_batch([jobs[j]["sub"] for j in js])
            for j, o in zip(js, outs):
                orth[j] = o

        # Pass 3: assemble and apply.
        for j, job in enumerate(jobs):
            B, idx, st = job["B"], job["idx"], job["st"]
            m, n, krows = job["m"], job["n"], job["krows"]
            O_sub = orth[j]

            U = torch.empty_like(B)
            U[idx] = O_sub
            if idx.numel() < krows:
                rest = torch.ones(krows, dtype=torch.bool, device=B.device)
                rest[idx] = False
                tail = B[rest]
                U[rest] = tail / (tail.norm(dim=1, keepdim=True) + 1e-12) * math.sqrt(n)

            # Normalise the assembled update to unit elementwise RMS.
            #
            # The orthogonalized block has elementwise RMS 1/sqrt(n) while the
            # tail block was normalised to RMS 1, and the blocks have different
            # sizes, so ||U||_F has no fixed relationship to the gradient scale
            # (measured: ||U||_F / ||G||_F ~ 5e2).  Applying lr_muon to that
            # makes the effective step size arbitrary and shape dependent.
            # Dividing by the measured RMS gives an update with elementwise RMS
            # exactly 1 -- the same convention as AdamW -- so that lr_muon is
            # directly comparable to lr_adam.
            U = U / (U.pow(2).mean().sqrt() + 1e-12)
            # record the applied update RMS so a control optimizer can be run at
            # matched update norm (this is the control that arXiv:2607.16169
            # argues is missing from most Muon comparisons)
            self.update_rms = (self.update_rms * self._rms_count
                               + float(U.pow(2).mean().sqrt())) / (self._rms_count + 1)
            self._rms_count += 1
            step_lr = self.lr_muon * lr_scale * self.update_rms_multiplier

            if collect_stats:
                self.orth_stats.append({
                    "name": job["name"],
                    "shape": [int(m), int(n)],
                    "k": int(idx.numel()),
                    "rho_actual": float(idx.numel()) / krows,
                    "orth_err_sub": ro.orthogonality_error(O_sub),
                })

            st["prev"] = B.clone()
            U = U.T if job["transposed"] else U
            job["p"].add_(U, alpha=-step_lr)
            if self.weight_decay:
                # decoupled weight decay; the update above is subtracted, so the
                # decay must *shrink* the parameter: multiply by (1 + lr*wd)
                # corresponds to adding -lr*wd*W.
                job["p"].add_(job["p"], alpha=-step_lr * self.weight_decay)
