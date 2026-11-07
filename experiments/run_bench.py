"""
End-to-end training benchmark for CS-Muon.

Runs a small GPT and compares optimizers on identical batches/seeds, recording
training and validation loss, throughput, and the measured orthogonality error
of the orthogonalized submatrix.

Usage:
    python run_bench.py --config configs/main.json --out results/main.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import data as data_mod
from csmuon import CSMuon
from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def pick_device(pref="auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class BenchConfig:
    name: str = "run"
    optimizer: str = "cs_muon"     # adamw | cs_muon
    seed: int = 0
    max_steps: int = 1500
    batch_size: int = 32
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    lr_adam: float = 3e-4
    lr_muon: float = 0.02
    weight_decay: float = 0.1
    warmup: int = 100
    rho: float = 0.15
    select: str = "curvature"      # curvature | norm | random | all
    orth: str = "gram"             # gram | svd | ns | ns_cubic
    momentum: float = 0.95
    beta: float = 0.98
    gamma: float = 0.5
    eval_every: int = 100
    eval_iters: int = 20
    log_every: int = 25
    dataset: str = "shakespeare"
    data_bytes: int | None = None
    device: str = "auto"
    grad_clip: float = 1.0
    collect_orth_stats: bool = False


def lr_scale_at(step: int, cfg: BenchConfig) -> float:
    if step < cfg.warmup:
        return (step + 1) / max(cfg.warmup, 1)
    prog = min(max((step - cfg.warmup) / max(cfg.max_steps - cfg.warmup, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def load_data(cfg: BenchConfig):
    if cfg.dataset == "shakespeare":
        return data_mod.prepare_shakespeare()
    r = data_mod.prepare_tinystories(max_bytes=cfg.data_bytes)
    if r is None:
        raise RuntimeError("TinyStories unavailable")
    return r


@torch.no_grad()
def evaluate(model, val, cfg, device, gen) -> float:
    model.eval()
    ls = []
    for _ in range(cfg.eval_iters):
        x, y = data_mod.get_batch(val, cfg.block_size, cfg.batch_size, device, gen)
        _, loss = model(x, y)
        ls.append(float(loss))
    model.train()
    return float(np.mean(ls))


def make_optimizer(model, cfg: BenchConfig, device):
    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_adam,
                                betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
        return ("adamw", opt)
    if cfg.optimizer == "muon_ns":
        # Standard Muon: orthogonalize the FULL matrix with the published
        # Newton-Schulz quintic.  Identical LR scaling to CS-Muon so the only
        # difference is the orthogonalization itself.
        return ("cs_muon", CSMuon(
            model, lr_adam=cfg.lr_adam, lr_muon=cfg.lr_muon,
            weight_decay=cfg.weight_decay, momentum=cfg.momentum,
            rho=1.0, select="all", beta=cfg.beta, gamma=cfg.gamma,
            orth="ns", seed=cfg.seed))
    if cfg.optimizer == "cs_muon":
        return ("cs_muon", CSMuon(
            model, lr_adam=cfg.lr_adam, lr_muon=cfg.lr_muon,
            weight_decay=cfg.weight_decay, momentum=cfg.momentum,
            rho=cfg.rho, select=cfg.select, beta=cfg.beta, gamma=cfg.gamma,
            orth=cfg.orth, seed=cfg.seed))
    raise ValueError(cfg.optimizer)


def run_one(cfg: BenchConfig, train, val, vocab) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    gen = torch.Generator().manual_seed(cfg.seed + 1)

    gpt = GPT(GPTConfig(vocab_size=vocab, block_size=cfg.block_size,
                        n_layer=cfg.n_layer, n_head=cfg.n_head,
                        n_embd=cfg.n_embd)).to(device)
    kind, opt = make_optimizer(gpt, cfg, device)

    history, step_times = [], []
    tokens, t0 = 0, time.time()
    diverged = False

    for step in range(cfg.max_steps):
        ts = time.time()
        scale = lr_scale_at(step, cfg)
        x, y = data_mod.get_batch(train, cfg.block_size, cfg.batch_size, device, gen)
        _, loss = gpt(x, y)
        lval = float(loss.detach())

        if kind == "adamw":
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(gpt.parameters(), cfg.grad_clip)
            for pg in opt.param_groups:
                pg["lr"] = cfg.lr_adam * scale
            opt.step()
        else:
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(gpt.parameters(), cfg.grad_clip)
            opt.step(lr_scale=scale,
                     collect_stats=cfg.collect_orth_stats and step % cfg.log_every == 0)

        tokens += x.numel()
        step_times.append(time.time() - ts)

        if not math.isfinite(lval):
            diverged = True
            print(f"  [{cfg.name}] DIVERGED at step {step} (loss={lval})", flush=True)
            break

        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            rec = {"step": step, "train_loss": lval, "lr_scale": scale,
                   "tokens": tokens, "elapsed": time.time() - t0,
                   "step_time_ms": 1000 * float(np.median(step_times[-50:]))}
            if step % cfg.eval_every == 0 or step == cfg.max_steps - 1:
                vl = evaluate(gpt, val, cfg, device, gen)
                rec["val_loss"] = vl
                rec["val_ppl"] = float(math.exp(min(vl, 20)))
            history.append(rec)
            if step % (cfg.log_every * 8) == 0:
                print(f"  [{cfg.name}] step {step:5d} train {lval:.4f} "
                      f"val {rec.get('val_loss', float('nan')):.4f} "
                      f"{tokens/(time.time()-t0):,.0f} tok/s", flush=True)

    wall = time.time() - t0
    out = {
        "config": asdict(cfg),
        "optimizer": cfg.optimizer,
        "n_params": gpt.num_params(),
        "history": history,
        "diverged": diverged,
        "wall_time_s": wall,
        "tokens_seen": tokens,
        "tokens_per_s": tokens / max(wall, 1e-9),
        "median_step_ms": 1000 * float(np.median(step_times)) if step_times else None,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1].get("val_loss") if history else None,
        "device": str(device),
    }
    if kind == "cs_muon" and cfg.collect_orth_stats and opt.orth_stats:
        errs = [s["orth_err_sub"] for s in opt.orth_stats
                if math.isfinite(s["orth_err_sub"])]
        out["orth_stats"] = {
            "n": len(opt.orth_stats),
            "orth_err_sub_median": float(np.median(errs)) if errs else None,
            "orth_err_sub_max": float(np.max(errs)) if errs else None,
            "k_median": float(np.median([s["k"] for s in opt.orth_stats])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON list of BenchConfig dicts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    specs = json.loads(Path(args.config).read_text())
    if args.limit:
        specs = specs[: args.limit]

    first = BenchConfig(**{k: v for k, v in specs[0].items()
                           if k in BenchConfig.__dataclass_fields__})
    first.device = args.device
    print(f"loading {first.dataset} ...", flush=True)
    train, val, vocab = load_data(first)

    results = []
    for spec in specs:
        cfg = BenchConfig(**{k: v for k, v in spec.items()
                             if k in BenchConfig.__dataclass_fields__})
        cfg.device = args.device
        print(f"\n=== {cfg.name} ({cfg.optimizer}, {cfg.orth}, "
              f"select={cfg.select}, rho={cfg.rho}) ===", flush=True)
        res = run_one(cfg, train, val, vocab)
        results.append(res)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  -> final val {res['final_val_loss']}  "
              f"{res['tokens_per_s']:,.0f} tok/s", flush=True)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
