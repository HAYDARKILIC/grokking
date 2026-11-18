#!/usr/bin/env python3
"""
CPU SCOUTING RUN — (P4) init-scale robustness.

This is NOT manuscript data. It runs on CPU in float64, whereas every
number in the manuscript came from an A100 in float64. The manuscript
itself argues (citing Ootani 2026) that CPU-versus-GPU execution flips
individual seed outcomes, so these results must not be merged into the
A100 grid. Their purpose is to tell the author, before spending GPU
time, whether the high-momentum (P4) deviation survives a change of
initialization scale.

Design: 3 seeds x 3 momenta x 3 init scales x 2 decay variants = 54 runs.
beta = 0 is omitted (it is the algebraic null and the longest arm).
s = 1.5 is INCLUDED as a within-CPU bridge, so that the scale comparison
is internally consistent rather than cross-environment.
"""

import json, math, os, time, hashlib, sys
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

P, H = 17, 128
ETA, LAM = 0.2, 3e-4
ALPHA, SPLIT_SEED = 0.55, 12345
MAX_STEPS, EVAL_EVERY = 400_000, 25
TAU_MEM, TAU_GROK = 1.00, 0.90
EARLY_STOP_AFTER = 5_000
GAMMA, MIN_PTS = 0.20, 40

OUT = "/home/claude/cpu_scout.jsonl"


def derive_seed(*parts):
    key = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % (2**31 - 1)


def build_data():
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    c = (a + b) % P
    X = torch.zeros(P * P, 2 * P); X[torch.arange(P * P), a] = 1.0
    X[torch.arange(P * P), P + b] = 1.0
    Y = torch.zeros(P * P, P); Y[torch.arange(P * P), c] = 1.0
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(P * P, generator=g)
    n = int(ALPHA * P * P)
    return X[perm[:n]], Y[perm[:n]], X[perm[n:]], Y[perm[n:]]


XTR, YTR, XTE, YTE = build_data()
N1, N2 = 2 * P * H, H * P


def fit_rate(steps, vals):
    ok = np.isfinite(vals) & (vals > 0)
    s, v = steps[ok], vals[ok]
    if len(s) < MIN_PTS:
        return float("nan"), float("nan")
    y = np.log(v)
    A = np.vstack([s, np.ones_like(s)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(-coef[0]), (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def train(beta, decay, seed, scale):
    t0 = time.time()
    sd = derive_seed(seed, P, H, beta, decay, scale, "C2cpu")
    torch.manual_seed(sd)
    theta = torch.cat([torch.randn(N1) * scale / math.sqrt(2 * P),
                       torch.randn(N2) * scale / math.sqrt(H)]).requires_grad_(True)
    v = torch.zeros_like(theta)
    k_mem = k_grok = None
    stop_at = None
    hs, hn = [], []
    tgt_tr = YTR.argmax(1); tgt_te = YTE.argmax(1)
    for k in range(1, MAX_STEPS + 1):
        W1 = theta[:N1].view(2 * P, H); W2 = theta[N1:].view(H, P)
        loss = torch.mean(((XTR @ W1).pow(2) @ W2 - YTR) ** 2)
        gr = torch.autograd.grad(loss, theta)[0]
        with torch.no_grad():
            if decay == "coupled_l2":
                v.mul_(beta).add_(gr + LAM * theta)
                theta.add_(v, alpha=-ETA)
            else:
                v.mul_(beta).add_(gr)
                theta.add_(v, alpha=-ETA)
                theta.add_(theta, alpha=-ETA * LAM)
        if k % EVAL_EVERY == 0:
            with torch.no_grad():
                W1 = theta[:N1].view(2 * P, H); W2 = theta[N1:].view(H, P)
                tr = ((XTR @ W1).pow(2) @ W2).argmax(1).eq(tgt_tr).double().mean().item()
                te = ((XTE @ W1).pow(2) @ W2).argmax(1).eq(tgt_te).double().mean().item()
                nrm = theta.norm().item()
            hs.append(k); hn.append(nrm)
            if k_mem is None and tr >= TAU_MEM:
                k_mem = k
            if k_grok is None and te >= TAU_GROK:
                k_grok = k; stop_at = k + EARLY_STOP_AFTER
            if stop_at is not None and k >= stop_at:
                break
            if not math.isfinite(nrm):
                break
    st = np.array(hs, float); nn = np.array(hn, float)
    hi = k_grok if k_grok else (st[-1] if len(st) else 0)
    lo = max(k_mem or (st[0] if len(st) else 0), GAMMA * hi) if hi else 0
    w = (st >= lo) & (st <= hi)
    r, r2 = fit_rate(st[w], nn[w])
    return dict(beta=beta, decay=decay, seed=seed, init_scale=scale,
                k_mem=k_mem, k_grok=k_grok, grokked=k_grok is not None,
                delay_ratio=(k_grok / k_mem) if (k_mem and k_grok) else None,
                slow_rate=r, slow_rate_r2=r2, steps_run=k,
                wall_clock_s=round(time.time() - t0, 1))


SEEDS = [0, 1, 2]
BETAS = [0.5, 0.9, 0.95]
SCALES = [1.0, 1.5, 2.5]
DECAYS = ["coupled_l2", "decoupled_wd"]

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
            done.add((r["beta"], r["decay"], r["seed"], r["init_scale"]))
        except Exception:
            pass

cells = [(b, d, s, sc) for sc in SCALES for b in BETAS
         for d in DECAYS for s in SEEDS]
todo = [c for c in cells if c not in done]
print(f"{len(cells)} cells, {len(todo)} pending", flush=True)
t_start = time.time()
for i, (b, d, s, sc) in enumerate(todo, 1):
    rec = train(b, d, s, sc)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
    el = time.time() - t_start
    print(f"[{i}/{len(todo)}] s={sc} beta={b} {d:<13} seed={s} "
          f"k_grok={rec['k_grok']} ({rec['wall_clock_s']:.0f}s) "
          f"elapsed {el/60:.1f}m eta {el/i*(len(todo)-i)/60:.0f}m", flush=True)
print("DONE", flush=True)
