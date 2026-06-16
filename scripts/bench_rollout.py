"""Head-to-head: rollout wall-clock, fused on-GPU kernel vs the host-round-trip loop.
Identical config (N, H, substeps), same policy net, no eval/no update -- pure rollout cost.

  uv run python scripts/bench_rollout.py [N] [H] [reps]
"""
import ctypes
import sys
import time

import numpy as np
import torch

from ppo import AC, RunNorm, G1Env, _lib, _fp
from ppo_fused import pack_weights, _dp

dev = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
H = int(sys.argv[2]) if len(sys.argv) > 2 else 32
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
SUB = 10


def sync():
    torch.cuda.synchronize()


def main():
    env = G1Env(N, substeps=SUB, pgs_iters=8, seed=1)
    od, ad = env.obs_dim, env.act_dim
    ac = AC(od, ad).to(dev)
    norm = RunNorm(od, dev)

    # ---- host-round-trip rollout (the ppo.py path) ----
    obs = torch.tensor(env.reset(), device=dev)
    for _ in range(2):  # warmup
        for _ in range(H):
            with torch.no_grad():
                mu, std, val = ac(norm.norm(obs))
                act = torch.distributions.Normal(mu, std).sample()
            no, rw, dn = env.step(act.cpu().numpy())
            obs = torch.tensor(no, device=dev)
    sync(); t = time.time()
    for _ in range(REPS):
        for _ in range(H):
            norm.update(obs); nobs = norm.norm(obs)
            with torch.no_grad():
                mu, std, val = ac(nobs)
                dist = torch.distributions.Normal(mu, std)
                act = dist.sample(); lp = dist.log_prob(act).sum(-1)
            no, rw, dn = env.step(act.cpu().numpy())
            obs = torch.tensor(no, device=dev)
    sync(); host_t = (time.time() - t) / REPS

    # ---- fused on-GPU rollout ----
    O = torch.zeros(H, N, od, device=dev); A = torch.zeros(H, N, ad, device=dev)
    LP = torch.zeros(H, N, device=dev); V = torch.zeros(H, N, device=dev)
    R = torch.zeros(H, N, device=dev); Dn = torch.zeros(H, N, device=dev)
    Vlast = torch.zeros(N, device=dev)
    env.reset()
    W = pack_weights(ac); fmean = norm.mean.clone(); fvar = norm.var.clone()
    for _ in range(2):  # warmup
        _lib.g1_rollout(env.h, H, _dp(W), _dp(fmean), _dp(fvar),
                        _dp(O), _dp(A), _dp(LP), _dp(V), _dp(R), _dp(Dn), _dp(Vlast))
    sync(); t = time.time()
    for _ in range(REPS):
        _lib.g1_rollout(env.h, H, _dp(W), _dp(fmean), _dp(fvar),
                        _dp(O), _dp(A), _dp(LP), _dp(V), _dp(R), _dp(Dn), _dp(Vlast))
    sync(); fused_t = (time.time() - t) / REPS
    env.close()

    sub = H * N * SUB
    print(f"\n=== rollout wall-clock (N={N}, H={H}, substeps={SUB}; mean of {REPS}) ===")
    print(f"host round-trip : {host_t*1e3:8.1f} ms/rollout   {sub/host_t:.3e} substeps/s")
    print(f"fused on-GPU    : {fused_t*1e3:8.1f} ms/rollout   {sub/fused_t:.3e} substeps/s")
    print(f"speedup         : {host_t/fused_t:6.2f}x")


if __name__ == "__main__":
    main()
