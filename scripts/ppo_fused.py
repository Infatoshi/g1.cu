"""Fully-on-GPU PPO: the entire H-step rollout (physics + policy inference + action
sampling + reward) runs in ONE fused CUDA kernel (g1_rollout) with ZERO per-step host
round trips. Compare to scripts/ppo.py, which bounces GPU->host->GPU every control step
(32 PCIe syncs/iter) -- that orchestration cost, not the math, dominated the 67% "env" phase.

Here the trajectory is written straight into torch-owned GPU buffers; the host never touches
obs/action/reward during the rollout. torch only does the gradient update, reading those same
device pointers in place. No cuBLAS in the rollout: the MLP is inlined per-world in the kernel.

  uv run python scripts/ppo_fused.py [iters] [n_worlds]
  uv run python scripts/ppo_fused.py --smoke
"""
import ctypes
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from ppo import AC, RunNorm, G1Env, evaluate, _lib, _fp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# extra binding: the fused rollout entry point (all pointers are DEVICE addresses)
_lib.g1_rollout.restype = None
_lib.g1_rollout.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 10


def _dp(t):
    return ctypes.c_void_p(t.data_ptr())


def pack_weights(ac):
    """Flatten AC params into one contiguous cuda buffer in the order the kernel unpacks:
    W0,b0, W1,b1, Wmu,bmu, Wv,bv, log_std. nn.Linear weight is [out,in] row-major."""
    return torch.cat([
        ac.body[0].weight.reshape(-1), ac.body[0].bias,
        ac.body[2].weight.reshape(-1), ac.body[2].bias,
        ac.mu.weight.reshape(-1), ac.mu.bias,
        ac.v.weight.reshape(-1), ac.v.bias,
        ac.log_std,
    ]).contiguous()


def main():
    dev = "cuda"
    smoke = "--smoke" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    iters = int(args[0]) if args else (3 if smoke else 1500)
    N = int(args[1]) if len(args) > 1 else (256 if smoke else 8192)
    H = 16 if smoke else 32
    substeps = 10
    warm = "--warm" in sys.argv
    tag = "fused"
    for a in sys.argv:
        if a.startswith("--tag="):
            tag = a.split("=", 1)[1]
    ckpt_dir = os.path.join(HERE, "models", "ppo_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_name = f"{tag}_best.pt" if tag != "fused" else "best.pt"
    last_name = f"{tag}_last.pt" if tag != "fused" else "last.pt"
    log_path = os.path.join(ckpt_dir, f"reward_curve_{tag}.csv")
    logf = open(log_path, "w")
    logf.write("iter,mean_ep_rew_step,ret,loss,fwd_mean,ep_len,fall_rate,vx_mean,sps\n")

    env = G1Env(N, substeps=substeps, pgs_iters=8, seed=1)
    O_dim, A_dim = env.obs_dim, env.act_dim
    ac = AC(O_dim, A_dim).to(dev)
    norm = RunNorm(O_dim, dev)
    if warm:  # fine-tune from the existing walker (best.pt) -- transfers forward locomotion
        wp = os.path.join(ckpt_dir, "best.pt")
        ck = torch.load(wp, map_location=dev)
        ac.load_state_dict(ck["model"]); norm.mean = ck["obs_mean"].to(dev); norm.var = ck["obs_var"].to(dev)
        print(f"warm-started from {wp}")
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    gamma, lam, clip, epochs, mb = 0.99, 0.95, 0.2, 5, 4
    best_fwd = -1e9

    # persistent GPU trajectory buffers (written by the kernel, read by torch in place)
    O = torch.zeros(H, N, O_dim, device=dev)        # RAW obs
    A = torch.zeros(H, N, A_dim, device=dev)
    LP = torch.zeros(H, N, device=dev)
    V = torch.zeros(H, N, device=dev)
    R = torch.zeros(H, N, device=dev)
    D = torch.zeros(H, N, device=dev)
    Vlast = torch.zeros(N, device=dev)

    env.reset()  # init persistent device state (obs copy unused; rollout rebuilds obs on-GPU)
    print(f"FUSED env: N={N} obs={O_dim} act={A_dim} substeps={substeps} H={H} dev={dev}")
    t0 = time.time()
    first_cross = -1  # iter where a world first reaches the 5m finish line (eval, deterministic)
    first_rel = -1    # iter where the mean world crosses 5m (reliable)
    for it in range(iters):
        # freeze the normalizer stats the kernel will use; renormalize O with the SAME stats
        # afterwards so training obs is bit-consistent with what the behavior policy saw.
        fmean = norm.mean.clone()
        fvar = norm.var.clone()
        W = pack_weights(ac)
        _lib.g1_rollout(env.h, H, _dp(W), _dp(fmean), _dp(fvar),
                        _dp(O), _dp(A), _dp(LP), _dp(V), _dp(R), _dp(D), _dp(Vlast))

        with torch.no_grad():
            nobs = torch.clamp((O - fmean) / torch.sqrt(fvar + 1e-8), -10.0, 10.0)
            norm.update(O.reshape(-1, O_dim))  # advance running stats from raw obs

        # GAE on the GPU (V is critic value at each step, Vlast bootstraps s_H)
        adv = torch.zeros_like(R)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(H)):
            nextv = Vlast if t == H - 1 else V[t + 1]
            delta = R[t] + gamma * nextv * (1 - D[t]) - V[t]
            gae = delta + gamma * lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        b_o = nobs.reshape(-1, O_dim); b_a = A.reshape(-1, A_dim)
        b_lp = LP.reshape(-1); b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_o.shape[0]
        last_loss = 0.0
        for _ in range(epochs):
            idx = torch.randperm(n, device=dev)
            for s in range(0, n, n // mb):
                j = idx[s:s + n // mb]
                mu, std, val = ac(b_o[j])
                dist = torch.distributions.Normal(mu, std)
                lp = dist.log_prob(b_a[j]).sum(-1)
                ratio = (lp - b_lp[j]).exp()
                s1 = ratio * b_adv[j]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv[j]
                pl = -torch.min(s1, s2).mean()
                vl = (val - b_ret[j]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pl + 0.5 * vl - 0.01 * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
                last_loss = loss.item()

        sps = (it + 1) * H * N / (time.time() - t0)
        ep_r = R.mean().item()
        ev = {"fwd_mean": float("nan"), "ep_len_mean": float("nan"),
              "fall_rate": float("nan"), "vx_mean": float("nan")}
        if (it % 25 == 0) or (it == iters - 1):
            # eval reuses the host-step path (off the hot rollout); rollout owns the device
            # state, so restore it afterwards via reset on the next iter's frozen-stat rollout.
            ev = evaluate(env, ac, norm, dev, steps=400 if not smoke else 20)
            ckpt = dict(model=ac.state_dict(), obs_mean=norm.mean, obs_var=norm.var)
            if ev["fwd_mean"] > best_fwd:
                best_fwd = ev["fwd_mean"]
                torch.save(ckpt, os.path.join(ckpt_dir, best_name))
            torch.save(ckpt, os.path.join(ckpt_dir, last_name))
            el = time.time() - t0
            if first_cross < 0 and ev["fwd_max"] >= 4.8:
                first_cross = it
                print(f">>> FIRST FINISH CROSS (a world reached 5m) @ iter {it}  elapsed {el:.0f}s = {el/60:.1f} min")
            if first_rel < 0 and ev["fwd_mean"] >= 4.8:
                first_rel = it
                print(f">>> RELIABLE FINISH (mean world crosses 5m) @ iter {it}  elapsed {el:.0f}s = {el/60:.1f} min")
        print(f"iter {it:4d}  rew/step={ep_r:7.3f}  ret={b_ret.mean().item():7.3f}  "
              f"loss={last_loss:7.3f}  fwd={ev['fwd_mean']:6.2f}m max={ev.get('fwd_max',float('nan')):5.2f} "
              f"ep_len={ev['ep_len_mean']:6.1f} fall={ev['fall_rate']:.2f} vx={ev['vx_mean']:5.2f}  sps={sps:.2e}")
        logf.write(f"{it},{ep_r:.4f},{b_ret.mean().item():.4f},{last_loss:.4f},"
                   f"{ev['fwd_mean']:.4f},{ev['ep_len_mean']:.2f},{ev['fall_rate']:.4f},"
                   f"{ev['vx_mean']:.4f},{sps:.3e}\n")
        logf.flush()
    env.close()
    logf.close()
    print(f"reward curve -> {log_path}; checkpoints -> {ckpt_dir} (best fwd={best_fwd:.2f}m)")
    if smoke:
        print("SMOKE OK: fused on-GPU rollout + PPO update ran end-to-end.")


if __name__ == "__main__":
    main()
