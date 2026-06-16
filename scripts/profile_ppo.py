"""Measure where PPO training wall-clock actually goes, at the run-2 config.

Replicates ppo.py's loop phase-by-phase with CUDA synchronization so each segment's
time is attributed correctly. Reports mean ms/iter and % per phase. Eval+ckpt happens
1-in-10 iters in the real run, so it's measured separately and amortized /10.

  uv run python scripts/profile_ppo.py [iters] [N]
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from ppo import AC, RunNorm, G1Env, evaluate

dev = "cuda"
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
HORIZON = 32
SUB = 10


def sync():
    torch.cuda.synchronize()


def main():
    env = G1Env(N, substeps=SUB, pgs_iters=8, seed=1)
    ac = AC(env.obs_dim, env.act_dim).to(dev)
    norm = RunNorm(env.obs_dim, dev)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    gamma, lam, clip, epochs, mb = 0.99, 0.95, 0.2, 5, 4

    obs = torch.tensor(env.reset(), device=dev)
    acc = {"policy": 0.0, "env": 0.0, "gae": 0.0, "update": 0.0}
    eval_t = 0.0
    # warmup
    for _ in range(3):
        a = torch.zeros(N, env.act_dim)
        env.step(a.numpy())

    t_start = time.time()
    for it in range(ITERS):
        O, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(HORIZON):
            sync(); t = time.time()
            norm.update(obs); nobs = norm.norm(obs)
            with torch.no_grad():
                mu, std, val = ac(nobs)
                dist = torch.distributions.Normal(mu, std)
                act = dist.sample(); lp = dist.log_prob(act).sum(-1)
            sync(); acc["policy"] += time.time() - t

            t = time.time()
            no, rw, dn = env.step(act.cpu().numpy())
            obs2 = torch.tensor(no, device=dev)
            sync(); acc["env"] += time.time() - t

            O.append(nobs); A.append(act); LP.append(lp); V.append(val)
            R.append(torch.tensor(rw, device=dev)); D.append(torch.tensor(dn, device=dev))
            obs = obs2

        sync(); t = time.time()
        with torch.no_grad():
            _, _, last_v = ac(norm.norm(obs))
        O = torch.stack(O); A = torch.stack(A); LP = torch.stack(LP)
        R = torch.stack(R); D = torch.stack(D); V = torch.stack(V)
        adv = torch.zeros_like(R); gae = torch.zeros(N, device=dev)
        for tt in reversed(range(HORIZON)):
            nextv = last_v if tt == HORIZON - 1 else V[tt + 1]
            delta = R[tt] + gamma * nextv * (1 - D[tt]) - V[tt]
            gae = delta + gamma * lam * (1 - D[tt]) * gae
            adv[tt] = gae
        ret = adv + V
        b_o = O.reshape(-1, env.obs_dim); b_a = A.reshape(-1, env.act_dim)
        b_lp = LP.reshape(-1); b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        sync(); acc["gae"] += time.time() - t

        sync(); t = time.time()
        n = b_o.shape[0]
        for _ in range(epochs):
            idx = torch.randperm(n, device=dev)
            for s in range(0, n, n // mb):
                j = idx[s:s + n // mb]
                mu, std, val = ac(b_o[j])
                dist = torch.distributions.Normal(mu, std)
                lp = dist.log_prob(b_a[j]).sum(-1)
                ratio = (lp - b_lp[j]).exp()
                s1 = ratio * b_adv[j]; s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv[j]
                pl = -torch.min(s1, s2).mean(); vl = (val - b_ret[j]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pl + 0.5 * vl - 0.01 * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
        sync(); acc["update"] += time.time() - t

    # eval+ckpt cost (happens 1-in-10 iters): measure one and amortize
    sync(); t = time.time()
    evaluate(env, ac, norm, dev, steps=150)
    torch.save(dict(model=ac.state_dict(), obs_mean=norm.mean, obs_var=norm.var),
               os.path.join(HERE_CK, "prof_tmp.pt"))
    sync(); eval_once = time.time() - t
    obs = torch.tensor(env.reset(), device=dev)
    env.close()

    total_loop = time.time() - t_start
    eval_amort = eval_once / 10.0  # 1 eval per 10 iters
    per_iter = {k: v / ITERS for k, v in acc.items()}
    per_iter["eval+ckpt"] = eval_amort
    grand = sum(per_iter.values())
    print(f"\n=== PPO time breakdown (N={N}, horizon={HORIZON}, substeps={SUB}, pgs=8; {ITERS} iters) ===")
    print(f"{'phase':<14}{'ms/iter':>10}{'%':>8}")
    for k in ["env", "policy", "gae", "update", "eval+ckpt"]:
        print(f"{k:<14}{per_iter[k]*1e3:>10.1f}{100*per_iter[k]/grand:>8.1f}")
    print(f"{'TOTAL':<14}{grand*1e3:>10.1f}{100.0:>8.1f}")
    print(f"\nmeasured loop {total_loop/ITERS*1e3:.1f} ms/iter (excl one-time eval); "
          f"eval once = {eval_once*1e3:.0f} ms")
    # sim-only ceiling: env phase includes host copies + torch.tensor; raw kernel rate is 2.78e7/s
    sub_steps_per_iter = HORIZON * N * SUB
    sim_ceiling_ms = sub_steps_per_iter / 2.78e7 * 1e3
    print(f"env phase {per_iter['env']*1e3:.1f} ms/iter; pure-kernel ceiling for the same "
          f"{sub_steps_per_iter:.2e} substeps @2.78e7/s = {sim_ceiling_ms:.1f} ms")


HERE_CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ppo_ckpt")
if __name__ == "__main__":
    main()
