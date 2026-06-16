"""Minimal PPO (clip + GAE) driving the batched GPU G1 walking env (build/libg1env.so).

Env: ctypes wrapper around src/g1_env.cu. All N worlds stay resident on the GPU;
only obs/reward/done cross to host each policy step. Position (PD) actuators: the
policy outputs joint targets, the env converts to torque tau=kp*(target-q)-kd*qd.

Goal of this scaffold: a training loop that RUNS and shows reward moving. Not a
finished walker. Run:  uv run python scripts/ppo.py [iters] [n_worlds]
Smoke test:            uv run python scripts/ppo.py --smoke
"""
import ctypes
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(HERE, "build", "libg1env.so")

# ---------------- ctypes env binding ----------------
_lib = ctypes.CDLL(LIB)
_lib.g1_env_create.restype = ctypes.c_void_p
_lib.g1_env_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
_lib.g1_env_obs_dim.restype = ctypes.c_int
_lib.g1_env_act_dim.restype = ctypes.c_int
_fp = ctypes.POINTER(ctypes.c_float)
_lib.g1_env_reset.argtypes = [ctypes.c_void_p, _fp]
_lib.g1_env_step.argtypes = [ctypes.c_void_p, _fp, _fp, _fp, _fp]
_lib.g1_env_destroy.argtypes = [ctypes.c_void_p]


def _p(a):
    return a.ctypes.data_as(_fp)


class G1Env:
    def __init__(self, n_worlds, substeps=10, pgs_iters=2000, seed=0):
        self.N = n_worlds
        self.obs_dim = _lib.g1_env_obs_dim()
        self.act_dim = _lib.g1_env_act_dim()
        self.h = _lib.g1_env_create(n_worlds, substeps, pgs_iters, seed)
        self._obs = np.zeros((n_worlds, self.obs_dim), np.float32)
        self._rew = np.zeros(n_worlds, np.float32)
        self._done = np.zeros(n_worlds, np.float32)

    def reset(self):
        _lib.g1_env_reset(self.h, _p(self._obs))
        return self._obs.copy()

    def step(self, act):
        act = np.ascontiguousarray(act, np.float32)
        _lib.g1_env_step(self.h, _p(act), _p(self._obs), _p(self._rew), _p(self._done))
        return self._obs.copy(), self._rew.copy(), self._done.copy()

    def close(self):
        _lib.g1_env_destroy(self.h)


# ---------------- running obs normalizer ----------------
class RunNorm:
    """Welford running mean/var for observation normalization (clipped). Standard for PPO
    locomotion -- raw joint pos/vel + velocities span very different scales."""
    def __init__(self, dim, dev):
        self.mean = torch.zeros(dim, device=dev)
        self.var = torch.ones(dim, device=dev)
        self.count = 1e-4

    def update(self, x):  # x: [B, dim]
        b = x.shape[0]
        bm = x.mean(0); bv = x.var(0, unbiased=False)
        tot = self.count + b
        delta = bm - self.mean
        self.mean += delta * b / tot
        self.var = (self.var * self.count + bv * b + delta**2 * self.count * b / tot) / tot
        self.count = tot

    def norm(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)


# ---------------- actor-critic ----------------
class AC(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(d, act_dim)
        self.v = nn.Linear(d, 1)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, x):
        z = self.body(x)
        return self.mu(z), self.log_std.exp(), self.v(z).squeeze(-1)


@torch.no_grad()
def evaluate(env, ac, norm, dev, steps=200):
    """Roll the deterministic (mean) policy and quantify locomotion behavior.
    Tracks per-world forward distance (integrating base lin-vel obs[0]*control_dt), episode
    length until fall, fall rate, base-height proxy (not in obs; use upright as proxy)."""
    obs = torch.tensor(env.reset(), device=dev)
    N = env.N
    dt_ctrl = 0.02
    fwd = np.zeros(N); alive_len = np.zeros(N); done_any = np.zeros(N, bool)
    vx_acc = np.zeros(N)
    for _ in range(steps):
        mu, _, _ = ac(norm.norm(obs))
        no, rw, dn = env.step(mu.cpu().numpy())
        vx = no[:, 0]  # base forward lin vel (world x)
        live = ~done_any
        fwd[live] += vx[live] * dt_ctrl
        vx_acc[live] += vx[live]
        alive_len[live] += 1
        done_any |= dn.astype(bool)
        obs = torch.tensor(no, device=dev)
    fall_rate = done_any.mean()
    return dict(fwd_mean=float(fwd.mean()), fwd_max=float(fwd.max()),
                ep_len_mean=float(alive_len.mean()), fall_rate=float(fall_rate),
                vx_mean=float((vx_acc / np.maximum(alive_len, 1)).mean()))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smoke = "--smoke" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    iters = int(args[0]) if args else (3 if smoke else 100)
    N = int(args[1]) if len(args) > 1 else (256 if smoke else 4096)
    horizon = 16 if smoke else 32
    substeps = 10  # control dt = 10 * 0.002 = 0.02 s
    ckpt_dir = os.path.join(HERE, "models", "ppo_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "reward_curve.csv")
    logf = open(log_path, "w")
    logf.write("iter,mean_ep_rew_step,ret,loss,fwd_mean,ep_len,fall_rate,vx_mean,env_sps\n")

    env = G1Env(N, substeps=substeps, pgs_iters=8, seed=1)
    ac = AC(env.obs_dim, env.act_dim).to(dev)
    norm = RunNorm(env.obs_dim, dev)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    gamma, lam, clip, epochs, mb = 0.99, 0.95, 0.2, 5, 4
    best_fwd = -1e9

    obs = torch.tensor(env.reset(), device=dev)
    print(f"env: N={N} obs={env.obs_dim} act={env.act_dim} substeps={substeps} dev={dev}")
    t0 = time.time()
    for it in range(iters):
        O, A, LP, R, D, V = [], [], [], [], [], []
        ep_r = 0.0
        for _ in range(horizon):
            norm.update(obs)
            nobs = norm.norm(obs)
            with torch.no_grad():
                mu, std, val = ac(nobs)
                dist = torch.distributions.Normal(mu, std)
                act = dist.sample()
                lp = dist.log_prob(act).sum(-1)
            no, rw, dn = env.step(act.cpu().numpy())
            O.append(nobs); A.append(act); LP.append(lp); V.append(val)
            R.append(torch.tensor(rw, device=dev)); D.append(torch.tensor(dn, device=dev))
            obs = torch.tensor(no, device=dev)
            ep_r += rw.mean()
        with torch.no_grad():
            _, _, last_v = ac(norm.norm(obs))
        # GAE
        O = torch.stack(O); A = torch.stack(A); LP = torch.stack(LP)
        R = torch.stack(R); D = torch.stack(D); V = torch.stack(V)
        adv = torch.zeros_like(R); gae = torch.zeros(N, device=dev)
        for t in reversed(range(horizon)):
            nextv = last_v if t == horizon - 1 else V[t + 1]
            delta = R[t] + gamma * nextv * (1 - D[t]) - V[t]
            gae = delta + gamma * lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        b_o = O.reshape(-1, env.obs_dim); b_a = A.reshape(-1, env.act_dim)
        b_lp = LP.reshape(-1); b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = b_o.shape[0]
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
        sps = (it + 1) * horizon * N / (time.time() - t0)
        # periodic quantitative eval (deterministic policy)
        ev = {"fwd_mean": float("nan"), "ep_len_mean": float("nan"),
              "fall_rate": float("nan"), "vx_mean": float("nan")}
        if (it % 10 == 0) or (it == iters - 1):
            ev = evaluate(env, ac, norm, dev, steps=150 if not smoke else 20)
            obs = torch.tensor(env.reset(), device=dev)  # reset after eval rollout
            ckpt = dict(model=ac.state_dict(), obs_mean=norm.mean, obs_var=norm.var)
            if ev["fwd_mean"] > best_fwd:
                best_fwd = ev["fwd_mean"]
                torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
            torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))
        print(f"iter {it:3d}  rew/step={ep_r/horizon:7.3f}  ret={b_ret.mean().item():7.3f}  "
              f"loss={loss.item():7.3f}  fwd={ev['fwd_mean']:6.2f}m ep_len={ev['ep_len_mean']:6.1f} "
              f"fall={ev['fall_rate']:.2f} vx={ev['vx_mean']:5.2f}  sps={sps:.2e}")
        logf.write(f"{it},{ep_r/horizon:.4f},{b_ret.mean().item():.4f},{loss.item():.4f},"
                   f"{ev['fwd_mean']:.4f},{ev['ep_len_mean']:.2f},{ev['fall_rate']:.4f},"
                   f"{ev['vx_mean']:.4f},{sps:.3e}\n")
        logf.flush()
    env.close()
    logf.close()
    print(f"reward curve -> {log_path}; checkpoints -> {ckpt_dir} (best fwd={best_fwd:.2f}m)")
    if smoke:
        print("SMOKE OK: rollout + PPO update ran end-to-end.")


if __name__ == "__main__":
    main()
