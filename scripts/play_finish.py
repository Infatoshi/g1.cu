"""Roll the trained finish-line policy through our CUDA env, pick the world that walks
farthest (a clean winner that crosses x=X_FIN without falling), record its qpos trajectory
to bench/finish_traj.npy for rendering. Reports how many of the N worlds reached the line.

  uv run python scripts/play_finish.py
"""
import ctypes
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from ppo import AC, RunNorm, G1Env, _lib, _fp  # noqa: E402

_lib.g1_env_qpos_all.argtypes = [ctypes.c_void_p, _fp]
_lib.g1_env_qpos_all.restype = None

NQ = 36
CTRL_DT = 0.02
SUB = 10
PGS = 8
X_FIN = 5.0
N = 256
MAX_STEPS = 480   # 5m / 0.8 m/s ~ 312 control steps; 480 leaves margin


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(HERE, "models", "ppo_ckpt", "finish_best.pt"), map_location=dev)
    env = G1Env(N, substeps=SUB, pgs_iters=PGS, seed=0)
    ac = AC(env.obs_dim, env.act_dim).to(dev)
    ac.load_state_dict(ckpt["model"]); ac.eval()
    norm = RunNorm(env.obs_dim, dev)
    norm.mean = ckpt["obs_mean"].to(dev); norm.var = ckpt["obs_var"].to(dev)

    obs = torch.tensor(env.reset(), device=dev)
    buf = np.zeros((N, NQ), np.float32)
    traj = np.zeros((MAX_STEPS + 1, N, NQ), np.float32)
    _lib.g1_env_qpos_all(env.h, buf.ctypes.data_as(_fp)); traj[0] = buf
    done_step = np.full(N, -1, int)
    with torch.no_grad():
        for t in range(MAX_STEPS):
            mu, _, _ = ac(norm.norm(obs))
            no, rw, dn = env.step(mu.cpu().numpy())
            _lib.g1_env_qpos_all(env.h, buf.ctypes.data_as(_fp)); traj[t + 1] = buf
            newly = (dn > 0.5) & (done_step < 0)
            done_step[newly] = t + 1
            obs = torch.tensor(no, device=dev)
    env.close()

    # per world: usable horizon = up to first done (the done frame is the post-win reset, drop it)
    maxx = np.full(N, -1e9)
    for w in range(N):
        end = done_step[w] if done_step[w] > 0 else MAX_STEPS + 1
        maxx[w] = traj[:end, w, 0].max()
    won = (done_step > 0) & (maxx >= X_FIN - 0.1)   # reached the line just before the reset
    best_w = int(np.argmax(maxx))
    end = (done_step[best_w] - 1) if done_step[best_w] > 0 else (MAX_STEPS + 1)
    wtraj = traj[:end, best_w, :]
    out = os.path.join(HERE, "bench", "finish_traj.npy")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, wtraj)

    x = wtraj[:, 0]
    print(f"finish line X_FIN={X_FIN} m, N={N} worlds, deterministic policy, <= {MAX_STEPS} steps")
    print(f"reached the line: {int(won.sum())}/{N} worlds ({100*won.mean():.0f}%)")
    print(f"picked world {best_w}: {wtraj.shape[0]} frames, x {x[0]:+.2f} -> {x[-1]:+.2f} m, "
          f"reached {maxx[best_w]:.2f} m in {wtraj.shape[0]*CTRL_DT:.1f} s "
          f"({'WIN' if won[best_w] else 'did not cross'})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
