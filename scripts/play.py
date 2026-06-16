"""Faithful playback of the trained G1 walking policy.

Robust approach (avoids any obs/action-mapping mismatch): drive the TRAINED policy
through our OWN validated CUDA env, record world-0's qpos each control step, then replay
that trajectory KINEMATICALLY in mujoco.viewer (set data.qpos per frame). What you see is
exactly what the policy produces in our sim -- obs comes straight from the env step(), we
never reconstruct it; the only new data path is qpos out (g1_env_qpos).

Usage:
  uv run python scripts/play.py            # record + watch live (needs the local display)
  uv run python scripts/play.py --no-view  # record + print sanity stats only (headless)
"""
import ctypes
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

# Import the EXACT policy net + normalizer + env wrapper used in training so the arch and
# obs handling are guaranteed identical (no drift between trainer and player).
from ppo import AC, RunNorm, G1Env, _lib, _fp  # noqa: E402

# Add the additive qpos getter binding to the already-loaded ctypes lib.
_lib.g1_env_qpos.argtypes = [ctypes.c_void_p, _fp]
_lib.g1_env_qpos.restype = None

NQ = 36  # G1_NQ
CTRL_DT = 0.02  # control dt = substeps(10) * G1_DT(0.002)
SUBSTEPS = 10
PGS_ITERS = 8  # matches ppo.py G1Env(..., pgs_iters=8, ...)
MAX_STEPS = 300


def qpos0(env):
    out = np.zeros(NQ, np.float32)
    _lib.g1_env_qpos(env.h, out.ctypes.data_as(_fp))
    return out.copy()


def rollout():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(HERE, "models", "ppo_ckpt", "finish_best.pt")
    ckpt = torch.load(ckpt_path, map_location=dev)

    env = G1Env(64, substeps=SUBSTEPS, pgs_iters=PGS_ITERS, seed=0)
    ac = AC(env.obs_dim, env.act_dim).to(dev)
    ac.load_state_dict(ckpt["model"])
    ac.eval()
    norm = RunNorm(env.obs_dim, dev)
    norm.mean = ckpt["obs_mean"].to(dev)
    norm.var = ckpt["obs_var"].to(dev)

    obs = torch.tensor(env.reset(), device=dev)
    traj = [qpos0(env)]  # initial pose
    early_done = False
    steps = 0
    with torch.no_grad():
        for t in range(MAX_STEPS):
            mu, _, _ = ac(norm.norm(obs))  # deterministic mean action
            no, rw, dn = env.step(mu.cpu().numpy())
            traj.append(qpos0(env))
            steps += 1
            obs = torch.tensor(no, device=dev)
            if dn[0] > 0.5:
                early_done = True
                break
    env.close()
    traj = np.asarray(traj, np.float32)  # [T, NQ]

    out_path = os.path.join(HERE, "bench", "walk_traj.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, traj)

    x = traj[:, 0]
    z = traj[:, 2]
    print(f"recorded {traj.shape[0]} frames ({steps} control steps) -> {out_path}")
    print(f"base-x: {x[0]:+.3f} -> {x[-1]:+.3f}  (displacement {x[-1]-x[0]:+.3f} m)")
    print(f"base-z: min {z.min():.3f}  max {z.max():.3f}  (standing ~0.78)")
    print(f"episode length: {steps} control steps ({steps*CTRL_DT:.2f} s)")
    print(f"early done flag: {early_done}")
    avg_vx = (x[-1] - x[0]) / max(steps * CTRL_DT, 1e-6)
    print(f"avg forward speed: {avg_vx:+.3f} m/s")
    return traj


def view(traj):
    import mujoco
    import mujoco.viewer

    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "models", "scene.xml"))
    d = mujoco.MjData(m)
    assert m.nq == traj.shape[1], f"nq mismatch: model {m.nq} vs traj {traj.shape[1]}"

    print("launching viewer (loops the recorded gait; close window to quit)...")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            for t in range(traj.shape[0]):
                if not viewer.is_running():
                    break
                tic = time.time()
                d.qpos[:] = traj[t]
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                viewer.sync()
                dt = CTRL_DT - (time.time() - tic)
                if dt > 0:
                    time.sleep(dt)
            time.sleep(0.5)  # brief pause before looping


def main():
    no_view = "--no-view" in sys.argv
    traj = rollout()
    if no_view:
        return
    view(traj)


if __name__ == "__main__":
    main()
