"""Benchmark MuJoCo Warp (NVIDIA Warp GPU engine, MJX's faster successor) on the G1.

Same models/config as bench_mjx.py for a fair GPU-vs-GPU-vs-ours comparison:
  smooth-only: CONSTRAINT+ACTUATION disabled, Euler   (matches our CUDA sim)
  full-physics: contacts + solver on
"""
import os, time
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mw

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSTEPS = 200


def build(full):
    path = "models/scene_mjx.xml" if full else "models/g1_mjx.xml"
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, path))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    if not full:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    return m


def bench(m, N):
    d = mujoco.MjData(m)   # default pose (qpos0) -- matches the MJX benchmark init
    mujoco.mj_forward(m, d)
    wm = mw.put_model(m)
    wd = mw.put_data(m, d, nworld=N)
    for _ in range(5):
        mw.step(wm, wd)   # warmup / graph capture
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        mw.step(wm, wd)
    wp.synchronize()
    dt = time.perf_counter() - t0
    return N * NSTEPS / dt


print("warp", wp.config.version, "| mujoco_warp", mw.__version__ if hasattr(mw, "__version__") else "?")
for full in (False, True):
    label = "full-physics (contacts on)" if full else "smooth-only (our config)"
    m = build(full)
    print(f"\n{label}:")
    for N in (1024, 4096, 16384, 65536):
        try:
            r = bench(m, N)
            print(f"  N={N:7d}  {r:.3e} env-steps/s")
        except Exception as e:
            print(f"  N={N:7d}  FAILED: {str(e)[:120]}")
            break
