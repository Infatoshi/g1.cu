"""Benchmark MJX (MuJoCo-on-GPU/JAX) env-steps/s on the G1 -- the fair GPU-vs-GPU
comparison against our CUDA sim. Uses the MJX-tuned model (primitive colliders).

  smooth-only: CONSTRAINT+ACTUATION disabled, Euler  -> matches our CUDA sim exactly
  full-physics: contacts + Newton solver on           -> what people actually run

Batches N envs via vmap, rolls out NSTEPS via lax.scan, jit-compiled. Reports
N*NSTEPS/elapsed (compile excluded; timed with block_until_ready).
"""
import os, time, functools
import numpy as np
import jax, jax.numpy as jnp
import mujoco
from mujoco import mjx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSTEPS = 200
print("jax", jax.__version__, jax.devices())


def build(full):
    path = "models/scene_mjx.xml" if full else "models/g1_mjx.xml"
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, path))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    if not full:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    return m


def bench(m, N):
    mx = mjx.put_model(m)
    d0 = mjx.make_data(mx)
    batch = jax.vmap(lambda _: d0)(jnp.arange(N))

    @jax.jit
    def rollout(dx):
        step = lambda d, _: (jax.vmap(mjx.step, in_axes=(None, 0))(mx, d), None)
        d, _ = jax.lax.scan(step, dx, None, length=NSTEPS)
        return d

    out = rollout(batch)            # compile + warm
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    out = rollout(batch)
    jax.block_until_ready(out)
    dt = time.perf_counter() - t0
    return N * NSTEPS / dt


for full in (False, True):
    label = "full-physics (contacts on)" if full else "smooth-only (our config)"
    m = build(full)
    print(f"\n{label}:")
    for N in (1024, 4096, 16384, 65536):
        try:
            r = bench(m, N)
            print(f"  N={N:7d}  {r:.3e} env-steps/s")
        except Exception as e:
            print(f"  N={N:7d}  FAILED: {str(e)[:80]}")
            break
