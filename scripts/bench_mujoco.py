"""Benchmark MuJoCo (CPU) env-steps/s on the G1, for honest comparison vs our CUDA sim.

Measures four points:
  - smooth-only (constraint+actuation disabled, Euler) = the SAME physics our sim does
  - full physics (contacts on, scene.xml floor) = what people actually run
each single-thread and across all cores (separate MjData per thread; mj_step drops the GIL).
"""
import os, time, threading
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCORE = os.cpu_count()
STEPS = 2000


def make(full):
    path = "models/scene.xml" if full else "models/g1.xml"
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, path))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    if not full:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    return m


def time_one(m, steps):
    d = mujoco.MjData(m)
    d.qpos[:] = m.key("stand").qpos if m.nkey else d.qpos
    for _ in range(50):
        mujoco.mj_step(m, d)  # warmup
    t0 = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return steps / (time.perf_counter() - t0)


def time_threaded(m, steps, nthreads):
    rates = [0.0] * nthreads
    def work(i):
        rates[i] = time_one(m, steps)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(nthreads)]
    t0 = time.perf_counter()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - t0
    return steps * nthreads / wall  # aggregate env-steps/s


print(f"MuJoCo {mujoco.__version__} CPU, {NCORE} logical cores, G1, fp64, Euler:")
for full in (False, True):
    label = "full-physics (contacts on)" if full else "smooth-only (our config)"
    m = make(full)
    one = time_one(m, STEPS)
    many = time_threaded(m, STEPS, NCORE)
    print(f"  {label:30s}: 1-core {one:10.0f}   {NCORE}-thread {many:11.0f} env-steps/s")
