"""Pure on-device physics throughput of our RL env (env_advance, zero action) -- the
apples-to-apples unit vs MJX's mjx.step rollout (substeps = physics steps). No policy, no
host I/O during timing. Reports substeps/s across world counts.

  uv run python scripts/bench_env_phys.py
"""
import ctypes, os, time
from ppo import _lib, G1Env  # reuse the ctypes binding

_lib.g1_env_bench_phys.restype = None
_lib.g1_env_bench_phys.argtypes = [ctypes.c_void_p, ctypes.c_int]

SUB = 10
print(f"env physics throughput (substeps = physics steps, S={SUB}/control step):")
for N in (8192, 16384, 32768, 65536):
    try:
        env = G1Env(N, substeps=SUB, pgs_iters=8, seed=1)
        env.reset()
        _lib.g1_env_bench_phys(env.h, 5)            # warm
        t0 = time.perf_counter()
        K = 50
        _lib.g1_env_bench_phys(env.h, K)            # blocking (syncs internally)
        dt = time.perf_counter() - t0
        sps = N * K * SUB / dt
        print(f"  N={N:7d}  {sps:.3e} substeps/s   ({K*SUB} phys steps in {dt*1e3:.0f} ms)")
        env.close()
    except Exception as e:
        print(f"  N={N:7d}  FAILED: {str(e)[:80]}")
        break
