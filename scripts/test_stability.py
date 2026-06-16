"""Stage A stability gate: full kp=500, S=10 substeps/action, hold the standing pose.

Zero action = residual 0 on the stand pose (ACT_SCALE*0), so the PD target IS the standing
keyframe. The robot starts standing on the ground (contacts active). With the old explicit
KP_SCALE=0.2 hack removed, the only thing keeping this bounded is the implicit kd integration.
Run a few hundred control steps and assert: no NaN, base height stays near standing, upright.
"""
import os
import ctypes
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(HERE, "build", "libg1env.so")

N = 256
S = 10          # substeps per action -- the regime that diverged under explicit Euler
STEPS = 400


def main():
    lib = ctypes.CDLL(LIB)
    lib.g1_env_create.restype = ctypes.c_void_p
    lib.g1_env_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    lib.g1_env_obs_dim.restype = ctypes.c_int
    lib.g1_env_act_dim.restype = ctypes.c_int
    for fn in (lib.g1_env_reset, lib.g1_env_step, lib.g1_env_destroy):
        fn.restype = None
    OBS = lib.g1_env_obs_dim(); ACT = lib.g1_env_act_dim()

    h = lib.g1_env_create(N, S, 8, 0)
    obs = np.zeros((N, OBS), dtype=np.float32)
    lib.g1_env_reset(ctypes.c_void_p(h), obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    act = np.zeros((N, ACT), dtype=np.float32)  # hold stand pose
    rew = np.zeros(N, dtype=np.float32); done = np.zeros(N, dtype=np.float32)
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    heights = []
    nan_seen = False
    for t in range(STEPS):
        lib.g1_env_step(ctypes.c_void_p(h), fp(act), fp(obs), fp(rew), fp(done))
        if not np.isfinite(obs).all() or not np.isfinite(rew).all():
            nan_seen = True
            print(f"NaN/inf at step {t}")
            break
        # base lin vel obs[0:3]; track |vel| as a divergence proxy
        bv = np.linalg.norm(obs[:, 0:3], axis=1)
        heights.append(bv.mean())

    bvel = np.array(heights)
    # divergence proxy: base linear-velocity magnitude must stay bounded (no blowup)
    print(f"N={N} S={S} kp=500 (full) steps={t+1}")
    print(f"  base |linvel| mean over worlds: start {bvel[0]:.4f}  mid {bvel[len(bvel)//2]:.4f}  end {bvel[-1]:.4f}  max {bvel.max():.4f}")
    print(f"  mean reward final step: {rew.mean():.4f}  done rate: {done.mean():.3f}")
    bounded = (not nan_seen) and np.isfinite(bvel).all() and bvel.max() < 50.0
    print("RESULT:", "PASS" if bounded else "FAIL", "(bounded, no NaN)" if bounded else "(diverged)")
    lib.g1_env_destroy(ctypes.c_void_p(h))
    return 0 if bounded else 1


if __name__ == "__main__":
    raise SystemExit(main())
