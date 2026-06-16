"""Contact benchmark oracle: MuJoCo full-physics G1 drop-and-settle on a floor.

Two purposes:
  1. Golden trajectory (qpos over time) = validation target for a future specialized
     CUDA contact solver (the M5 hard part).
  2. Contact-count DIVERGENCE PROFILE = the number that decides the moat. A SIMT
     "lane = world" solver is only efficient if worlds mostly agree on contact count.
     We report ncon over time (one trajectory) and the cross-world ncon distribution
     (many perturbed states) -> the bucket histogram (ballistic / 1-2 / 3-4 / 5+ / tail)
     from the CLAUDE.md contact-bucketing design.
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "bench", "contact_ref.npz")
N_STEPS = 600
DROP_H = 0.15          # lift base so feet impact the floor
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "models", "scene.xml"))
    # realistic config: contacts + Newton solver ON, actuation OFF (passive ragdoll drop
    # sweeps ballistic -> impact -> settling -> rest = full divergence spectrum).
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    print(f"scene: nq={m.nq} nv={m.nv} solver={mujoco.mjtSolver(m.opt.solver).name} "
          f"cone={mujoco.mjtCone(m.opt.cone).name} dt={m.opt.timestep}")

    d = mujoco.MjData(m)
    d.qpos[:] = m.key("stand").qpos
    d.qpos[2] += DROP_H
    mujoco.mj_forward(m, d)

    qpos = np.zeros((N_STEPS + 1, m.nq))
    ncon = np.zeros(N_STEPS + 1, dtype=int)
    qpos[0] = d.qpos; ncon[0] = d.ncon
    for t in range(1, N_STEPS + 1):
        mujoco.mj_step(m, d)
        qpos[t] = d.qpos; ncon[t] = d.ncon

    # cross-world contact distribution: sample states along the trajectory + small
    # per-joint perturbations (proxy for N parallel worlds at mixed phases).
    samples = []
    for t in rng.integers(50, N_STEPS, size=4000):
        d.qpos[:] = qpos[t]
        d.qpos[7:] += rng.normal(0, 0.05, size=m.nq - 7)
        d.qvel[:] = rng.normal(0, 0.2, size=m.nv)
        mujoco.mj_forward(m, d)
        samples.append(d.ncon)
    samples = np.array(samples)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, qpos=qpos, ncon=ncon, dt=m.opt.timestep, samples=samples)

    def bucket(arr):
        b = [(arr == 0).mean(), ((arr >= 1) & (arr <= 2)).mean(),
             ((arr >= 3) & (arr <= 4)).mean(), ((arr >= 5) & (arr <= 8)).mean(),
             (arr > 8).mean()]
        return b

    print(f"\ntrajectory ncon: min={ncon.min()} max={ncon.max()} "
          f"mean={ncon.mean():.1f}  (settles to {ncon[-50:].mean():.1f})")
    print("contact-count buckets (fraction of samples), cross-world (N=4000 perturbed states):")
    names = ["ballistic(0)", "1-2", "3-4", "5-8", "tail(>8)"]
    for n, f in zip(names, bucket(samples)):
        print(f"  {n:14s} {f*100:5.1f}%")
    print(f"  cross-world ncon: mean={samples.mean():.1f} max={samples.max()} "
          f"std={samples.std():.1f}")
    # SIMT efficiency proxy: within a 32-world warp, work = max ncon in the warp.
    warp_max = samples[:len(samples)//32*32].reshape(-1, 32).max(1)
    print(f"  32-world warp max-ncon: mean={warp_max.mean():.1f} "
          f"(SIMT does max, not mean -> ~{warp_max.mean()/max(samples.mean(),1e-9):.1f}x waste)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
