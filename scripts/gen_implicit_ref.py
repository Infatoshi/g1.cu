"""Stage A accuracy oracle: MuJoCo implicitfast reference for the IMPLICIT PD integrator.

Generates a short controlled trajectory of the G1 with the position (PD) actuators active
under MuJoCo's implicitfast integrator (the integrator g1_env.cu now matches), with foot
contacts DISABLED so we validate the smooth+actuator implicit-in-velocity step in isolation
(the contact path is validated separately by the existing acceptance tests).

Two regimes are written:
  - "air": base lifted 2 m, random PD targets + random initial joint velocities. Excites the
    stiff kd damping that implicitfast implicitizes. This is the decisive integrator check.

Output: bench/implicit_ref.npz with qpos/qvel per step + the control sequence + dt + B diag.
The CUDA side (scripts/test_implicit.py via build/test_implicit) replays the same controls
through env_qacc's implicit step and compares qvel/qpos.
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(HERE, "models", "g1.xml")
OUT = os.path.join(HERE, "bench", "implicit_ref.npz")

N_STEPS = 100
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    m = mujoco.MjModel.from_xml_path(MODEL)
    assert mujoco.mjtIntegrator(m.opt.integrator).name == "mjINT_IMPLICITFAST", \
        "model default integrator must be implicitfast"
    # isolate smooth + actuator implicit step: disable contacts, limits, frictionloss
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)

    nq, nv, nu = m.nq, m.nv, m.nu
    dt = m.opt.timestep
    act_dof = np.array([int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in range(nu)])
    act_qadr = np.array([int(m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in range(nu)])
    kd = -m.actuator_biasprm[:, 2]
    B = np.zeros(nv)
    B[act_dof] = kd  # + dof_damping (0 for G1)

    d = mujoco.MjData(m)
    key = m.key("stand")
    d.qpos[:] = key.qpos
    d.qpos[2] += 2.0  # lift into air
    d.qvel[6:] = 0.1 * rng.standard_normal(nv - 6)
    # fixed random PD target sequence (residual on the stand pose), held per step
    qstand = key.qpos.copy()
    targets = np.zeros((N_STEPS, nu))
    base_t = qstand[act_qadr] + 0.1 * rng.standard_normal(nu)
    for t in range(N_STEPS):
        targets[t] = base_t + 0.05 * np.sin(0.3 * t + np.arange(nu))
    mujoco.mj_forward(m, d)

    qpos = np.zeros((N_STEPS + 1, nq))
    qvel = np.zeros((N_STEPS + 1, nv))
    qpos[0] = d.qpos
    qvel[0] = d.qvel
    for t in range(N_STEPS):
        d.ctrl[:] = targets[t]
        mujoco.mj_step(m, d)
        qpos[t + 1] = d.qpos
        qvel[t + 1] = d.qvel

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, dt=dt, nq=nq, nv=nv, nu=nu,
             qpos=qpos, qvel=qvel, targets=targets,
             act_dof=act_dof, act_qadr=act_qadr, kd=kd, Bdiag=B,
             qstand=qstand)
    print(f"wrote {OUT}: {N_STEPS} steps, integrator=implicitfast, contacts OFF")
    print(f"  base z {qpos[0,2]:.3f} -> {qpos[-1,2]:.3f}")
    print(f"  |qvel| step0 {np.linalg.norm(qvel[0]):.3f} -> stepN {np.linalg.norm(qvel[-1]):.3f}")
    print(f"  kd range [{kd.min():.2f}, {kd.max():.2f}], dt={dt}")


if __name__ == "__main__":
    main()
