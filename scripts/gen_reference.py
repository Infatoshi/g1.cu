"""M0 reference trajectory generator (MuJoCo).

Generates a golden trajectory for the SMOOTH multibody dynamics of the Unitree G1,
in isolation from contacts and actuation, so a from-scratch CUDA engine can be
validated stage-by-stage (FK -> CRBA -> RNE -> solve -> integrate).

Config (documented, see DEVLOG):
  - integrator = Euler (semi-implicit); model has zero joint damping/stiffness, so
    this is pure: qvel += dt*qacc; qpos = integratePos(qpos, qvel_new, dt).
  - contacts + actuation DISABLED. armature stays on (adds to M diagonal).
  - initial state = 'stand' keyframe, base lifted into the air, small deterministic
    qvel perturbation to excite Coriolis/centrifugal terms.

Outputs:
  bench/ref_traj.npz  -- trajectory + per-step diagnostics for Python-side checks.
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(HERE, "models", "g1.xml")
OUT = os.path.join(HERE, "bench", "ref_traj.npz")

N_STEPS = 300
BASE_LIFT = 5.0          # raise base z so nothing is near the ground
SEED = 0


def probe_freejoint_convention(model):
    """Empirically determine the frame of free-joint qvel[0:6].

    Set a pure linear qvel, then a pure angular qvel, and read the resulting
    body spatial velocity (mj_objectVelocity) in both world and local frames.
    Returns a human-readable description; printed so the CUDA side matches.
    """
    d = mujoco.MjData(model)
    # tilt the base so global != local and the frames are distinguishable
    q = np.array([np.cos(0.4), 0.0, np.sin(0.4), 0.0])  # rot about +y
    d.qpos[3:7] = q / np.linalg.norm(q)

    out = {}
    for label, qv in (("lin_x", [1, 0, 0, 0, 0, 0]), ("ang_x", [0, 0, 0, 1, 0, 0])):
        d.qvel[:6] = qv
        mujoco.mj_forward(model, d)
        v_world = np.zeros(6)
        v_local = np.zeros(6)
        # flg_local=0 -> world frame, =1 -> body local frame; obj=BODY, pelvis id=1
        mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, 1, v_world, 0)
        mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, 1, v_local, 1)
        # mj_objectVelocity returns [angular(3), linear(3)]
        out[label] = (v_world.copy(), v_local.copy())
        d.qvel[:6] = 0
    return out


def main():
    rng = np.random.default_rng(SEED)
    model = mujoco.MjModel.from_xml_path(MODEL)

    # --- config for clean smooth-dynamics reference ---
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    # disable the entire constraint solver (contacts + joint limits + equality) and
    # actuation -> pure smooth multibody dynamics: M*qacc = -qfrc_bias.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)

    nq, nv = model.nq, model.nv
    print(f"model: nq={nq} nv={nv} nu={model.nu} nbody={model.nbody} dt={model.opt.timestep}")
    print(f"integrator={mujoco.mjtIntegrator(model.opt.integrator).name} "
          f"disableflags=0x{model.opt.disableflags:x}")

    print("\n--- free-joint qvel convention probe (returns [ang(3), lin(3)]) ---")
    probe = probe_freejoint_convention(model)
    for label, (vw, vl) in probe.items():
        print(f"  qvel={label}: v_world={np.round(vw,4)}  v_local={np.round(vl,4)}")

    # --- initial state ---
    d = mujoco.MjData(model)
    key = model.key("stand")
    d.qpos[:] = key.qpos
    d.qpos[2] += BASE_LIFT
    # small deterministic perturbation to all velocity dofs to excite dynamics
    d.qvel[:] = rng.uniform(-0.3, 0.3, size=nv)
    mujoco.mj_forward(model, d)

    nbody = model.nbody
    qpos = np.zeros((N_STEPS + 1, nq))
    qvel = np.zeros((N_STEPS + 1, nv))
    qacc = np.zeros((N_STEPS + 1, nv))
    qfrc_bias = np.zeros((N_STEPS + 1, nv))
    xpos = np.zeros((N_STEPS + 1, nbody, 3))
    xquat = np.zeros((N_STEPS + 1, nbody, 4))
    xipos = np.zeros((N_STEPS + 1, nbody, 3))   # body com, world
    M = np.zeros((nv, nv))  # dense mass matrix at step 0

    def record(t):
        qpos[t] = d.qpos
        qvel[t] = d.qvel
        qacc[t] = d.qacc
        qfrc_bias[t] = d.qfrc_bias
        xpos[t] = d.xpos
        xquat[t] = d.xquat
        xipos[t] = d.xipos

    # step-0 dense mass matrix + com-frame intermediates (for CRBA/RNE bring-up)
    mujoco.mj_forward(model, d)
    mujoco.mj_fullM(model, M, d.qM)
    cdof0 = d.cdof.copy()            # (nv,6) dof motion axes, com frame
    cinert0 = d.cinert.copy()        # (nbody,10) body spatial inertia, com frame
    cvel0 = d.cvel.copy()            # (nbody,6) body spatial velocity, com frame
    subtree_com0 = d.subtree_com.copy()  # (nbody,3)
    record(0)

    for t in range(1, N_STEPS + 1):
        mujoco.mj_step(model, d)
        record(t)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT,
        dt=model.opt.timestep,
        gravity=model.opt.gravity,
        qpos=qpos, qvel=qvel, qacc=qacc, qfrc_bias=qfrc_bias,
        xpos=xpos, xquat=xquat, xipos=xipos, M0=M,
        cdof0=cdof0, cinert0=cinert0, cvel0=cvel0, subtree_com0=subtree_com0,
        qpos_init=qpos[0], qvel_init=qvel[0],
    )
    # init state for the CUDA program (float64; cast to float in C)
    with open(os.path.join(HERE, "bench", "init_state.bin"), "wb") as f:
        qpos[0].astype(np.float64).tofile(f)
        qvel[0].astype(np.float64).tofile(f)
    print(f"\nwrote {OUT}: {N_STEPS} steps")
    print(f"  base z: {qpos[0,2]:.3f} -> {qpos[-1,2]:.3f} (free fall ~{0.5*9.81*(N_STEPS*model.opt.timestep)**2:.3f}m)")
    print(f"  |qvel| step0={np.linalg.norm(qvel[0]):.3f} stepN={np.linalg.norm(qvel[-1]):.3f}")
    print(f"  M0 diag range: {np.diag(M).min():.4f} .. {np.diag(M).max():.4f}  cond~{np.linalg.cond(M):.1f}")


if __name__ == "__main__":
    main()
