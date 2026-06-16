"""Richer contact ORACLE for validating a from-scratch foot-ground contact solver.

The existing bench/contact_ref.npz only stores qpos + ncon -- enough for the
divergence profile, but NOT enough to validate a solver (no qacc, no contact
forces, no contact geometry per state). This script regenerates a *solver*
oracle: for a handful of representative states spanning the ncon buckets, it
dumps everything needed to check a contact step bit-for-bit against MuJoCo:

  per state: qpos, qvel, ncon, contact points/normals/penetration/geom ids,
             efc_J (constraint Jacobian), efc_R/efc_aref (MuJoCo's soft-constraint
             regularization + reference accel), qfrc_constraint (contact qfrc),
             qacc (with contacts), and qacc_smooth (no constraints).

ISOLATION: actuation OFF, and crucially dof FRICTIONLOSS + joint LIMIT constraints
OFF, so efc / qfrc_constraint / qacc reflect ONLY foot-ground contacts -- the thing
our solver computes. (With them on, nefc mixes 29 dof-friction + 2 limit rows in
and the comparison is apples-to-oranges.)

Writes bench/contact_solve_ref.npz.
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "bench", "contact_solve_ref.npz")
SEED = 1


def build_model():
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "models", "scene.xml"))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    # isolate pure foot-ground contact: kill dof dry friction + joint limits
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    return m


def collect_state(m, d):
    """Dense snapshot of the contact problem at the current MjData."""
    mujoco.mj_forward(m, d)
    nv, ncon, nefc = m.nv, d.ncon, d.nefc
    # smooth (unconstrained) acceleration: forward with constraints disabled
    save = m.opt.disableflags
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    d2.qvel[:] = d.qvel
    mujoco.mj_forward(m, d2)
    qacc_smooth = d2.qacc.copy()
    m.opt.disableflags = save

    J = np.array(d.efc_J).reshape(nefc, nv) if nefc else np.zeros((0, nv))
    # per-contact geometry
    cgeom1 = np.array([d.contact[i].geom1 for i in range(ncon)], dtype=np.int32)
    cgeom2 = np.array([d.contact[i].geom2 for i in range(ncon)], dtype=np.int32)
    cdist = np.array([d.contact[i].dist for i in range(ncon)])
    cpos = np.array([d.contact[i].pos for i in range(ncon)]).reshape(ncon, 3)
    cframe = np.array([d.contact[i].frame for i in range(ncon)]).reshape(ncon, 9)
    cfric = np.array([d.contact[i].friction for i in range(ncon)]).reshape(ncon, 5)
    cdim = np.array([d.contact[i].dim for i in range(ncon)], dtype=np.int32)
    return dict(
        qpos=d.qpos.copy(), qvel=d.qvel.copy(), ncon=ncon, nefc=nefc,
        qacc=d.qacc.copy(), qacc_smooth=qacc_smooth,
        qfrc_constraint=d.qfrc_constraint.copy(),
        qfrc_bias=d.qfrc_bias.copy(),
        efc_J=J, efc_R=d.efc_R[:nefc].copy(), efc_aref=d.efc_aref[:nefc].copy(),
        efc_D=d.efc_D[:nefc].copy(), efc_type=d.efc_type[:nefc].copy(),
        efc_force=d.efc_force[:nefc].copy(),
        cgeom1=cgeom1, cgeom2=cgeom2, cdist=cdist, cpos=cpos,
        cframe=cframe, cfric=cfric, cdim=cdim,
    )


def main():
    rng = np.random.default_rng(SEED)
    m = build_model()
    d = mujoco.MjData(m)
    d.qpos[:] = m.key("stand").qpos
    d.qpos[2] += 0.15
    mujoco.mj_forward(m, d)

    # roll a drop trajectory, snapshot states binned by ncon bucket
    buckets = {"ballistic": [], "lo": [], "mid": [], "hi": [], "tail": []}

    def bname(n):
        if n == 0: return "ballistic"
        if n <= 2: return "lo"
        if n <= 4: return "mid"
        if n <= 8: return "hi"
        return "tail"

    for t in range(600):
        mujoco.mj_step(m, d)
        b = bname(d.ncon)
        if len(buckets[b]) < 3:
            # perturb velocity a touch so states aren't degenerate
            dd = mujoco.MjData(m)
            dd.qpos[:] = d.qpos
            dd.qvel[:] = d.qvel + rng.normal(0, 0.05, size=m.nv)
            buckets[b].append(collect_state(m, dd))

    states = []
    labels = []
    for bk, lst in buckets.items():
        for s in lst:
            states.append(s)
            labels.append(bk)

    # flatten to a savez (object arrays, variable nefc/ncon per state)
    out = {f"label": np.array(labels)}
    keys = states[0].keys() if states else []
    for k in keys:
        out[k] = np.array([s[k] for s in states], dtype=object)
    np.savez(OUT, **out)
    print(f"wrote {OUT}  ({len(states)} states)")
    for s, lab in zip(states, labels):
        print(f"  {lab:10s} ncon={s['ncon']:2d} nefc={s['nefc']:2d} "
              f"|qfrc_c|={np.linalg.norm(s['qfrc_constraint']):8.2f} "
              f"|qacc|={np.linalg.norm(s['qacc']):7.2f}")


if __name__ == "__main__":
    main()
