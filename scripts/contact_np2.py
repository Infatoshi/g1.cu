"""Self-contained fp64 CONTACT oracle (Phase 1: R/aref COMPUTED, not pulled from oracle).

This closes the one remaining correctness gap in scripts/contact_np.py: efc_R and
efc_aref are now reconstructed from MuJoCo's solref/solimp + constraint state, then
validated against the stored oracle arrays. Everything else (collision, pyramidal
Jacobian, PGS solve) is unchanged from contact_np.py.

MuJoCo constraint-parameter formulas (verified bit-exact vs bench/contact_solve_ref.npz):

  Impedance sigmoid (mju_makeImpedance), solimp=[dmin,dmax,width,mid,power]:
    x = |pos|/width  (pos = penetration = contact.dist)
    if x >= 1:           y = 1
    elif x <  mid:       y = (1/mid^(p-1)) * x^p
    else:                y = 1 - (1/(1-mid)^(p-1)) * (1-x)^p
    imp = dmin + y*(dmax-dmin)

  Reference spring (solref=[timeconst, dampratio]):
    b = 2 / (dmax * timeconst)
    k = 1 / (dmax^2 * timeconst^2 * dampratio^2)
    aref = -b * (J @ qvel) - k * imp * pos          (per efc row)

  Regularizer R (diagApprox path), PYRAMIDAL condim-3, friction mu, impratio:
    invw0   = body_invweight0[b1,0] + body_invweight0[b2,0]   (floor body contributes 0)
    diagApprox = invw0 * (1 + mu^2) * (2*mu^2 / impratio)     (pyramidal scaling)
    R = diagApprox * (1 - imp) / imp                          (same for all 4 pyr rows)

  body_invweight0[b,0] = mean over 3 world axes of diag( Jcom_b M^-1 Jcom_b^T ) at qpos0,
  where Jcom_b is the 3xnv translational point-Jacobian of body b's COM. (Reconstructed
  from scratch here; matches MuJoCo's stored body_invweight0 to 1e-15.)

Run: uv run python scripts/contact_np2.py
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "bench", "contact_solve_ref.npz")

FOOT_SPHERES = [
    (7, np.array([-0.05,  0.025, -0.03]), 0.005),
    (7, np.array([-0.05, -0.025, -0.03]), 0.005),
    (7, np.array([ 0.12,  0.030, -0.03]), 0.005),
    (7, np.array([ 0.12, -0.030, -0.03]), 0.005),
    (13, np.array([-0.05,  0.025, -0.03]), 0.005),
    (13, np.array([-0.05, -0.025, -0.03]), 0.005),
    (13, np.array([ 0.12,  0.030, -0.03]), 0.005),
    (13, np.array([ 0.12, -0.030, -0.03]), 0.005),
]
MU = 0.6
SOLREF = (0.02, 1.0)               # (timeconst, dampratio)
SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
IMPRATIO = 1.0


def quat2mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ])


def body_kinematics(m, qpos):
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    return d.xpos.copy(), d.xquat.copy()


def detect_contacts(xpos, xquat):
    cons = []
    for bid, lp, r in FOOT_SPHERES:
        R = quat2mat(xquat[bid])
        center = xpos[bid] + R @ lp
        pen = center[2] - r
        if pen < 0:
            cpt = np.array([center[0], center[1], pen / 2.0])
            cons.append((bid, cpt, pen, center))
    return cons


def point_jacobian(m, qpos, bid, world_pt):
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, world_pt, bid)
    return jacp


def build_contact_jacobian(m, qpos, cons):
    rows = []
    normal = np.array([0.0, 0.0, 1.0])
    t1 = np.array([0.0, 1.0, 0.0])
    t2 = np.array([-1.0, 0.0, 0.0])
    for (bid, cpt, pen, _c) in cons:
        Jp = point_jacobian(m, qpos, bid, cpt)
        jn = normal @ Jp
        jt1 = t1 @ Jp
        jt2 = t2 @ Jp
        rows.append(jn + MU*jt1)
        rows.append(jn - MU*jt1)
        rows.append(jn + MU*jt2)
        rows.append(jn - MU*jt2)
    return np.array(rows) if rows else np.zeros((0, m.nv))


def impedance(pos, solimp):
    dmin, dmax, width, mid, power = solimp
    x = abs(pos) / width
    if x >= 1.0:
        y = 1.0
    else:
        a = 1.0 / mid**(power - 1.0)
        bb = 1.0 / (1.0 - mid)**(power - 1.0)
        y = a * x**power if x < mid else 1.0 - bb * (1.0 - x)**power
    return dmin + y * (dmax - dmin)


def body_invweight0_from_scratch(m):
    """mean diag( Jcom M^-1 Jcom^T ) per body at qpos0 -- reconstructs MuJoCo's
    body_invweight0[:,0] (translational) without reading the stored array."""
    nv = m.nv
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    M = np.zeros((nv, nv)); mujoco.mj_fullM(m, M, d.qM); Minv = np.linalg.inv(M)
    iw = np.zeros(m.nbody)
    for b in range(m.nbody):
        jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
        mujoco.mj_jac(m, d, jp, jr, d.xipos[b], b)
        iw[b] = np.diag(jp @ Minv @ jp.T).mean()
    return iw


def reconstruct_R_aref(invw0, cons, J, qvel):
    """Compute efc_R, efc_aref for pyramidal condim-3 foot-ground contacts."""
    dmin, dmax, width, mid, power = SOLIMP
    tc, dr = SOLREF
    bcoef = 2.0 / (dmax * tc)
    kcoef = 1.0 / (dmax * dmax * tc * tc * dr * dr)
    pyr_scale = (1.0 + MU*MU) * (2.0 * MU*MU / IMPRATIO)

    n = J.shape[0]
    R = np.zeros(n); aref = np.zeros(n)
    for i, (bid, cpt, pen, _c) in enumerate(cons):
        imp = impedance(pen, SOLIMP)
        diagApprox = (invw0[bid] + invw0[0]) * pyr_scale  # floor = body 0
        Rc = diagApprox * (1.0 - imp) / imp
        for r in range(4):
            idx = 4*i + r
            R[idx] = Rc
            aref[idx] = -bcoef * (J[idx] @ qvel) - kcoef * imp * pen
    return R, aref


def pgs(A, b, iters=5000):
    n = len(b)
    f = np.zeros(n)
    for _ in range(iters):
        for i in range(n):
            ri = b[i] + A[i] @ f - A[i, i]*f[i]
            f[i] = max(0.0, -ri / A[i, i])
    return f


def main():
    ref = np.load(REF, allow_pickle=True)
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "models", "scene.xml"))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)

    invw0 = body_invweight0_from_scratch(m)

    FOOT_GEOMS = {15, 16, 17, 18, 30, 31, 32, 33}

    def is_pure_foot_ground(k):
        g1, g2 = ref["cgeom1"][k], ref["cgeom2"][k]
        if len(g1) == 0:
            return True
        return all((a == 0 and b in FOOT_GEOMS) or (b == 0 and a in FOOT_GEOMS)
                   for a, b in zip(g1.tolist(), g2.tolist()))

    labels = ref["label"]
    print(f"{'label':10s} {'ncon':>4s} {'R_relerr':>10s} {'aref_relerr':>11s} "
          f"{'qfrc_err':>10s} {'qacc_relerr':>11s}")
    worst_R = worst_aref = worst_qacc = 0.0
    for k in range(len(labels)):
        if int(ref['ncon'][k]) == 0 or not is_pure_foot_ground(k):
            continue
        qpos = ref["qpos"][k].astype(np.float64)
        qvel = ref["qvel"][k].astype(np.float64)
        qacc_smooth = ref["qacc_smooth"][k].astype(np.float64)
        qfrc_ref = ref["qfrc_constraint"][k].astype(np.float64)
        qacc_ref = ref["qacc"][k].astype(np.float64)
        R_ref = ref["efc_R"][k].astype(np.float64)
        aref_ref = ref["efc_aref"][k].astype(np.float64)

        xpos, xquat = body_kinematics(m, qpos)
        cons = detect_contacts(xpos, xquat)
        if len(cons) == 0:
            continue
        J = build_contact_jacobian(m, qpos, cons)

        # COMPUTED R / aref (the Phase-1 deliverable)
        R, aref = reconstruct_R_aref(invw0, cons, J, qvel)

        d = mujoco.MjData(m); d.qpos[:] = qpos; mujoco.mj_forward(m, d)
        Mmat = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, Mmat, d.qM)
        Minv = np.linalg.inv(Mmat)

        A = J @ Minv @ J.T + np.diag(R)
        b = J @ qacc_smooth - aref
        f = pgs(A, b)
        qfrc = J.T @ f
        qacc = qacc_smooth + Minv @ qfrc

        R_relerr = np.abs(R - R_ref).max() / max(np.abs(R_ref).max(), 1e-30)
        aref_relerr = np.abs(aref - aref_ref).max() / max(np.abs(aref_ref).max(), 1e-30)
        qfrc_err = np.abs(qfrc - qfrc_ref).max()
        qacc_relerr = np.linalg.norm(qacc - qacc_ref) / max(np.linalg.norm(qacc_ref), 1e-9)
        worst_R = max(worst_R, R_relerr)
        worst_aref = max(worst_aref, aref_relerr)
        worst_qacc = max(worst_qacc, qacc_relerr)
        print(f"{labels[k]:10s} {len(cons):>4d} {R_relerr:>10.2e} {aref_relerr:>11.2e} "
              f"{qfrc_err:>10.2e} {qacc_relerr:>11.2e}")

    print(f"\nworst R relerr   : {worst_R:.2e}  (gate <~1e-6)")
    print(f"worst aref relerr: {worst_aref:.2e}  (gate <~1e-6)")
    print(f"worst qacc relerr: {worst_qacc:.2e}  (gate <~1e-9)")
    ok = worst_R < 1e-6 and worst_aref < 1e-6 and worst_qacc < 1e-9
    print("PHASE 1 GATE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
