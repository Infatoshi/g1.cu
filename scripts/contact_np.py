"""fp64 numpy CONTACT oracle prototype (mirror of scripts/aba_np.py for the contact path).

Goal: a from-scratch single-world foot-ground contact step that matches MuJoCo's
qfrc_constraint / qacc on real states, BEFORE any CUDA. Same proven pattern the
smooth-dynamics build used (aba_np.py nailed correctness first).

WHAT THIS BUILDS FROM SCRATCH (the genuinely-new physics):
  1. collision detection: 8 foot spheres (radius 5mm, on ankle_roll bodies 7 & 13)
     vs the z=0 ground plane. contact iff sphere_center_z - r < 0.
  2. contact Jacobian J (nefc x nv): maps qvel -> contact-point velocity in the
     contact frame, in MuJoCo's PYRAMIDAL condim-3 layout (4 rows / contact:
     normal +/- mu*tangent for the two tangent dirs).
  3. the boxed-QP contact solve via Projected Gauss-Seidel (PGS):
        min_f  0.5 f^T A f + f^T b ,  f >= 0 ,
        A = J M^-1 J^T + diag(R),  b = J*qacc_smooth - aref.
     qfrc_constraint = J^T f ;  qacc = qacc_smooth + M^-1 J^T f.

WHAT IS STILL TAKEN FROM THE ORACLE (next correctness step, see contact_DESIGN.md):
  - efc_R (constraint regularization) and efc_aref (reference accel). These come
    from MuJoCo's solref/solimp; reconstructing them bit-exactly needs MuJoCo's
    internal impedance sigmoid + impratio scaling on pyramidal friction rows
    (R reconstruct currently off by ~1e3, aref off too -> NOT yet matched). We
    pull them from the ref so the COLLISION + JACOBIAN + SOLVER core is validated
    in isolation first. Matching R/aref from solref/solimp is the immediate TODO.
  - M^-1 and qacc_smooth come from the validated ABA path in production; here we
    read M from the oracle to isolate the contact solve from ABA noise.

Run: uv run python scripts/contact_np.py
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "bench", "contact_solve_ref.npz")

# foot collision spheres: (host_body_id, local_pos, radius). From models/scene.xml,
# exported by the (to-be-extended) export_model.py. body 7 = left_ankle_roll_link,
# body 13 = right_ankle_roll_link. radius 5mm, friction 0.6, condim 3.
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
MU = 0.6  # tangential friction coeff (foot priority wins)


def quat2mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ])


def body_kinematics(m, qpos):
    """World pos/rot of every body via MuJoCo FK (stands in for our forward_kinematics)."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    return d.xpos.copy(), d.xquat.copy()


def detect_contacts(xpos, xquat):
    """Sphere-vs-plane, reproducing MuJoCo's geometry exactly.
      penetration  cdist = center_z - r
      contact pt   cpos  = [cx, cy, (center_z - r)/2]   (midpoint surface<->plane)
    Returns list of (body_id, contact_point, penetration, sphere_world_center)."""
    cons = []
    for bid, lp, r in FOOT_SPHERES:
        R = quat2mat(xquat[bid])
        center = xpos[bid] + R @ lp
        pen = center[2] - r          # signed dist of sphere SURFACE to plane (z=0)
        if pen < 0:                   # penetrating
            cpt = np.array([center[0], center[1], pen / 2.0])  # MuJoCo midpoint
            cons.append((bid, cpt, pen, center))
    return cons


def point_jacobian(m, qpos, bid, world_pt):
    """3xnv translational Jacobian of a world point fixed to body bid (MuJoCo mj_jac)."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, world_pt, bid)
    return jacp


def build_contact_jacobian(m, qpos, cons):
    """Pyramidal condim-3 J: 4 rows per contact = [n + mu*t1, n - mu*t1, n + mu*t2, n - mu*t2].
    Frame matches MuJoCo's deterministic basis for a +z normal:
      normal=[0,0,1], t1=[0,1,0], t2=[-1,0,0]."""
    rows = []
    normal = np.array([0.0, 0.0, 1.0])
    t1 = np.array([0.0, 1.0, 0.0])
    t2 = np.array([-1.0, 0.0, 0.0])
    for (bid, cpt, pen, _c) in cons:
        Jp = point_jacobian(m, qpos, bid, cpt)   # 3 x nv: d(world point vel)/d(qvel)
        jn = normal @ Jp
        jt1 = t1 @ Jp
        jt2 = t2 @ Jp
        rows.append(jn + MU*jt1)
        rows.append(jn - MU*jt1)
        rows.append(jn + MU*jt2)
        rows.append(jn - MU*jt2)
    return np.array(rows) if rows else np.zeros((0, m.nv))


def pgs(A, b, iters=5000):
    """Projected Gauss-Seidel for min 0.5 f^T A f + f^T b, f >= 0."""
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

    FOOT_GEOMS = {15, 16, 17, 18, 30, 31, 32, 33}

    def is_pure_foot_ground(k):
        g1, g2 = ref["cgeom1"][k], ref["cgeom2"][k]
        if len(g1) == 0:
            return True
        return all((a == 0 and b in FOOT_GEOMS) or (b == 0 and a in FOOT_GEOMS)
                   for a, b in zip(g1.tolist(), g2.tolist()))

    labels = ref["label"]
    print(f"{'label':10s} {'ncon':>4s} {'ncon_ours':>9s}  {'qfrc_err':>10s} {'qacc_relerr':>11s} {'qacc_maxerr':>11s}")
    worst = 0.0
    for k in range(len(labels)):
        if not is_pure_foot_ground(k):
            print(f"{labels[k]:10s} {int(ref['ncon'][k]):>4d}  SKIP (ref has non-foot-ground contacts: "
                  f"arm/leg self-collision out of solver scope)")
            continue
        qpos = ref["qpos"][k].astype(np.float64)
        ncon_ref = int(ref["ncon"][k])
        J_ref = ref["efc_J"][k].astype(np.float64)
        R = ref["efc_R"][k].astype(np.float64)
        aref = ref["efc_aref"][k].astype(np.float64)
        qacc_smooth = ref["qacc_smooth"][k].astype(np.float64)
        qfrc_ref = ref["qfrc_constraint"][k].astype(np.float64)
        qacc_ref = ref["qacc"][k].astype(np.float64)

        # --- our collision detection ---
        xpos, xquat = body_kinematics(m, qpos)
        cons = detect_contacts(xpos, xquat)
        ncon_ours = len(cons)

        if ncon_ours == 0:
            print(f"{labels[k]:10s} {ncon_ref:>4d} {ncon_ours:>9d}  {'--':>10s} {'--':>11s} {'--':>11s}")
            continue

        # --- our contact Jacobian ---
        J = build_contact_jacobian(m, qpos, cons)

        # sanity: does our J match MuJoCo's efc_J (up to row order)? (only when ncon matches)
        # --- M^-1 from oracle (isolate contact solve from ABA) ---
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        Mmat = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, Mmat, d.qM)
        Minv = np.linalg.inv(Mmat)

        # --- assemble + solve (R/aref from oracle for now) ---
        A = J @ Minv @ J.T + np.diag(R)
        b = J @ qacc_smooth - aref
        f = pgs(A, b)
        qfrc = J.T @ f
        qacc = qacc_smooth + Minv @ qfrc

        qfrc_err = np.abs(qfrc - qfrc_ref).max()
        qacc_relerr = np.linalg.norm(qacc - qacc_ref) / max(np.linalg.norm(qacc_ref), 1e-9)
        qacc_maxerr = np.abs(qacc - qacc_ref).max()
        worst = max(worst, qacc_relerr)
        match = "OK" if ncon_ours == ncon_ref else f"!={ncon_ref}"
        print(f"{labels[k]:10s} {ncon_ref:>4d} {ncon_ours:>9d}  {qfrc_err:>10.2e} {qacc_relerr:>11.2e} {qacc_maxerr:>11.2e}  {match}")

    print(f"\nworst qacc relerr across states: {worst:.2e}")
    print("(R/aref taken from oracle; collision + Jacobian + PGS solve are from scratch.)")


if __name__ == "__main__":
    main()
