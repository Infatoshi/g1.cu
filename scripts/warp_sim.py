"""Specialized G1 smooth dynamics in NVIDIA Warp -- the compiler-tax experiment.

Same algorithm as our hand-CUDA src/dynamics.cuh (FK -> CRBA -> RNE -> Cholesky ->
semi-implicit Euler), one thread per world, hardcoded single model, fp32. Question:
how much of the 3x hand-CUDA advantage (7.4e6) survives going through Warp, vs the
general MuJoCo-Warp (2.4e6)?

Validates world 0 vs the M0 oracle (bench/ref_traj.npz) then benchmarks throughput.
Quaternion note: Warp wp.quat is (x,y,z,w); MuJoCo is (w,x,y,z) -> convert on upload.
"""
import os, time, argparse
import numpy as np
import mujoco
import warp as wp

wp.init()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- model constants ----
M = mujoco.MjModel.from_xml_path(os.path.join(HERE, "models", "g1.xml"))
NB, NV, NQ, NJ = M.nbody, M.nv, M.nq, M.njnt

vecNV = wp.types.vector(length=NV, dtype=wp.float32)
matNVNV = wp.types.matrix(shape=(NV, NV), dtype=wp.float32)
matNV6 = wp.types.matrix(shape=(NV, 6), dtype=wp.float32)
matNB3 = wp.types.matrix(shape=(NB, 3), dtype=wp.float32)
matNB4 = wp.types.matrix(shape=(NB, 4), dtype=wp.float32)
matNB9 = wp.types.matrix(shape=(NB, 9), dtype=wp.float32)
matNB36 = wp.types.matrix(shape=(NB, 36), dtype=wp.float32)
matNV3 = wp.types.matrix(shape=(NV, 3), dtype=wp.float32)


def wquat(wxyz):  # MuJoCo (w,x,y,z) -> Warp (x,y,z,w)
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)


dev = "cuda:0"
c_parent = wp.array(M.body_parentid.astype(np.int32), dtype=wp.int32, device=dev)
c_bpos = wp.array(M.body_pos.astype(np.float32), dtype=wp.vec3, device=dev)
c_bquat = wp.array(np.array([wquat(q) for q in M.body_quat]), dtype=wp.quat, device=dev)
c_ipos = wp.array(M.body_ipos.astype(np.float32), dtype=wp.vec3, device=dev)
c_iquat = wp.array(np.array([wquat(q) for q in M.body_iquat]), dtype=wp.quat, device=dev)
c_mass = wp.array(M.body_mass.astype(np.float32), dtype=wp.float32, device=dev)
c_inertia = wp.array(M.body_inertia.astype(np.float32), dtype=wp.vec3, device=dev)
c_bdofadr = wp.array(M.body_dofadr.astype(np.int32), dtype=wp.int32, device=dev)
c_bdofnum = wp.array(M.body_dofnum.astype(np.int32), dtype=wp.int32, device=dev)
c_bjntadr = wp.array(M.body_jntadr.astype(np.int32), dtype=wp.int32, device=dev)
c_bjntnum = wp.array(M.body_jntnum.astype(np.int32), dtype=wp.int32, device=dev)
c_jtype = wp.array(M.jnt_type.astype(np.int32), dtype=wp.int32, device=dev)
c_jbody = wp.array(M.jnt_bodyid.astype(np.int32), dtype=wp.int32, device=dev)
c_jqadr = wp.array(M.jnt_qposadr.astype(np.int32), dtype=wp.int32, device=dev)
c_jdadr = wp.array(M.jnt_dofadr.astype(np.int32), dtype=wp.int32, device=dev)
c_jaxis = wp.array(M.jnt_axis.astype(np.float32), dtype=wp.vec3, device=dev)
c_jpos = wp.array(M.jnt_pos.astype(np.float32), dtype=wp.vec3, device=dev)
c_arm = wp.array(M.dof_armature.astype(np.float32), dtype=wp.float32, device=dev)
GRAV = wp.vec3(float(M.opt.gravity[0]), float(M.opt.gravity[1]), float(M.opt.gravity[2]))
DT = wp.constant(float(M.opt.timestep))
_NB = wp.constant(NB); _NV = wp.constant(NV); _NJ = wp.constant(NJ); _NQ = wp.constant(NQ)


@wp.kernel
def step_kernel(
    qpos: wp.array(dtype=wp.float32), qvel: wp.array(dtype=wp.float32),
    parent: wp.array(dtype=wp.int32), bpos: wp.array(dtype=wp.vec3),
    bquat: wp.array(dtype=wp.quat), ipos: wp.array(dtype=wp.vec3),
    iquat: wp.array(dtype=wp.quat), mass: wp.array(dtype=wp.float32),
    inertia: wp.array(dtype=wp.vec3), bdofadr: wp.array(dtype=wp.int32),
    bdofnum: wp.array(dtype=wp.int32), bjntadr: wp.array(dtype=wp.int32),
    bjntnum: wp.array(dtype=wp.int32), jtype: wp.array(dtype=wp.int32),
    jbody: wp.array(dtype=wp.int32), jqadr: wp.array(dtype=wp.int32),
    jdadr: wp.array(dtype=wp.int32), jaxis: wp.array(dtype=wp.vec3),
    jpos: wp.array(dtype=wp.vec3), arm: wp.array(dtype=wp.float32)):
    w = wp.tid()
    qo = w * _NQ
    vo = w * _NV

    xpos = matNB3(); xquat = matNB4(); xipos = matNB3()
    xquat[0, 0] = 0.0; xquat[0, 1] = 0.0; xquat[0, 2] = 0.0; xquat[0, 3] = 1.0

    # ---- forward kinematics ----
    for b in range(1, _NB):
        p = parent[b]
        qp = wp.quat(xquat[p, 0], xquat[p, 1], xquat[p, 2], xquat[p, 3])
        pp = wp.vec3(xpos[p, 0], xpos[p, 1], xpos[p, 2])
        pos = pp + wp.quat_rotate(qp, bpos[b])
        quat = wp.mul(qp, bquat[b])
        ja = bjntadr[b]
        for jj in range(bjntnum[b]):
            j = ja + jj
            qa = qo + jqadr[j]
            if jtype[j] == 0:
                pos = wp.vec3(qpos[qa + 0], qpos[qa + 1], qpos[qa + 2])
                quat = wp.normalize(wp.quat(qpos[qa + 4], qpos[qa + 5], qpos[qa + 6], qpos[qa + 3]))
            if jtype[j] == 3:
                anchor = pos + wp.quat_rotate(quat, jpos[j])
                quat = wp.mul(quat, wp.quat_from_axis_angle(jaxis[j], qpos[qa]))
                pos = anchor - wp.quat_rotate(quat, jpos[j])
        xpos[b, 0] = pos[0]; xpos[b, 1] = pos[1]; xpos[b, 2] = pos[2]
        xquat[b, 0] = quat[0]; xquat[b, 1] = quat[1]; xquat[b, 2] = quat[2]; xquat[b, 3] = quat[3]
        cw = pos + wp.quat_rotate(quat, ipos[b])
        xipos[b, 0] = cw[0]; xipos[b, 1] = cw[1]; xipos[b, 2] = cw[2]

    pref = wp.vec3(xpos[1, 0], xpos[1, 1], xpos[1, 2])
    q1 = wp.quat(xquat[1, 0], xquat[1, 1], xquat[1, 2], xquat[1, 3])
    R1 = wp.quat_to_matrix(q1)

    # ---- motion axes S (about pref, world axes) ----
    S = matNV6()
    for j in range(_NJ):
        b = jbody[j]; d = jdadr[j]
        if jtype[j] == 0:
            S[d + 0, 3] = 1.0; S[d + 1, 4] = 1.0; S[d + 2, 5] = 1.0
            for k in range(3):
                ax = wp.vec3(R1[0, k], R1[1, k], R1[2, k])
                S[d + 3 + k, 0] = ax[0]; S[d + 3 + k, 1] = ax[1]; S[d + 3 + k, 2] = ax[2]
        if jtype[j] == 3:
            qb = wp.quat(xquat[b, 0], xquat[b, 1], xquat[b, 2], xquat[b, 3])
            aw = wp.quat_rotate(qb, jaxis[j])
            anchor = wp.vec3(xpos[b, 0], xpos[b, 1], xpos[b, 2]) + wp.quat_rotate(qb, jpos[j])
            r = anchor - pref
            rxa = wp.cross(r, aw)
            S[d, 0] = aw[0]; S[d, 1] = aw[1]; S[d, 2] = aw[2]
            S[d, 3] = rxa[0]; S[d, 4] = rxa[1]; S[d, 5] = rxa[2]

    # ---- per-body spatial inertia about pref: store Iw(9), c(3), m ----
    Iw = matNB9(); Cc = matNB3(); Mm = vecNV()  # Mm reused as mass[NB<=NV]
    Icp = matNB36()
    for b in range(1, _NB):
        qb = wp.quat(xquat[b, 0], xquat[b, 1], xquat[b, 2], xquat[b, 3])
        Ri = wp.quat_to_matrix(wp.mul(qb, iquat[b]))
        di = inertia[b]
        # world inertia Ri diag(di) Ri^T
        m9 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for ii in range(3):
            for jj in range(3):
                m9[ii, jj] = Ri[ii, 0] * di[0] * Ri[jj, 0] + Ri[ii, 1] * di[1] * Ri[jj, 1] + Ri[ii, 2] * di[2] * Ri[jj, 2]
        mb = mass[b]
        c = wp.vec3(xipos[b, 0] - pref[0], xipos[b, 1] - pref[1], xipos[b, 2] - pref[2])
        cc = wp.dot(c, c)
        # A = Iw + m(cc I - c c^T)
        A00 = m9[0, 0] + mb * (cc - c[0] * c[0]); A01 = m9[0, 1] + mb * (-c[0] * c[1]); A02 = m9[0, 2] + mb * (-c[0] * c[2])
        A10 = m9[1, 0] + mb * (-c[1] * c[0]); A11 = m9[1, 1] + mb * (cc - c[1] * c[1]); A12 = m9[1, 2] + mb * (-c[1] * c[2])
        A20 = m9[2, 0] + mb * (-c[2] * c[0]); A21 = m9[2, 1] + mb * (-c[2] * c[1]); A22 = m9[2, 2] + mb * (cc - c[2] * c[2])
        # skew(c)
        B00 = 0.0; B01 = -mb * c[2]; B02 = mb * c[1]
        B10 = mb * c[2]; B11 = 0.0; B12 = -mb * c[0]
        B20 = -mb * c[1]; B21 = mb * c[0]; B22 = 0.0
        # rows 0..2 = [A | B], rows 3..5 = [-B | mI]  (6x6 row-major into Icp[b, :])
        A = wp.mat33(A00, A01, A02, A10, A11, A12, A20, A21, A22)
        B = wp.mat33(B00, B01, B02, B10, B11, B12, B20, B21, B22)
        for ii in range(3):
            for jj in range(3):
                Icp[b, 6 * ii + jj] = A[ii, jj]
                Icp[b, 6 * ii + (jj + 3)] = B[ii, jj]
                Icp[b, 6 * (ii + 3) + jj] = -B[ii, jj]
                Icp[b, 6 * (ii + 3) + (jj + 3)] = wp.where(ii == jj, mb, 0.0)
        for k in range(9):
            Iw[b, k] = m9[k // 3, k % 3]
        Cc[b, 0] = c[0]; Cc[b, 1] = c[1]; Cc[b, 2] = c[2]; Mm[b] = mb

    # ---- CRBA: composite inertia (sum subtree) ----
    for b in range(_NB - 1, 0, -1):
        par = parent[b]
        if par >= 1:
            for k in range(36):
                Icp[par, k] = Icp[par, k] + Icp[b, k]
    # M = S^T Icp S
    Mat = matNVNV()
    for b in range(1, _NB):
        da = bdofadr[b]; dn = bdofnum[b]
        for a in range(da, da + dn):
            F0 = float(0.0); F1 = float(0.0); F2 = float(0.0); F3 = float(0.0); F4 = float(0.0); F5 = float(0.0)
            for k in range(6):
                F0 += Icp[b, 0 * 6 + k] * S[a, k]; F1 += Icp[b, 1 * 6 + k] * S[a, k]; F2 += Icp[b, 2 * 6 + k] * S[a, k]
                F3 += Icp[b, 3 * 6 + k] * S[a, k]; F4 += Icp[b, 4 * 6 + k] * S[a, k]; F5 += Icp[b, 5 * 6 + k] * S[a, k]
            jb = b
            while jb >= 1:
                dj = bdofadr[jb]; djn = bdofnum[jb]
                for cc2 in range(dj, dj + djn):
                    v = S[cc2, 0] * F0 + S[cc2, 1] * F1 + S[cc2, 2] * F2 + S[cc2, 3] * F3 + S[cc2, 4] * F4 + S[cc2, 5] * F5
                    Mat[a, cc2] = v; Mat[cc2, a] = v
                jb = parent[jb]
    for a in range(_NV):
        Mat[a, a] = Mat[a, a] + arm[a]

    # ---- RNE bias (qacc=0, gravity via base accel = -g) ----
    VW = matNB3(); VV = matNB3(); AW = matNB3(); AV = matNB3()
    SDW = matNV3(); SDV = matNV3()
    AV[0, 0] = -GRAV[0]; AV[0, 1] = -GRAV[1]; AV[0, 2] = -GRAV[2]
    for b in range(1, _NB):
        par = parent[b]
        wcur = wp.vec3(VW[par, 0], VW[par, 1], VW[par, 2])
        vcur = wp.vec3(VV[par, 0], VV[par, 1], VV[par, 2])
        da = bdofadr[b]; dn = bdofnum[b]
        if dn == 6:
            for a in range(da, da + 3):
                sw = wp.vec3(S[a, 0], S[a, 1], S[a, 2]); sv = wp.vec3(S[a, 3], S[a, 4], S[a, 5])
                ow = wp.cross(wcur, sw); ov = wp.cross(wcur, sv) + wp.cross(vcur, sw)
                SDW[a, 0] = ow[0]; SDW[a, 1] = ow[1]; SDW[a, 2] = ow[2]
                SDV[a, 0] = ov[0]; SDV[a, 1] = ov[1]; SDV[a, 2] = ov[2]
                qd = qvel[vo + a]; wcur = wcur + sw * qd; vcur = vcur + sv * qd
            wr = wcur; vr = vcur
            for a in range(da + 3, da + 6):
                sw = wp.vec3(S[a, 0], S[a, 1], S[a, 2]); sv = wp.vec3(S[a, 3], S[a, 4], S[a, 5])
                ow = wp.cross(wr, sw); ov = wp.cross(wr, sv) + wp.cross(vr, sw)
                SDW[a, 0] = ow[0]; SDW[a, 1] = ow[1]; SDW[a, 2] = ow[2]
                SDV[a, 0] = ov[0]; SDV[a, 1] = ov[1]; SDV[a, 2] = ov[2]
            for a in range(da + 3, da + 6):
                sw = wp.vec3(S[a, 0], S[a, 1], S[a, 2]); sv = wp.vec3(S[a, 3], S[a, 4], S[a, 5])
                qd = qvel[vo + a]; wcur = wcur + sw * qd; vcur = vcur + sv * qd
        else:
            for a in range(da, da + dn):
                sw = wp.vec3(S[a, 0], S[a, 1], S[a, 2]); sv = wp.vec3(S[a, 3], S[a, 4], S[a, 5])
                ow = wp.cross(wcur, sw); ov = wp.cross(wcur, sv) + wp.cross(vcur, sw)
                SDW[a, 0] = ow[0]; SDW[a, 1] = ow[1]; SDW[a, 2] = ow[2]
                SDV[a, 0] = ov[0]; SDV[a, 1] = ov[1]; SDV[a, 2] = ov[2]
                qd = qvel[vo + a]; wcur = wcur + sw * qd; vcur = vcur + sv * qd
        VW[b, 0] = wcur[0]; VW[b, 1] = wcur[1]; VW[b, 2] = wcur[2]
        VV[b, 0] = vcur[0]; VV[b, 1] = vcur[1]; VV[b, 2] = vcur[2]
        # acceleration
        aw_ = wp.vec3(AW[par, 0], AW[par, 1], AW[par, 2])
        av_ = wp.vec3(AV[par, 0], AV[par, 1], AV[par, 2])
        for a in range(da, da + dn):
            qd = qvel[vo + a]
            aw_ = aw_ + wp.vec3(SDW[a, 0], SDW[a, 1], SDW[a, 2]) * qd
            av_ = av_ + wp.vec3(SDV[a, 0], SDV[a, 1], SDV[a, 2]) * qd
        AW[b, 0] = aw_[0]; AW[b, 1] = aw_[1]; AW[b, 2] = aw_[2]
        AV[b, 0] = av_[0]; AV[b, 1] = av_[1]; AV[b, 2] = av_[2]

    # forces f = I a + v x* (I v); apply_inertia with stored Iw,c,m
    FW = matNB3(); FV = matNB3()
    for b in range(1, _NB):
        c = wp.vec3(Cc[b, 0], Cc[b, 1], Cc[b, 2]); mb = Mm[b]
        Iwb = wp.mat33(Iw[b, 0], Iw[b, 1], Iw[b, 2], Iw[b, 3], Iw[b, 4], Iw[b, 5], Iw[b, 6], Iw[b, 7], Iw[b, 8])
        # I * a
        aw_ = wp.vec3(AW[b, 0], AW[b, 1], AW[b, 2]); av_ = wp.vec3(AV[b, 0], AV[b, 1], AV[b, 2])
        Iaw = Iwb * aw_ + mb * (wp.dot(c, c) * aw_ - wp.dot(c, aw_) * c) + mb * wp.cross(c, av_)
        Iav = -mb * wp.cross(c, aw_) + mb * av_
        # I * v
        vw_ = wp.vec3(VW[b, 0], VW[b, 1], VW[b, 2]); vv_ = wp.vec3(VV[b, 0], VV[b, 1], VV[b, 2])
        Ivw = Iwb * vw_ + mb * (wp.dot(c, c) * vw_ - wp.dot(c, vw_) * c) + mb * wp.cross(c, vv_)
        Ivv = -mb * wp.cross(c, vw_) + mb * vv_
        # v x* (I v): [w x n + v x f0 ; w x f0]
        cw = wp.cross(vw_, Ivw) + wp.cross(vv_, Ivv)
        cv = wp.cross(vw_, Ivv)
        fw_ = Iaw + cw; fv_ = Iav + cv
        FW[b, 0] = fw_[0]; FW[b, 1] = fw_[1]; FW[b, 2] = fw_[2]
        FV[b, 0] = fv_[0]; FV[b, 1] = fv_[1]; FV[b, 2] = fv_[2]
    bias = vecNV()
    for b in range(_NB - 1, 0, -1):
        da = bdofadr[b]; dn = bdofnum[b]
        for a in range(da, da + dn):
            bias[a] = S[a, 0] * FW[b, 0] + S[a, 1] * FW[b, 1] + S[a, 2] * FW[b, 2] + S[a, 3] * FV[b, 0] + S[a, 4] * FV[b, 1] + S[a, 5] * FV[b, 2]
        par = parent[b]
        if par >= 1:
            for k in range(3):
                FW[par, k] = FW[par, k] + FW[b, k]; FV[par, k] = FV[par, k] + FV[b, k]

    # ---- solve Mat qacc = -bias (Cholesky) ----
    for i in range(_NV):
        for j in range(i + 1):
            s = Mat[i, j]
            for k in range(j):
                s = s - Mat[i, k] * Mat[j, k]
            if i == j:
                Mat[i, i] = wp.sqrt(s)
            else:
                Mat[i, j] = s / Mat[j, j]
    qacc = vecNV()
    for i in range(_NV):
        s = -bias[i]
        for k in range(i):
            s = s - Mat[i, k] * qacc[k]
        qacc[i] = s / Mat[i, i]
    for i in range(_NV - 1, -1, -1):
        s = qacc[i]
        for k in range(i + 1, _NV):
            s = s - Mat[k, i] * qacc[k]
        qacc[i] = s / Mat[i, i]

    # ---- semi-implicit Euler ----
    for i in range(_NV):
        qvel[vo + i] = qvel[vo + i] + DT * qacc[i]
    for j in range(_NJ):
        t = jtype[j]; qa = qo + jqadr[j]; d = jdadr[j]
        if t == 0:
            qpos[qa + 0] = qpos[qa + 0] + DT * qvel[vo + d + 0]
            qpos[qa + 1] = qpos[qa + 1] + DT * qvel[vo + d + 1]
            qpos[qa + 2] = qpos[qa + 2] + DT * qvel[vo + d + 2]
            wl = wp.vec3(qvel[vo + d + 3], qvel[vo + d + 4], qvel[vo + d + 5])
            ang = wp.length(wl) * DT
            q0 = wp.quat(qpos[qa + 4], qpos[qa + 5], qpos[qa + 6], qpos[qa + 3])
            if ang > 1.0e-9:
                axis = wl / wp.length(wl)
                dq = wp.quat_from_axis_angle(axis, ang)
            else:
                dq = wp.quat(0.5 * wl[0] * DT, 0.5 * wl[1] * DT, 0.5 * wl[2] * DT, 1.0)
            qn = wp.normalize(wp.mul(q0, dq))
            qpos[qa + 3] = qn[3]; qpos[qa + 4] = qn[0]; qpos[qa + 5] = qn[1]; qpos[qa + 6] = qn[2]
        if t == 3:
            qpos[qa] = qpos[qa] + DT * qvel[vo + d]


MODEL_ARGS = [c_parent, c_bpos, c_bquat, c_ipos, c_iquat, c_mass, c_inertia,
              c_bdofadr, c_bdofnum, c_bjntadr, c_bjntnum, c_jtype, c_jbody,
              c_jqadr, c_jdadr, c_jaxis, c_jpos, c_arm]


def make_state(N, qpos0, qvel0):
    qp = np.tile(qpos0.astype(np.float32), N)
    qv = np.tile(qvel0.astype(np.float32), N)
    return wp.array(qp, dtype=wp.float32, device=dev), wp.array(qv, dtype=wp.float32, device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=300)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    ref = np.load(os.path.join(HERE, "bench", "ref_traj.npz"))
    qpos0, qvel0 = ref["qpos_init"], ref["qvel_init"]

    if args.validate:
        qp, qv = make_state(1, qpos0, qvel0)
        for _ in range(args.nsteps):
            wp.launch(step_kernel, dim=1, inputs=[qp, qv] + MODEL_ARGS)
        wp.synchronize()
        out = qp.numpy()[:NQ]
        rq = ref["qpos"][args.nsteps]
        dq = min(np.abs(out[3:7] - rq[3:7]).sum(), np.abs(out[3:7] + rq[3:7]).sum())
        print(f"warp world-0 vs MuJoCo @ step {args.nsteps}: "
              f"base_pos {np.abs(out[:3]-rq[:3]).max():.2e}  quat {dq:.2e}  joints {np.abs(out[7:]-rq[7:]).max():.2e}")
        return

    # throughput sweep -- raw (Python per-step launch) and CUDA-graph captured
    # (graph removes Python launch overhead -> true kernel-bound throughput).
    print("specialized G1 dynamics in Warp (1 thread/world):")
    print(f"{'N':>9} {'python-loop':>14} {'cuda-graph':>14}")
    for N in (1024, 4096, 16384, 65536, 262144):
        qp, qv = make_state(N, qpos0, qvel0)
        for _ in range(20):
            wp.launch(step_kernel, dim=N, inputs=[qp, qv] + MODEL_ARGS)
        wp.synchronize()
        # python-loop timing
        qp, qv = make_state(N, qpos0, qvel0)
        t0 = time.perf_counter()
        for _ in range(args.nsteps):
            wp.launch(step_kernel, dim=N, inputs=[qp, qv] + MODEL_ARGS)
        wp.synchronize()
        esps_py = N * args.nsteps / (time.perf_counter() - t0)
        # cuda-graph timing
        qp, qv = make_state(N, qpos0, qvel0)
        with wp.ScopedCapture() as cap:
            for _ in range(args.nsteps):
                wp.launch(step_kernel, dim=N, inputs=[qp, qv] + MODEL_ARGS)
        graph = cap.graph
        wp.capture_launch(graph); wp.synchronize()  # warm
        qp2, qv2 = make_state(N, qpos0, qvel0)  # graph keeps refs to qp,qv; reset their data
        wp.copy(qp, qp2); wp.copy(qv, qv2); wp.synchronize()
        t0 = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize()
        esps_g = N * args.nsteps / (time.perf_counter() - t0)
        print(f"{N:>9} {esps_py:>14.3e} {esps_g:>14.3e}")


if __name__ == "__main__":
    main()
