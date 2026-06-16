"""fp64 numpy port of the CUDA Featherstone CRBA/RNE (about pelvis origin, world axes).

Debugging oracle: isolates formula correctness from fp32 noise by reproducing the
exact same algorithm as src/dynamics.cu and diffing against MuJoCo's mj_fullM /
qfrc_bias. If this matches MuJoCo, any CUDA gap is pure fp32.
"""
import os, numpy as np, mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "models", "g1.xml"))
m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
NB, NV = m.nbody, m.nv

ref = np.load(os.path.join(HERE, "bench", "ref_traj.npz"))
qpos = ref["qpos_init"].astype(np.float64)
qvel = ref["qvel_init"].astype(np.float64)

d = mujoco.MjData(m)
d.qpos[:] = qpos; d.qvel[:] = qvel
mujoco.mj_forward(m, d)
xpos = d.xpos.copy(); xquat = d.xquat.copy(); xipos = d.xipos.copy()


def skew(c):
    return np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])


def crossm(a, b):  # spatial motion x motion, [ang;lin]
    w1, v1, w2, v2 = a[:3], a[3:], b[:3], b[3:]
    return np.concatenate([np.cross(w1, w2), np.cross(w1, v2) + np.cross(v1, w2)])


def crossf(a, f):  # spatial motion x force
    w1, v1, n, f0 = a[:3], a[3:], f[:3], f[3:]
    return np.concatenate([np.cross(w1, n) + np.cross(v1, f0), np.cross(w1, f0)])


import sys
REF = sys.argv[1] if len(sys.argv) > 1 else "pelvis"
if REF == "pelvis":
    p = xpos[1].copy()
elif REF == "com":
    p = d.subtree_com[0].copy()
elif REF == "origin":
    p = np.zeros(3)
print(f"reference point = {REF} {np.round(p,3)}")
R1 = d.xmat[1].reshape(3, 3)
pelvis_org = xpos[1].copy()

# ---- motion axes S (NV x 6), about p, world axes ----
S = np.zeros((NV, 6))
for j in range(m.njnt):
    t, b, dadr = m.jnt_type[j], m.jnt_bodyid[j], m.jnt_dofadr[j]
    if t == 0:
        S[dadr+0, 3], S[dadr+1, 4], S[dadr+2, 5] = 1, 1, 1
        for k in range(3):
            ax = R1[:, k]
            S[dadr+3+k, :3] = ax
            S[dadr+3+k, 3:] = np.cross(pelvis_org - p, ax)  # moment about ref p
    elif t == 3:
        Rb = d.xmat[b].reshape(3, 3)
        aw = Rb @ m.jnt_axis[j]
        anchor = xpos[b] + Rb @ m.jnt_pos[j]
        r = anchor - p
        S[dadr, :3] = aw
        S[dadr, 3:] = np.cross(r, aw)

# ---- spatial inertia about p (6x6) ----
I6 = np.zeros((NB, 6, 6))
for b in range(1, NB):
    Rb = d.xmat[b].reshape(3, 3)
    Ri_arr = np.zeros(9); mujoco.mju_quat2Mat(Ri_arr, m.body_iquat[b])
    Ri = Rb @ Ri_arr.reshape(3, 3)
    Iw = Ri @ np.diag(m.body_inertia[b]) @ Ri.T
    mass = m.body_mass[b]
    c = xipos[b] - p
    A = Iw + mass*(np.dot(c, c)*np.eye(3) - np.outer(c, c))
    B = mass*skew(c)
    M6 = np.zeros((6, 6))
    M6[:3, :3] = A; M6[:3, 3:] = B; M6[3:, :3] = B.T; M6[3:, 3:] = mass*np.eye(3)
    I6[b] = M6

# ---- CRBA ----
Icp = I6.copy()
for b in range(NB-1, 0, -1):
    par = m.body_parentid[b]
    if par >= 1:
        Icp[par] += Icp[b]
M = np.zeros((NV, NV))
for b in range(1, NB):
    da, dn = m.body_dofadr[b], m.body_dofnum[b]
    for a in range(da, da+dn):
        F = Icp[b] @ S[a]
        j = b
        while j >= 1:
            dj, djn = m.body_dofadr[j], m.body_dofnum[j]
            for c in range(dj, dj+djn):
                M[a, c] = M[c, a] = S[c] @ F
            j = m.body_parentid[j]
for a in range(NV):
    M[a, a] += m.dof_armature[a]

# ---- RNE bias ----
cvel = np.zeros((NB, 6))
Sdot = np.zeros((NV, 6))
for b in range(1, NB):
    par = m.body_parentid[b]
    vv = cvel[par].copy()
    da, dn = m.body_dofadr[b], m.body_dofnum[b]
    if dn == 6:  # free joint: rotational dofs' cdof_dot use post-translation partial
        for a in range(da, da+3):           # translation
            Sdot[a] = crossm(vv, S[a]); vv = vv + S[a]*qvel[a]
        vrot = vv.copy()
        for a in range(da+3, da+6):          # rotation: all from same partial (no self-accum)
            Sdot[a] = crossm(vrot, S[a])
        for a in range(da+3, da+6):
            vv = vv + S[a]*qvel[a]
    else:
        for a in range(da, da+dn):
            Sdot[a] = crossm(vv, S[a]); vv = vv + S[a]*qvel[a]
    cvel[b] = vv
cacc = np.zeros((NB, 6))
cacc[0, 3:] = -m.opt.gravity
for b in range(1, NB):
    par = m.body_parentid[b]
    aa = cacc[par].copy()
    da, dn = m.body_dofadr[b], m.body_dofnum[b]
    for a in range(da, da+dn):
        aa = aa + Sdot[a]*qvel[a]
    cacc[b] = aa
cfrc = np.zeros((NB, 6))
for b in range(1, NB):
    cfrc[b] = I6[b] @ cacc[b] + crossf(cvel[b], I6[b] @ cvel[b])
bias = np.zeros(NV)
for b in range(NB-1, 0, -1):
    da, dn = m.body_dofadr[b], m.body_dofnum[b]
    for a in range(da, da+dn):
        bias[a] = S[a] @ cfrc[b]
    par = m.body_parentid[b]
    if par >= 1:
        cfrc[par] += cfrc[b]

# ---- compare intermediates to MuJoCo (cdof0/cvel0 are about subtree_com) ----
cdof_mj = ref["cdof0"]   # (NV,6) [ang,lin]
cvel_mj = ref["cvel0"]   # (NB,6) [ang,lin]
print(f"  cdof ang err: {np.abs(S[:,:3]-cdof_mj[:,:3]).max():.3e}  cdof lin err: {np.abs(S[:,3:]-cdof_mj[:,3:]).max():.3e}")
print(f"  cvel ang err: {np.abs(cvel[:,:3]-cvel_mj[:,:3]).max():.3e}  cvel lin err: {np.abs(cvel[:,3:]-cvel_mj[:,3:]).max():.3e}")
# show a hinge cdof row (dof 6) both
print("  cdof[6] np:", np.round(S[6],4), " mj:", np.round(cdof_mj[6],4))
print("  cvel[1] np:", np.round(cvel[1],4), " mj:", np.round(cvel_mj[1],4))
cdofdot_mj = d.cdof_dot  # (NV,6), filled by mj_forward
print(f"  cdof_dot ang err: {np.abs(Sdot[:,:3]-cdofdot_mj[:,:3]).max():.3e}  lin err: {np.abs(Sdot[:,3:]-cdofdot_mj[:,3:]).max():.3e}")
kk = np.abs(Sdot-cdofdot_mj).max(axis=1).argmax()
print(f"  worst cdof_dot dof {kk} np:", np.round(Sdot[kk],4), " mj:", np.round(cdofdot_mj[kk],4))

# ---- compare to MuJoCo ----
Mmj = np.zeros((NV, NV)); mujoco.mj_fullM(m, Mmj, d.qM)
print(f"M   rel err: {np.linalg.norm(M-Mmj)/np.linalg.norm(Mmj):.3e}  max abs: {np.abs(M-Mmj).max():.3e}")
print(f"bias rel err: {np.linalg.norm(bias-d.qfrc_bias)/np.linalg.norm(d.qfrc_bias):.3e}  max abs: {np.abs(bias-d.qfrc_bias).max():.3e}")
k = np.abs(bias-d.qfrc_bias).argmax()
print(f"  worst bias[{k}] np={bias[k]:.6f} mj={d.qfrc_bias[k]:.6f}")
print("  bias[:6] np:", np.round(bias[:6], 5))
print("  bias[:6] mj:", np.round(d.qfrc_bias[:6], 5))
