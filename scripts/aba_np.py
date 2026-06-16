"""fp64 numpy ABA debug oracle. Reuses the validated S / I6 / cvel setup from
ref_dynamics_np (about pelvis origin, world axes), runs ABA, compares qacc to
MuJoCo and to the dense solve M^-1(-bias). Bisects the ABA bug without fp32/CUDA noise.
"""
import os, numpy as np, mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "models", "g1.xml"))
m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
NB, NV = m.nbody, m.nv
ref = np.load(os.path.join(HERE, "bench", "ref_traj.npz"))
d = mujoco.MjData(m); d.qpos[:] = ref["qpos_init"]; d.qvel[:] = ref["qvel_init"]
mujoco.mj_forward(m, d)
xpos, xipos = d.xpos.copy(), d.xipos.copy()
p = xpos[1].copy(); R1 = d.xmat[1].reshape(3, 3)


def skew(c): return np.array([[0,-c[2],c[1]],[c[2],0,-c[0]],[-c[1],c[0],0]])
def crossm(a,b):
    w1,v1,w2,v2=a[:3],a[3:],b[:3],b[3:]
    return np.concatenate([np.cross(w1,w2), np.cross(w1,v2)+np.cross(v1,w2)])
def crossf(a,f):
    w1,v1,n,f0=a[:3],a[3:],f[:3],f[3:]
    return np.concatenate([np.cross(w1,n)+np.cross(v1,f0), np.cross(w1,f0)])

# S (NV x 6) about p, world axes
S = np.zeros((NV,6))
for j in range(m.njnt):
    t,b,da=m.jnt_type[j],m.jnt_bodyid[j],m.jnt_dofadr[j]
    if t==0:
        S[da+0,3],S[da+1,4],S[da+2,5]=1,1,1
        for k in range(3): S[da+3+k,:3]=R1[:,k]
    elif t==3:
        Rb=d.xmat[b].reshape(3,3); aw=Rb@m.jnt_axis[j]
        anchor=xpos[b]+Rb@m.jnt_pos[j]; r=anchor-p
        S[da,:3]=aw; S[da,3:]=np.cross(r,aw)

# body spatial inertia 6x6 about p
I6=np.zeros((NB,6,6))
for b in range(1,NB):
    Rb=d.xmat[b].reshape(3,3); Ria=np.zeros(9); mujoco.mju_quat2Mat(Ria,m.body_iquat[b])
    Ri=Rb@Ria.reshape(3,3); Iw=Ri@np.diag(m.body_inertia[b])@Ri.T
    mass=m.body_mass[b]; c=xipos[b]-p; A=Iw+mass*(np.dot(c,c)*np.eye(3)-np.outer(c,c)); B=mass*skew(c)
    M6=np.zeros((6,6)); M6[:3,:3]=A; M6[:3,3:]=B; M6[3:,:3]=B.T; M6[3:,3:]=mass*np.eye(3); I6[b]=M6

# pass1: velocities + bias accel c (free-joint-aware cdof_dot)
vel=np.zeros((NB,6)); cb=np.zeros((NB,6)); import sys as _s; qv=ref["qvel_init"]*(float(_s.argv[1]) if len(_s.argv)>1 else 1.0)
if len(_s.argv)>5 and _s.argv[5]!="all": qv=np.zeros(NV); qv[int(_s.argv[5])]=1.0   # isolate one dof
for b in range(1,NB):
    par=m.body_parentid[b]; w=vel[par].copy(); c6=np.zeros(6)
    da,dn=m.body_dofadr[b],m.body_dofnum[b]
    def addsd(a,wp):
        sd=crossm(wp,S[a]); return sd*qv[a]
    if dn==6:
        for a in range(da,da+3): c6+=addsd(a,w); w=w+S[a]*qv[a]
        wr=w.copy()
        for a in range(da+3,da+6): c6+=addsd(a,wr)
        for a in range(da+3,da+6): w=w+S[a]*qv[a]
    else:
        for a in range(da,da+dn): c6+=addsd(a,w); w=w+S[a]*qv[a]
    vel[b]=w; cb[b]=c6
pA=np.zeros((NB,6)); IA=I6.copy()
for b in range(1,NB): pA[b]=crossf(vel[b], I6[b]@vel[b])
if len(_s.argv)>2 and _s.argv[2]=="1": cb[:]=0.0   # zero ALL bias accel
if len(_s.argv)>3 and _s.argv[3]=="1": pA[:]=0.0   # zero velocity-product force

# pass2: leaf->root (hinges, b>=2)
Ustore={}; Dstore={}
for b in range(NB-1,1,-1):
    par=m.body_parentid[b]; dd=m.body_dofadr[b]
    U=IA[b]@S[dd]; D=S[dd]@U + m.dof_armature[dd]; u=-S[dd]@pA[b]
    Ia=IA[b]-np.outer(U,U)/D
    PV=_s.argv[6] if len(_s.argv)>6 else "A"
    if PV=="A":   pa=pA[b]+Ia@cb[b]+U*(u/D)        # Featherstone
    elif PV=="B": pa=pA[b]+Ia@cb[b]-U*(u/D)        # flip u sign
    elif PV=="C": pa=pA[b]+Ia@cb[b]                # drop u term
    elif PV=="D": pa=pA[b]+IA[b]@cb[b]+U*(u/D)     # full IA for c
    IA[par]+=Ia; pA[par]+=pa

# pass3
qacc=np.zeros(NV)
NOGRAV = len(_s.argv)>4 and _s.argv[4]=="1"
a0=np.zeros(6) if NOGRAV else np.array([0,0,0,-m.opt.gravity[0],-m.opt.gravity[1],-m.opt.gravity[2]])
a=np.zeros((NB,6))
# base body 1 (free)
b=1; dd=m.body_dofadr[b]; ap=a0+cb[b]
U=np.zeros((6,6))
for l in range(6): U[:,l]=IA[b]@S[dd+l]
D=np.zeros((6,6))
for k in range(6):
    for l in range(6): D[k,l]=S[dd+k]@U[:,l] + (m.dof_armature[dd+k] if k==l else 0.0)
rhs=np.array([-S[dd+k]@pA[b]-U[:,k]@ap for k in range(6)])
qddb=np.linalg.solve(D,rhs); qacc[dd:dd+6]=qddb
a[b]=ap+sum(S[dd+k]*qddb[k] for k in range(6))
for b in range(2,NB):
    par=m.body_parentid[b]; dd=m.body_dofadr[b]; ap=a[par]+cb[b]
    U=IA[b]@S[dd]; D=S[dd]@U + m.dof_armature[dd]; u=-S[dd]@pA[b]
    qdd=(u-U@ap)/D; qacc[dd]=qdd; a[b]=ap+S[dd]*qdd
a1_aba=a[1].copy(); pA1=pA[1].copy()

# sanity: straight RNE with the SAME primitives (vel, cb, pA) -> bias; should match MuJoCo
arne=np.zeros((NB,6)); arne[0]=np.array([0,0,0,-m.opt.gravity[0],-m.opt.gravity[1],-m.opt.gravity[2]])
for b in range(1,NB): arne[b]=arne[m.body_parentid[b]]+cb[b]
frne=np.zeros((NB,6))
for b in range(1,NB): frne[b]=I6[b]@arne[b]+pA[b]
bias_rne=np.zeros(NV)
for b in range(NB-1,0,-1):
    dd=m.body_dofadr[b]
    for a in range(dd,dd+m.body_dofnum[b]): bias_rne[a]=S[a]@frne[b]
    frne[m.body_parentid[b]]+=frne[b]

# compare (recompute MuJoCo targets at the SAME qv used by ABA)
d.qvel[:]=qv; mujoco.mj_forward(m,d)
M=np.zeros((NV,NV)); mujoco.mj_fullM(m,M,d.qM)
bias_full=d.qfrc_bias.copy()
d0=mujoco.MjData(m); d0.qpos[:]=ref["qpos_init"]; d0.qvel[:]=0; mujoco.mj_forward(m,d0)
bias_grav=d0.qfrc_bias.copy()
bias_target = (bias_full-bias_grav) if NOGRAV else bias_full   # coriolis-only if gravity off
qacc_dense=np.linalg.solve(M,-bias_target)
print("RNE-here bias vs MuJoCo bias: relerr", np.linalg.norm(bias_rne-bias_full)/np.linalg.norm(bias_full), " max", np.abs(bias_rne-bias_full).max())
print("ABA qacc vs MuJoCo qacc : relerr", np.linalg.norm(qacc-d.qacc)/np.linalg.norm(d.qacc), " max", np.abs(qacc-d.qacc).max())
print("ABA qacc vs dense solve : relerr", np.linalg.norm(qacc-qacc_dense)/np.linalg.norm(qacc_dense), " max", np.abs(qacc-qacc_dense).max())
e=np.abs(qacc-d.qacc); print("base[0:6] err", np.round(e[:6],4)); print("hinge max err", e[6:].max(),"at",6+e[6:].argmax())
# verify velocities vs MuJoCo cvel (angular part is reference-independent)
ang_err = np.abs(vel[:,:3] - d.cvel[:,:3]).max()
print(f"vel angular vs MuJoCo cvel: max err {ang_err:.2e}")
# verify bias accel c vs MuJoCo (sum of cdof_dot*qvel per body)
cb_mj = np.zeros((NB,6))
for b in range(1,NB):
    for a in range(m.body_dofadr[b], m.body_dofadr[b]+m.body_dofnum[b]):
        cb_mj[b]+=d.cdof_dot[a]*qv[a]
print(f"c angular vs MuJoCo: {np.abs(cb[:,:3]-cb_mj[:,:3]).max():.2e}  c linear vs MuJoCo: {np.abs(cb[:,3:]-cb_mj[:,3:]).max():.2e}")
# base spatial accel: ABA vs truth (from dense qacc)
a1_true = a0 + cb[1] + sum(S[kk]*qacc_dense[kk] for kk in range(6))
print("a[1] ABA :", np.round(a1_aba,4))
print("a[1] true:", np.round(a1_true,4))
print("pA[1] (accumulated base velocity force):", np.round(pA1,4))
resid = M@qacc + bias_target   # should be ~0 if ABA satisfies true dynamics
print("force-balance resid per dof (|.|>1e-3):")
for i in range(NV):
    if abs(resid[i])>1e-3:
        # which body owns dof i?
        bod=[b for b in range(1,NB) if m.body_dofadr[b]<=i<m.body_dofadr[b]+m.body_dofnum[b]][0]
        print(f"   dof {i:2d} body {bod} ({m.body(bod).name}): resid {resid[i]:+.4f}")
