"""Test the ABA-structure M^-1 solve in isolation: qacc = aba_solve(tau) should
equal np.linalg.solve(M, tau). This tests the inertia reduction + the torque
propagation path (pa = U tau/D) -- the path gravity doesn't exercise.
"""
import os, numpy as np, mujoco
HERE="/home/infatoshi/dev/cuda/g1.cu"
m=mujoco.MjModel.from_xml_path(HERE+"/models/g1.xml")
m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER
m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)|int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
NB,NV=m.nbody,m.nv
ref=np.load(HERE+"/bench/ref_traj.npz")
d=mujoco.MjData(m); d.qpos[:]=ref["qpos_init"]; d.qvel[:]=0; mujoco.mj_forward(m,d)
xpos,xipos=d.xpos.copy(),d.xipos.copy(); p=xpos[1].copy(); R1=d.xmat[1].reshape(3,3)

def skew(c): return np.array([[0,-c[2],c[1]],[c[2],0,-c[0]],[-c[1],c[0],0]])
# S and I6 about p
S=np.zeros((NV,6))
for j in range(m.njnt):
    t,b,da=m.jnt_type[j],m.jnt_bodyid[j],m.jnt_dofadr[j]
    if t==0:
        S[da+0,3],S[da+1,4],S[da+2,5]=1,1,1
        for k in range(3): S[da+3+k,:3]=R1[:,k]
    elif t==3:
        Rb=d.xmat[b].reshape(3,3); aw=Rb@m.jnt_axis[j]; anchor=xpos[b]+Rb@m.jnt_pos[j]; r=anchor-p
        S[da,:3]=aw; S[da,3:]=np.cross(r,aw)
I6=np.zeros((NB,6,6))
for b in range(1,NB):
    Rb=d.xmat[b].reshape(3,3); Ria=np.zeros(9); mujoco.mju_quat2Mat(Ria,m.body_iquat[b])
    Ri=Rb@Ria.reshape(3,3); Iw=Ri@np.diag(m.body_inertia[b])@Ri.T
    mass=m.body_mass[b]; c=xipos[b]-p; A=Iw+mass*(np.dot(c,c)*np.eye(3)-np.outer(c,c)); B=mass*skew(c)
    M6=np.zeros((6,6)); M6[:3,:3]=A; M6[:3,3:]=B; M6[3:,:3]=B.T; M6[3:,3:]=mass*np.eye(3); I6[b]=M6


import sys
VAR=sys.argv[1] if len(sys.argv)>1 else "A"
def aba_solve(tau):
    IA=I6.copy(); pAf=np.zeros((NB,6))   # articulated bias from applied torque
    # pass2 leaf->root (hinges)
    for b in range(NB-1,1,-1):
        par=m.body_parentid[b]; dd=m.body_dofadr[b]
        U=IA[b]@S[dd]; D=S[dd]@U
        u=tau[dd]-S[dd]@pAf[b]
        Ia=IA[b]-np.outer(U,U)/D
        pa=pAf[b]+U*(u/D)
        IA[par]+=Ia
        if VAR in ("A","P3"): pAf[par]+=pa
        else: pAf[par]-=pa   # B: flip pa propagation sign
    # pass3
    a=np.zeros((NB,6)); qacc=np.zeros(NV)
    b=1; dd=m.body_dofadr[b]; ap=np.zeros(6)
    U=np.zeros((6,6))
    for l in range(6): U[:,l]=IA[b]@S[dd+l]
    Dm=np.zeros((6,6))
    for k in range(6):
        for l in range(6): Dm[k,l]=S[dd+k]@U[:,l]
    sgn = -1.0 if VAR!="P3" else 1.0
    rhs=np.array([tau[dd+k]-S[dd+k]@pAf[b]+sgn*(U[:,k]@ap) for k in range(6)])
    qddb=np.linalg.solve(Dm,rhs); qacc[dd:dd+6]=qddb
    a[b]=ap+sum(S[dd+k]*qddb[k] for k in range(6))
    for b in range(2,NB):
        par=m.body_parentid[b]; dd=m.body_dofadr[b]; ap=a[par]
        U=IA[b]@S[dd]; D=S[dd]@U
        u=tau[dd]-S[dd]@pAf[b]
        qdd=(u+sgn*(U@ap))/D; qacc[dd]=qdd; a[b]=ap+S[dd]*qdd
    return qacc


M=np.zeros((NV,NV)); mujoco.mj_fullM(m,M,d.qM)
Minv=np.linalg.inv(M)
for k in [0,3,6,7,10,20,34]:
    e=np.zeros(NV); e[k]=1.0
    q=aba_solve(e)
    err=np.abs(q-Minv[:,k])
    print(f"tau=e_{k:2d}: relerr {np.linalg.norm(q-Minv[:,k])/np.linalg.norm(Minv[:,k]):.2e}  maxerr {err.max():.3e} at dof {err.argmax()}")

# reconstruct M from aba_solve, compare to true M
Minv_aba=np.column_stack([aba_solve(np.eye(NV)[:,k]) for k in range(NV)])
M_aba=np.linalg.inv(Minv_aba)
print("M_aba vs Mtrue relerr:", np.linalg.norm(M_aba-M)/np.linalg.norm(M), "max", np.abs(M_aba-M).max())
print("M_aba asym:", np.abs(M_aba-M_aba.T).max())
diff=np.abs(M_aba-M); i,j=np.unravel_index(diff.argmax(),diff.shape)
print(f"worst M[{i},{j}]: aba {M_aba[i,j]:.4f} true {M[i,j]:.4f}")
# is the error only in off-diagonal base-hinge coupling?
print("base-base block relerr:", np.linalg.norm(M_aba[:6,:6]-M[:6,:6])/np.linalg.norm(M[:6,:6]))
print("hinge-hinge block relerr:", np.linalg.norm(M_aba[6:,6:]-M[6:,6:])/np.linalg.norm(M[6:,6:]))
print("base-hinge block relerr:", np.linalg.norm(M_aba[:6,6:]-M[:6,6:])/np.linalg.norm(M[:6,6:]))
