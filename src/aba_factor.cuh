// ABA factor-once + O(nbody) M^-1 apply (MuJoCo mj_solveM-equivalent), fp32.
//
// Featherstone's articulated inertias ARE a factorization of the mass matrix M.
// aba.cuh folds bias/gravity/velocity into a single qacc = M^-1(tau - bias) sweep.
// Here we SPLIT that: aba_factorize() runs the leaf->root pass to build and STORE the
// per-body factor (U = I^A S, invD = 1/(S^T U + armature), plus the 6x6 base factor),
// then aba_solveM() applies M^-1 to an arbitrary generalized force vector x in O(nbody),
// reusing the stored factor. velocity=0, gravity=0 in the apply, so pass1 is trivially
// zero; only the pA back-propagation (driven by x) and the pass3 forward sweep depend on x.
//
// This replaces the dense CRBA + O(nv^3) Cholesky in the contact solver: M^-1 J^T is
// nefc cheap solveM applies, and qacc_smooth = solveM(tau - bias).
//
// Derived from src/aba.cuh's math (DO NOT MODIFY aba.cuh). Same common-frame formulation
// (world axes about the pelvis origin), so results land in MuJoCo's qvel basis.
#pragma once
#include "dynamics.cuh"
#include "aba.cuh"   // mv6/dot6/symidx/symmv6/crossf6/chol6_solve, ABA_SCR layout helpers

// ---- stored factor for one world ----
// S (NV*6) motion axes, shared with the contact-J build (caller already has it).
// Hinge bodies (b>=2): U[b*6..] (6 floats), invD[b] (scalar).
// Base (b=1, free joint, 6 dofs): U6[36] (column-major-ish: U6[i*6+l]=U_l[i]),
//   Dfac6[36] = Cholesky factor of the 6x6 D block.
// Scratch holds the propagated articulated inertia IA during factorize (transient).
struct AbaFactor {
    float U[G1_NBODY*6];   // hinge U columns (base slot unused)
    float invD[G1_NBODY];  // hinge 1/D (base slot unused)
    float U6[36];          // base: 6 U columns, U6[i*6+l] = (I^A S_l)[i]
    float Dfac6[36];       // base: 6x6 Cholesky factor of D
};

#define ABA_FAC_SCR (G1_NBODY*21)   // scratch floats for IA during factorize

// Build the factor. `S` must already be filled (NV*6, world axes about pelvis) by the
// caller -- it is identical to aba.cuh's S and to the contact path's S. `scr` = ABA_FAC_SCR
// floats (IA workspace). xpos/xquat/xipos from forward_kinematics.
//
// `Bdiag` (optional, NV floats or nullptr): per-dof diagonal damping ALREADY scaled by dt
// (i.e. dt*(kd + dof_damping)) to fold IMPLICITLY into the articulated-inertia diagonal D.
// This makes the M^-1 apply solve (M + dt*B) instead of M -- the MuJoCo implicitfast
// semi-implicit-in-velocity update for stiff velocity-dependent (PD kd / passive damping)
// forces. Bdiag=nullptr recovers the plain M^-1 factor (smooth callers unchanged).
__device__ void aba_factorize_damped(const float* xquat, const float* xipos,
                                     const float* xpos, const float* S,
                                     const float* Bdiag,
                                     float* scr, AbaFactor* fac) {
    const int NB = G1_NBODY;
    V3 p = {xpos[3], xpos[4], xpos[5]};

    // ---- per-body spatial inertia (6x6 about p), symmetric-packed -> init IA ----
    float* IA = scr;   // [NB*21]
    for (int b=1;b<NB;++b){
        Q4 qb={xquat[4*b],xquat[4*b+1],xquat[4*b+2],xquat[4*b+3]};
        Q4 qi={body_iquat[4*b],body_iquat[4*b+1],body_iquat[4*b+2],body_iquat[4*b+3]};
        M3 Ri=quat2mat(qmul(qb,qi));
        V3 di={body_inertia[3*b],body_inertia[3*b+1],body_inertia[3*b+2]};
        M3 Iw=world_inertia(Ri,di);
        float m=body_mass[b];
        V3 c={xipos[3*b]-p.x,xipos[3*b+1]-p.y,xipos[3*b+2]-p.z};
        float cc=dot3(c,c);
        float A[9]; for(int i=0;i<9;++i)A[i]=Iw.m[i];
        A[0]+=m*(cc-c.x*c.x);A[1]+=m*(-c.x*c.y);A[2]+=m*(-c.x*c.z);
        A[3]+=m*(-c.y*c.x);A[4]+=m*(cc-c.y*c.y);A[5]+=m*(-c.y*c.z);
        A[6]+=m*(-c.z*c.x);A[7]+=m*(-c.z*c.y);A[8]+=m*(cc-c.z*c.z);
        float Bx[9]={0,-c.z,c.y, c.z,0,-c.x, -c.y,c.x,0};
        float M6[36];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            M6[6*i+j]=A[3*i+j]; M6[6*i+(j+3)]=m*Bx[3*i+j];
            M6[6*(i+3)+j]=-m*Bx[3*i+j]; M6[6*(i+3)+(j+3)]=(i==j)?m:0.f;
        }
        float* X=&IA[b*21];
        for(int i=0;i<6;++i)for(int j=i;j<6;++j) X[symidx(i,j)]=M6[i*6+j];
    }

    // ---- leaf->root: propagate articulated inertia, store U/invD (hinges) ----
    for (int b=NB-1;b>=2;--b){
        int par=body_parentid[b], d=body_dofadr[b];   // hinge: single dof
        float U[6]; symmv6(&IA[b*21], &S[d*6], U);
        float D=dot6(&S[d*6], U) + dof_armature[d];
        if (Bdiag) D += Bdiag[d];   // implicit damping (dt*(kd+dof_damping)) folded into D
        float invD=1.f/D;
        for(int k=0;k<6;++k) fac->U[b*6+k]=U[k];
        fac->invD[b]=invD;
        // Ia = IA - U U^T / D, accumulate into parent (only the inertia part; pa needs x)
        for(int i=0;i<6;++i)for(int j=i;j<6;++j)
            IA[par*21+symidx(i,j)] += IA[b*21+symidx(i,j)] - U[i]*U[j]*invD;
    }

    // ---- base (body 1, free joint): build U6 columns + 6x6 D, store Cholesky factor ----
    {
        int b=1, d=body_dofadr[b];
        float Dm[36];
        for(int l=0;l<6;++l){
            float Ul[6]; symmv6(&IA[b*21], &S[(d+l)*6], Ul);
            for(int i=0;i<6;++i) fac->U6[i*6+l]=Ul[i];
        }
        for(int k=0;k<6;++k)for(int l=0;l<6;++l){
            float s=0; for(int i=0;i<6;++i) s+=S[(d+k)*6+i]*fac->U6[i*6+l];
            Dm[k*6+l]=s + (k==l?(dof_armature[d+k]+(Bdiag?Bdiag[d+k]:0.f)):0.f);
        }
        // in-place Cholesky factor (lower) of Dm -> Dfac6
        for(int i=0;i<6;++i)for(int j=0;j<=i;++j){
            float s=Dm[i*6+j]; for(int kk=0;kk<j;++kk) s-=Dm[i*6+kk]*Dm[j*6+kk];
            if(i==j) Dm[i*6+i]=sqrtf(s); else Dm[i*6+j]=s/Dm[j*6+j];
        }
        for(int i=0;i<36;++i) fac->Dfac6[i]=Dm[i];
    }
}

// Plain M^-1 factor (no damping) -- unchanged interface for the smooth/contact callers.
__device__ __forceinline__ void aba_factorize(const float* xquat, const float* xipos,
                                              const float* xpos, const float* S,
                                              float* scr, AbaFactor* fac) {
    aba_factorize_damped(xquat, xipos, xpos, S, nullptr, scr, fac);
}

// Apply M^-1 to generalized force x (NV): solve M y = x, O(nbody). velocity=0, gravity=0,
// bias=0 -> only x drives pA (leaf->root) and the acceleration sweep (root->leaf). Reuses
// the stored U/invD/U6/Dfac6. `scr` = G1_NBODY*6*2 floats (pA + a6 workspace).
__device__ void aba_solveM(const AbaFactor* fac, const float* S,
                           const float* x, float* scr, float* y) {
    const int NB = G1_NBODY;
    float* pA = scr;                 // [NB*6]
    float* a6 = scr + G1_NBODY*6;    // [NB*6]
    for (int i=0;i<NB*6;++i) pA[i]=0.f;

    // ---- leaf->root: pa = U*(u*invD), u = x[d] - S^T pA ----
    for (int b=NB-1;b>=2;--b){
        int par=body_parentid[b], d=body_dofadr[b];
        const float* U=&fac->U[b*6];
        float u = x[d] - dot6(&S[d*6], &pA[b*6]);
        float ud = u * fac->invD[b];
        // pa = pA[b] + U*(u/D)  (the Ia*cb term vanishes since cb=0); accumulate into parent.
        for(int k=0;k<6;++k) pA[par*6+k] += pA[b*6+k] + U[k]*ud;
    }

    // ---- root->leaf accelerations. a0=0 (no gravity). ap = a_parent (c=0). ----
    {
        int b=1, d=body_dofadr[b];
        float ap[6]={0,0,0,0,0,0};   // a_parent(=0) + c(=0)
        float rhs[6];
        for(int k=0;k<6;++k){
            float uk = x[d+k] - dot6(&S[(d+k)*6], &pA[b*6]);
            float UTa=0; for(int i=0;i<6;++i) UTa += fac->U6[i*6+k]*ap[i];
            rhs[k] = uk - UTa;
        }
        // solve Dfac6 (chol L) * Dfac6^T * qdd = rhs
        float qdd[6];
        const float* L=fac->Dfac6;
        for(int i=0;i<6;++i){ float s=rhs[i]; for(int k=0;k<i;++k) s-=L[i*6+k]*qdd[k]; qdd[i]=s/L[i*6+i]; }
        for(int i=5;i>=0;--i){ float s=qdd[i]; for(int k=i+1;k<6;++k) s-=L[k*6+i]*qdd[k]; qdd[i]=s/L[i*6+i]; }
        for(int k=0;k<6;++k) y[d+k]=qdd[k];
        for(int i=0;i<6;++i){ float Sqdd=0; for(int k=0;k<6;++k) Sqdd+=S[(d+k)*6+i]*qdd[k]; a6[b*6+i]=ap[i]+Sqdd; }
    }
    for (int b=2;b<NB;++b){
        int par=body_parentid[b], d=body_dofadr[b];
        float ap[6]; for(int k=0;k<6;++k) ap[k]=a6[par*6+k];   // c=0
        const float* U=&fac->U[b*6];
        float u = x[d] - dot6(&S[d*6], &pA[b*6]);
        float qdd = (u - dot6(U,ap)) * fac->invD[b];
        y[d]=qdd;
        for(int i=0;i<6;++i) a6[b*6+i]=ap[i]+S[d*6+i]*qdd;
    }
}

#define ABA_SOLVEM_SCR (G1_NBODY*6*2)

// Multiply M*x (RNE with qvel=0, gravity=0, qacc=x): y = M x. O(nbody). Recomputes the raw
// (un-propagated) body spatial inertias from kinematics, then forward accel sweep + backward
// force sweep. Used to reconstruct the smooth bias vector (-bias = M * qacc_smooth) so the
// damped factor can be applied to (-bias + tau). `scr` = ABA_MULM_SCR floats.
#define ABA_MULM_SCR (G1_NBODY*21 + 2*G1_NBODY*6)
__device__ void aba_mulM(const float* xquat, const float* xipos, const float* xpos,
                         const float* S, const float* x, float* scr, float* y) {
    const int NB = G1_NBODY;
    V3 p = {xpos[3], xpos[4], xpos[5]};
    float* I  = scr;                    // [NB*21] symmetric body spatial inertia about p
    float* a6 = scr + G1_NBODY*21;      // [NB*6] body spatial acceleration
    float* f6 = scr + G1_NBODY*21 + G1_NBODY*6;  // [NB*6] body spatial force

    for (int b=1;b<NB;++b){
        Q4 qb={xquat[4*b],xquat[4*b+1],xquat[4*b+2],xquat[4*b+3]};
        Q4 qi={body_iquat[4*b],body_iquat[4*b+1],body_iquat[4*b+2],body_iquat[4*b+3]};
        M3 Ri=quat2mat(qmul(qb,qi));
        V3 di={body_inertia[3*b],body_inertia[3*b+1],body_inertia[3*b+2]};
        M3 Iw=world_inertia(Ri,di);
        float m=body_mass[b];
        V3 c={xipos[3*b]-p.x,xipos[3*b+1]-p.y,xipos[3*b+2]-p.z};
        float cc=dot3(c,c);
        float A[9]; for(int i=0;i<9;++i)A[i]=Iw.m[i];
        A[0]+=m*(cc-c.x*c.x);A[1]+=m*(-c.x*c.y);A[2]+=m*(-c.x*c.z);
        A[3]+=m*(-c.y*c.x);A[4]+=m*(cc-c.y*c.y);A[5]+=m*(-c.y*c.z);
        A[6]+=m*(-c.z*c.x);A[7]+=m*(-c.z*c.y);A[8]+=m*(cc-c.z*c.z);
        float Bx[9]={0,-c.z,c.y, c.z,0,-c.x, -c.y,c.x,0};
        float M6[36];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            M6[6*i+j]=A[3*i+j]; M6[6*i+(j+3)]=m*Bx[3*i+j];
            M6[6*(i+3)+j]=-m*Bx[3*i+j]; M6[6*(i+3)+(j+3)]=(i==j)?m:0.f;
        }
        float* X=&I[b*21];
        for(int i=0;i<6;++i)for(int j=i;j<6;++j) X[symidx(i,j)]=M6[i*6+j];
    }

    // ---- forward (root->leaf): spatial accel a[b] = a[par] + S[d]*x[d] (c=0, qvel=0) ----
    {   int b=1, d=body_dofadr[b];
        for(int i=0;i<6;++i){ float s=0; for(int k=0;k<6;++k) s+=S[(d+k)*6+i]*x[d+k]; a6[b*6+i]=s; }
    }
    for (int b=2;b<NB;++b){
        int par=body_parentid[b], d=body_dofadr[b];
        for(int i=0;i<6;++i) a6[b*6+i]=a6[par*6+i]+S[d*6+i]*x[d];
    }
    // ---- backward (leaf->root): f[b] = I[b]*a[b] + sum_children f[child]; tau[d]=S^T f ----
    for (int b=1;b<NB;++b) symmv6(&I[b*21], &a6[b*6], &f6[b*6]);
    for (int b=NB-1;b>=2;--b){
        int par=body_parentid[b], d=body_dofadr[b];
        y[d]=dot6(&S[d*6], &f6[b*6]);
        for(int i=0;i<6;++i) f6[par*6+i]+=f6[b*6+i];
    }
    {   int b=1, d=body_dofadr[b];
        for(int k=0;k<6;++k) y[d+k]=dot6(&S[(d+k)*6], &f6[b*6]);
    }
}
