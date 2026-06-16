// env_qacc.cuh -- full per-substep generalized acceleration for the G1 RL env.
// PURE MOVE out of g1_env.cu (no logic change) so it can be unit-tested in isolation
// (src/test_envqacc.cu) and optimized under a numerical-equivalence gate. Depends on the
// validated dynamics/ABA/contact headers, all included by the translation unit that uses it.
#pragma once
#include "dynamics.cuh"
#include "aba.cuh"
#include "aba_factor.cuh"
#include "aba_solvem_multi.cuh"
#include "aba_bias.cuh"
#include "contact.cuh"

// torque-augmented full qacc with IMPLICIT (MuJoCo implicitfast) velocity-damping.
// The stiff PD velocity feedback (tau = kp*(target-q) - kd*qd) is unstable under explicit
// Euler at kp=500, S>=2. We fold the kd damping derivative implicitly: solve
//   (M + dt*B) qacc = (-bias) + tau,   B[d] = kd_i + dof_damping (diagonal, joint-space)
// matching `(M+dt*B) qvel_new = M*qvel + dt*qfrc`. Steps:
//   1) qacc_smooth_plain = M^-1(-bias)  (aba_qacc)
//   2) reconstruct -bias = M * qacc_smooth_plain  (aba_mulM)
//   3) build DAMPED factor (dt*B folded into D), apply (M+dtB)^-1 to (-bias + tau)
//   4) contacts solved through the damped factor (A = J (M+dtB)^-1 J^T + R) for consistency.
__device__ void env_qacc(const float* qp, const float* qv,
                         const float* xpos, const float* xquat, const float* xipos,
                         const float* tau, const float* Bdt, float* S, float* qacc, int pgs_iters){
    const int NV=G1_NV;
    float qacc_s[G1_NV];

    // smooth force -bias directly via RNE (qacc=0): one O(N) pass, no factorization/solve/mulM.
    // Equivalent to the validated compute_M_bias() bias (MuJoCo-matched); fills S in the same
    // basis the damped factor below consumes. Replaces the old aba_qacc + aba_mulM roundtrip.
    float bias[G1_NV], negbias[G1_NV];
    compute_bias(qp, qv, xpos, xquat, xipos, S, bias);
    for (int i=0;i<NV;++i) negbias[i] = -bias[i];

    AbaFactor fac; float facscr[ABA_FAC_SCR];
    aba_factorize_damped(xquat, xipos, xpos, S, Bdt, facscr, &fac);
    float solvescr[ABA_SOLVEM_SCR];

    // damped smooth+torque qacc: (M+dtB)^-1 ( -bias + tau )
    float rhs[G1_NV];
    for (int i=0;i<NV;++i) rhs[i]=negbias[i]+tau[i];
    aba_solveM(&fac, S, rhs, solvescr, qacc_s);

#ifdef SKIP_CONTACT
    for (int i=0;i<NV;++i) qacc[i]=qacc_s[i]; return;   // ablation: smooth+torque only, no contacts
#endif
    Contact cons[G1_MAX_CONTACT];
    int ncon = detect_foot_contacts(xpos, xquat, cons);
    if (ncon==0){ for(int i=0;i<NV;++i) qacc[i]=qacc_s[i]; return; }

    V3 p={xpos[3],xpos[4],xpos[5]};
    int nefc = 4*ncon;

    float invw0[G1_MAX_CONTACT];
    { int idx=0;
      for (int s=0;s<G1_NFOOT_SPHERE;++s){ int b=g1_foot_body[s];
        Q4 qb={xquat[4*b],xquat[4*b+1],xquat[4*b+2],xquat[4*b+3]};
        V3 lp={g1_foot_lpos[3*s],g1_foot_lpos[3*s+1],g1_foot_lpos[3*s+2]};
        V3 c=vadd((V3){xpos[3*b],xpos[3*b+1],xpos[3*b+2]}, qrot(qb,lp));
        if (c.z-G1_FOOT_RADIUS<0.f) invw0[idx++]=g1_foot_body_invweight0[s]; } }

    // R/aref coefficients (formerly build_R_aref) -- inlined so aref's velocity term is taken from
    // the 3-basis directly, eliminating the dense J array (G1_MAX_EFC*G1_NV = 1120 floats).
    const float dmaxRA=G1_FOOT_SOLIMP[1];
    const float bcoef=2.f/(dmaxRA*G1_FOOT_SOLREF[0]);
    const float kcoef=1.f/(dmaxRA*dmaxRA*G1_FOOT_SOLREF[0]*G1_FOOT_SOLREF[0]*G1_FOOT_SOLREF[1]*G1_FOOT_SOLREF[1]);
    const float pyrRA=(1.f+G1_FOOT_MU*G1_FOOT_MU)*(2.f*G1_FOOT_MU*G1_FOOT_MU/G1_FOOT_IMPRATIO);

    float R[G1_MAX_EFC], aref[G1_MAX_EFC];

    // --- SPARSITY: each contact's J rows are nonzero ONLY on the ancestor dofs of the foot
    // body (free base 0..5 + that leg's hinge chain) -- ~12 of 35 dofs. Build the per-contact
    // dof list by the same world->contact_body parent walk build_contact_jac uses (metadata
    // only). free base (6) + one leg hinge chain (<=10) bounds the count; 16 is safe.
    const int CDS=16;
    int cdof[G1_MAX_CONTACT*CDS];     // contact c's nonzero dofs, packed (stride CDS)
    int cnd[G1_MAX_CONTACT];          // count per contact
    for (int c=0;c<ncon;++c){
        int n=0;
        for (int b=cons[c].body; b>=1; b=body_parentid[b]){
            int da=body_dofadr[b], dn=body_dofnum[b];
            for (int a=da;a<da+dn;++a) cdof[c*CDS + n++] = a;
        }
        cnd[c]=n;
    }

    // --- 3-BASIS REFORMULATION: the 4 pyramidal rows of a contact (n+/-mu*t1, n+/-mu*t2) span a
    // 3-dim space with basis e0=n, e1=mu*t1, e2=mu*t2, so row_k = d_k . [e0,e1,e2] with coeff
    //   d0=(1,1,0) d1=(1,-1,0) d2=(1,0,1) d3=(1,0,-1).
    // We never materialize J: the 3 basis rows come DIRECTLY from the point Jacobian column
    // jc=point_jac_col, since (MuJoCo frame) e0=jc.z, e1=mu*jc.y, e2=-mu*jc.x. Store the 3 sparse
    // basis rows e_{c,a} (over c's ancestor dofs) and the 3 dense back-solves m_{c,a}=M^-1 e_{c,a}.
    // The 4x4 pyramid A-block expands from the 3x3 Gram G[a][b]=e_{c,a}.m_{c',b}; the final
    // contact_qfrc + aba_solveM vanish (dq = sum_c sum_a fb_c[a]*m_{c,a} reuses the back-solves).
    // Eb[c] = 3 sparse basis rows laid out [3][CDS] (stride 3*CDS). Mm[c] = 3 dense M^-1 e (stride 3*NV).
    const int EBS=3*CDS;
    float Eb[G1_MAX_CONTACT*EBS];        // basis rows over ancestor dofs (sparse, packed)
    float Mm[G1_MAX_CONTACT*3*G1_NV];    // M^-1 basis (dense)
    float bb[G1_MAX_CONTACT*3];          // e_{c,a} . qacc_smooth
    const float muRA=G1_FOOT_MU;
    // PASS 1: build the 3 sparse basis rows (Eb), bb=e.qacc_smooth, and R/aref for every contact.
    for (int c=0;c<ncon;++c){
        V3 pt={cons[c].pt[0],cons[c].pt[1],cons[c].pt[2]};
        const int* dc=&cdof[c*CDS]; const int nc=cnd[c];
        float* eb=&Eb[c*EBS]; float* bc=&bb[c*3];
        // basis rows direct from point jacobian; accumulate bb=e.qacc_smooth and bv=e.qvel (for aref).
        float b0=0.f,b1=0.f,b2=0.f, v0=0.f,v1=0.f,v2=0.f;
        for(int k=0;k<nc;++k){ int a=dc[k];
            float jc[3]; point_jac_col(S, a, pt, p, jc);
            float e0=jc[2];          // n     = jc.z
            float e1=muRA*jc[1];     // mu*t1 = mu*jc.y
            float e2=-muRA*jc[0];    // mu*t2 = -mu*jc.x
            eb[0*CDS+k]=e0; eb[1*CDS+k]=e1; eb[2*CDS+k]=e2;
            float qa=qacc_s[a]; b0+=e0*qa; b1+=e1*qa; b2+=e2*qa;
            float qd=qv[a];     v0+=e0*qd; v1+=e1*qd; v2+=e2*qd; }
        bc[0]=b0; bc[1]=b1; bc[2]=b2;
        // R/aref for this contact's 4 rows (R shared; aref row = -bcoef*(d_k.bv) - kcoef*imp*pen).
        float pen=cons[c].pen, imp=foot_impedance(pen);
        float Rc=invw0[c]*pyrRA*(1.f-imp)/imp;
        float kip=kcoef*imp*pen;
        for(int r=0;r<4;++r){ int idx=4*c+r; R[idx]=Rc;
            float dkv; switch(r){ case 0:dkv=v0+v1; break; case 1:dkv=v0-v1; break;
                                  case 2:dkv=v0+v2; break; default:dkv=v0-v2; }
            aref[idx]=-bcoef*dkv - kip; }
    }
    // PASS 2: M^-1 basis back-solves, BATCHED 2 contacts per factor traversal (K=6) to amortize
    // the local-memory factor-read traffic (the contact-solve hot spot). Mm lays contacts c,c+1
    // contiguously (6*NV) and the multi-solve's column-major out[col*NV+d] matches Mm exactly, so
    // one K=6 solve fills both. Odd last contact pads cols 3-5 with zeros (M^-1 0 = 0, harmless).
    { float basis[6*G1_NV]; float msolvescr[ABA_SOLVEM_MULTI_SCR(6)];
      for (int c=0;c<ncon;c+=2){
        for(int i=0;i<6*NV;++i) basis[i]=0.f;
        int ng = (c+1<ncon)?2:1;
        for(int g=0;g<ng;++g){ int cc=c+g; const int* dc=&cdof[cc*CDS]; int nc=cnd[cc]; const float* eb=&Eb[cc*EBS];
            for(int aa=0;aa<3;++aa){ const float* er=&eb[aa*CDS]; int col=g*3+aa;
                for(int k=0;k<nc;++k) basis[col*NV + dc[k]]=er[k]; } }
        aba_solveM_multi<6>(&fac, S, basis, msolvescr, &Mm[c*3*NV]);
      } }

    // d_k coefficient vectors in basis [e0,e1,e2]: d0=(1,1,0) d1=(1,-1,0) d2=(1,0,1) d3=(1,0,-1).
    // Stored as DC[k][a] so A[(c,k),(c',k')] = sum_{a,b} DC[k][a]*G[a][b]*DC[k'][b].
    const float DC[4][3] = {{1,1,0},{1,-1,0},{1,0,1},{1,0,-1}};

    // A = J M^-1 J^T + diag(R), SYMMETRIC. Build per contact PAIR: 3x3 Gram G[a][b]=e_{c,a}.m_{c',b}
    // (9 sparse dots over c's ancestor dofs), then expand to the 4x4 pyramid block via DC. Only
    // upper triangle (c'>=c); the diagonal block uses the symmetric G (G[a][b]==G[b][a] up to
    // numerics, but we compute the full 9 to match the reference's per-entry arithmetic exactly).
    // A kept DENSE so solve_pgs keeps fast contiguous row access. R added on the 4x4 diagonal only.
    float A[G1_MAX_EFC*G1_MAX_EFC], bvec[G1_MAX_EFC];
    for (int c=0;c<ncon;++c){
        const int* dc=&cdof[c*CDS]; const int nc=cnd[c]; const float* eb=&Eb[c*EBS];
        for (int cp=c;cp<ncon;++cp){
            const float* mcp=&Mm[cp*3*NV];
            // 3x3 Gram: G[a][b] = sum_k eb_a[dc[k]] * mcp_b[dc[k]]
            float G[3][3];
            for(int a=0;a<3;++a){ const float* ea=&eb[a*CDS];
                float g0=0.f,g1=0.f,g2=0.f;
                for(int k=0;k<nc;++k){ float v=ea[k]; int dd=dc[k];
                    g0+=v*mcp[0*NV+dd]; g1+=v*mcp[1*NV+dd]; g2+=v*mcp[2*NV+dd]; }
                G[a][0]=g0; G[a][1]=g1; G[a][2]=g2; }
            // expand 4x4 pyramid block: A[(c,k),(cp,kp)] = DC[k] . G . DC[kp]
            for(int k=0;k<4;++k){
                const float* dk=DC[k];
                // tmp[b] = sum_a dk[a]*G[a][b]
                float t0=dk[0]*G[0][0]+dk[1]*G[1][0]+dk[2]*G[2][0];
                float t1=dk[0]*G[0][1]+dk[1]*G[1][1]+dk[2]*G[2][1];
                float t2=dk[0]*G[0][2]+dk[1]*G[1][2]+dk[2]*G[2][2];
                int i=4*c+k;
                for(int kp=0;kp<4;++kp){
                    const float* dq4=DC[kp];
                    float aij=t0*dq4[0]+t1*dq4[1]+t2*dq4[2];
                    int j=4*cp+kp;
                    if (i==j) aij+=R[i];
                    A[i*nefc+j]=aij; A[j*nefc+i]=aij;
                }
            }
        }
        // b for this contact's 4 rows: b[(c,k)] = d_k . bb_c - aref
        const float* bc=&bb[c*3];
        for(int k=0;k<4;++k){ const float* dk=DC[k];
            int i=4*c+k; bvec[i]=dk[0]*bc[0]+dk[1]*bc[1]+dk[2]*bc[2]-aref[i]; }
    }
    float f[G1_MAX_EFC];
    solve_pgs(A, bvec, f, nefc, pgs_iters);

    // Final correction WITHOUT contact_qfrc(J^T f) and WITHOUT a final aba_solveM:
    // per contact, basis force fb_c[a] = sum_k f[(c,k)]*d_k[a], then dq += fb_c[a]*m_{c,a}.
    // This is exactly M^-1 J^T f as a linear combo of the basis back-solves already computed.
    float dq[G1_NV];
    for (int a=0;a<NV;++a) dq[a]=0.f;
    for (int c=0;c<ncon;++c){
        const int r0=4*c; const float* mc=&Mm[c*3*NV];
        float f0=f[r0],f1=f[r0+1],f2=f[r0+2],f3=f[r0+3];
        // fb[a] = sum_k f_k * DC[k][a]
        float fb0=f0+f1+f2+f3;        // DC[*][0] all 1
        float fb1=f0-f1;              // DC[*][1] = 1,-1,0,0
        float fb2=f2-f3;              // DC[*][2] = 0,0,1,-1
        const float* m0=&mc[0*NV]; const float* m1=&mc[1*NV]; const float* m2=&mc[2*NV];
        for(int a=0;a<NV;++a) dq[a]+=fb0*m0[a]+fb1*m1[a]+fb2*m2[a];
    }
    for (int i=0;i<NV;++i) qacc[i]=qacc_s[i]+dq[i];
}
