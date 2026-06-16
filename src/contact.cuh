// Foot-ground contact solver for the G1, fp32. SCAFFOLD — validated in numpy first
// (scripts/contact_np.py, qacc matched to MuJoCo at 1e-13). See src/contact_DESIGN.md.
//
// Pipeline per world:
//   1. detect_foot_contacts : 8 foot spheres vs z=0 plane (branchless, uniform)
//   2. build_contact_jac    : pyramidal condim-3 J (4 rows / contact), MuJoCo frame
//   3. solve_pgs            : boxed QP  min 0.5 f^T A f + f^T b, f>=0  via PGS
//   4. apply                : qacc = qacc_smooth + M^-1 J^T f   (M^-1 via aba.cuh)
//
// NOTE: this is the structural scaffold. The R/aref regularization (solref/solimp ->
// efc_R/efc_aref) is NOT yet reconstructed on-device (see DESIGN sec 4 "Open correctness
// item"); the host currently supplies R/aref. Wire that in before trusting CUDA numbers.
#pragma once
#include "dynamics.cuh"
// g1_model.h provides G1_NFOOT_SPHERE, G1_FOOT_RADIUS, G1_FOOT_MU, G1_FOOT_CONDIM,
// g1_foot_body[], g1_foot_lpos[], G1_FOOT_SOLREF/SOLIMP (emitted by export_model.py).
#include "g1_model.h"

#define G1_MAX_CONTACT  G1_NFOOT_SPHERE        // foot-ground: at most 1 contact / sphere
#define G1_MAX_EFC      (4 * G1_MAX_CONTACT)   // pyramidal condim-3: 4 rows / contact

// one detected contact: host body, world contact point, penetration (<0), + cached J rows.
struct Contact {
    int   body;
    float pt[3];     // world contact point = [cx, cy, pen/2]
    float pen;       // c_z - r  (< 0)
};

// ---- 1. collision detection: 8 foot spheres vs z=0 plane ----
// xpos/xquat from forward_kinematics (same as aba_qacc inputs). Returns ncon, fills cons[].
__device__ int detect_foot_contacts(const float* xpos, const float* xquat, Contact* cons) {
    int ncon = 0;
    for (int s = 0; s < G1_NFOOT_SPHERE; ++s) {
        int b = g1_foot_body[s];
        Q4 qb = {xquat[4*b],xquat[4*b+1],xquat[4*b+2],xquat[4*b+3]};
        V3 lp = {g1_foot_lpos[3*s],g1_foot_lpos[3*s+1],g1_foot_lpos[3*s+2]};
        V3 c  = vadd((V3){xpos[3*b],xpos[3*b+1],xpos[3*b+2]}, qrot(qb,lp));
        float pen = c.z - G1_FOOT_RADIUS;
        if (pen < 0.f) {
            Contact& k = cons[ncon++];
            k.body = b; k.pen = pen;
            k.pt[0] = c.x; k.pt[1] = c.y; k.pt[2] = pen * 0.5f;
        }
    }
    return ncon;
}

// translational point Jacobian column for dof a at world point pt (3 entries), using the
// ABA motion axes S (NV x 6, world axes about pelvis): col = S.ang x (pt - p) + S.lin,
// where p is the pelvis origin (the S reference point). Matches mj_jac for these contacts.
__device__ __forceinline__ void point_jac_col(const float* S, int a, V3 pt, V3 p, float* out3) {
    V3 w  = {S[a*6+0],S[a*6+1],S[a*6+2]};
    V3 lv = {S[a*6+3],S[a*6+4],S[a*6+5]};
    V3 r  = vsub(pt, p);
    V3 t  = vadd(cross(w, r), lv);
    out3[0]=t.x; out3[1]=t.y; out3[2]=t.z;
}

// ---- 2. contact Jacobian J (nefc x NV), pyramidal condim-3, MuJoCo frame ----
// normal=[0,0,1], t1=[0,1,0], t2=[-1,0,0]. Rows: n+mu*t1, n-mu*t1, n+mu*t2, n-mu*t2.
// S is the ABA motion-axis array; p = pelvis origin (xpos of body 1). nefc = 4*ncon.
// CRITICAL: a dof contributes to a contact only if it is an ANCESTOR of the contact body
// (it lies on the kinematic chain world->contact_body). Other dofs get exactly 0 — matching
// MuJoCo's mj_jac. Without this, distant dofs leak nonzero columns and A is wrong.
__device__ int build_contact_jac(const Contact* cons, int ncon,
                                 const float* S, V3 p, float* J /*G1_MAX_EFC*NV*/) {
    const int NV = G1_NV;
    int row = 0;
    for (int c = 0; c < ncon; ++c) {
        V3 pt = {cons[c].pt[0],cons[c].pt[1],cons[c].pt[2]};
        float* r_npt1 = &J[(row+0)*NV];
        float* r_nmt1 = &J[(row+1)*NV];
        float* r_npt2 = &J[(row+2)*NV];
        float* r_nmt2 = &J[(row+3)*NV];
        for (int a = 0; a < NV; ++a) { r_npt1[a]=r_nmt1[a]=r_npt2[a]=r_nmt2[a]=0.f; }
        // walk world->contact_body chain; for each ancestor body, fill its dofs.
        for (int b = cons[c].body; b >= 1; b = body_parentid[b]) {
            int da = body_dofadr[b], dn = body_dofnum[b];
            for (int a = da; a < da+dn; ++a) {
                float jc[3]; point_jac_col(S, a, pt, p, jc);
                float jn  =  jc[2];          // normal  [0,0,1]
                float jt1 =  jc[1];          // t1      [0,1,0]
                float jt2 = -jc[0];          // t2      [-1,0,0]
                r_npt1[a] = jn + G1_FOOT_MU*jt1;
                r_nmt1[a] = jn - G1_FOOT_MU*jt1;
                r_npt2[a] = jn + G1_FOOT_MU*jt2;
                r_nmt2[a] = jn - G1_FOOT_MU*jt2;
            }
        }
        row += 4;
    }
    return row; // nefc
}

// ---- 2b. R / aref reconstruction from solref/solimp (Phase-1 closed; see contact_np2.py) ----
// MuJoCo impedance sigmoid for solimp=[dmin,dmax,width,mid,power], pos=penetration(<0).
__device__ __forceinline__ float foot_impedance(float pos) {
    float dmin=G1_FOOT_SOLIMP[0], dmax=G1_FOOT_SOLIMP[1], width=G1_FOOT_SOLIMP[2];
    float mid=G1_FOOT_SOLIMP[3], power=G1_FOOT_SOLIMP[4];
    float x = fabsf(pos)/width;
    float y;
    if (x >= 1.f) y = 1.f;
    else {
        float a = 1.f/powf(mid, power-1.f);
        float b = 1.f/powf(1.f-mid, power-1.f);
        y = (x < mid) ? a*powf(x,power) : 1.f - b*powf(1.f-x,power);
    }
    return dmin + y*(dmax-dmin);
}

// Fill efc_R (nefc) and efc_aref (nefc) for pyramidal condim-3 foot contacts.
// R is per-contact (shared by its 4 rows); aref is per-row (uses J row . qvel).
// invw0_per_con[c] = body_invweight0 of contact c's foot body (floor contributes 0).
__device__ void build_R_aref(const Contact* cons, int ncon, const float* invw0_per_con,
                             const float* J, const float* qvel, float* R, float* aref) {
    const int NV = G1_NV;
    float dmax = G1_FOOT_SOLIMP[1];
    float tc = G1_FOOT_SOLREF[0], dr = G1_FOOT_SOLREF[1];
    float bcoef = 2.f/(dmax*tc);
    float kcoef = 1.f/(dmax*dmax*tc*tc*dr*dr);
    float mu = G1_FOOT_MU;
    float pyr = (1.f + mu*mu) * (2.f*mu*mu / G1_FOOT_IMPRATIO);
    for (int c = 0; c < ncon; ++c) {
        float pen = cons[c].pen;
        float imp = foot_impedance(pen);
        float diagApprox = invw0_per_con[c] * pyr;
        float Rc = diagApprox * (1.f - imp) / imp;
        for (int r = 0; r < 4; ++r) {
            int idx = 4*c + r;
            R[idx] = Rc;
            float jv = 0.f;
            const float* Jr = &J[idx*NV];
            for (int a = 0; a < NV; ++a) jv += Jr[a]*qvel[a];
            aref[idx] = -bcoef*jv - kcoef*imp*pen;
        }
    }
}

// ---- Cholesky factor-once + multi back-solve (reuse factor for the nefc M^-1 J^T columns) ----
__device__ void chol_factor(float* M) {
    const int N = G1_NV;
    for (int i=0;i<N;++i)
        for (int j=0;j<=i;++j){
            float s=M[i*N+j];
            for (int k=0;k<j;++k) s-=M[i*N+k]*M[j*N+k];
            if (i==j) M[i*N+i]=sqrtf(s);
            else      M[i*N+j]=s/M[j*N+j];
        }
}
// Solve M x = b given pre-factored L (from chol_factor). b may alias x.
__device__ void chol_backsolve(const float* L, const float* b, float* x) {
    const int N = G1_NV;
    for (int i=0;i<N;++i){ float s=b[i]; for(int k=0;k<i;++k) s-=L[i*N+k]*x[k]; x[i]=s/L[i*N+i]; }
    for (int i=N-1;i>=0;--i){ float s=x[i]; for(int k=i+1;k<N;++k) s-=L[k*N+i]*x[k]; x[i]=s/L[i*N+i]; }
}

// ---- 3. PGS solve of min 0.5 f^T A f + f^T b, f>=0 ----
// A (nefc x nefc) SPD-ish (= J M^-1 J^T + diag(R)), b (nefc). Writes f (nefc).
__device__ void solve_pgs(const float* A, const float* b, float* f, int nefc, int iters) {
    for (int i = 0; i < nefc; ++i) f[i] = 0.f;
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < nefc; ++i) {
            float Aii = A[i*nefc+i];
            float ri = b[i];
            for (int j = 0; j < nefc; ++j) ri += A[i*nefc+j]*f[j];
            ri -= Aii*f[i];
            float fi = -ri / Aii;
            f[i] = fi > 0.f ? fi : 0.f;
        }
    }
}

// ---- 4. assemble qacc correction: qacc = qacc_smooth + M^-1 J^T f ----
// Caller supplies Minv_action(JTf -> dqacc) via ABA, OR a dense Minv. Here: given Jtf (NV),
// the caller adds M^-1 Jtf to qacc_smooth. (Kept thin: M^-1 lives in aba.cuh.)
__device__ void contact_qfrc(const float* J, const float* f, int nefc, float* Jtf /*NV*/) {
    const int NV = G1_NV;
    for (int a = 0; a < NV; ++a) {
        float s = 0.f;
        for (int r = 0; r < nefc; ++r) s += J[r*NV+a]*f[r];
        Jtf[a] = s;
    }
}
