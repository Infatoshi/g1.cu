// Multi-RHS M^-1 apply: fuse K independent aba_solveM() applies into ONE traversal of the
// stored damped factor, so each factor element (U/invD/U6/Dfac6/S) is read ONCE and applied
// to all K right-hand sides. This amortizes the local-memory factor-read traffic that
// dominates the contact solve (each separate aba_solveM re-reads the whole factor).
//
// Math is byte-for-byte the same as aba_solveM() in aba_factor.cuh (leaf->root pA backprop,
// then root->leaf accel sweep with the 6x6 base Cholesky), just vectorized over K columns.
// FROZEN aba_factor.cuh is only READ (its math is mirrored here), never edited.
#pragma once
#include "dynamics.cuh"
#include "aba.cuh"
#include "aba_factor.cuh"   // AbaFactor, ABA helpers (read-only)

// Apply M^-1 to K generalized-force RHS at once.
//   rhs : [K*NV]  column-major by RHS  -> rhs[c*NV + d]
//   out : [K*NV]  same layout          -> out[c*NV + d]
//   scr : ABA_SOLVEM_MULTI_SCR(K) floats  (pA + a6 workspaces, K columns each)
// Equivalent to: for c in 0..K-1: aba_solveM(fac, S, rhs+c*NV, _, out+c*NV).
#define ABA_SOLVEM_MULTI_SCR(K) (G1_NBODY*6*(K)*2)

template<int K>
__device__ void aba_solveM_multi(const AbaFactor* fac, const float* S,
                                 const float* rhs, float* scr, float* out) {
    const int NB = G1_NBODY;
    // pA[(b*6+i)*K + c] : articulated bias force, body b, spatial comp i, column c.
    float* pA = scr;                       // [NB*6*K]
    float* a6 = scr + (size_t)NB*6*K;      // [NB*6*K]
    for (int i=0;i<NB*6*K;++i) pA[i]=0.f;

    // ---- leaf->root: pa = U*(u*invD), u = x[d] - S^T pA  (all K columns together) ----
    for (int b=NB-1;b>=2;--b){
        int par=body_parentid[b], d=body_dofadr[b];
        const float* U=&fac->U[b*6];
        const float* Sd=&S[d*6];
        const float invD=fac->invD[b];
        float* pAb=&pA[b*6*K];
        float* pAp=&pA[par*6*K];
        // u_c = x_c[d] - sum_i Sd[i]*pAb[i*K+c]
        float ud[K];
        #pragma unroll
        for(int c=0;c<K;++c) ud[c]=rhs[c*G1_NV + d];
        for(int i=0;i<6;++i){ float si=Sd[i]; const float* row=&pAb[i*K];
            #pragma unroll
            for(int c=0;c<K;++c) ud[c]-=si*row[c]; }
        #pragma unroll
        for(int c=0;c<K;++c) ud[c]*=invD;
        // pa = pAb + U*ud, accumulate into parent
        for(int i=0;i<6;++i){ float Ui=U[i]; const float* pbi=&pAb[i*K]; float* ppi=&pAp[i*K];
            #pragma unroll
            for(int c=0;c<K;++c) ppi[c]+=pbi[c]+Ui*ud[c]; }
    }

    // ---- root->leaf accelerations. base (free joint) 6x6 Cholesky solve per column. ----
    {
        int b=1, d=body_dofadr[b];
        float* a6b=&a6[b*6*K];
        const float* pAb=&pA[b*6*K];
        // rhs6[i*K+c] = x_c[d+i] - S^T pAb (ap=0 so no U6 term beyond uk; UTa with ap=0 is 0)
        float qdd[6*K];
        for(int i=0;i<6;++i){ const float* Sdi=&S[(d+i)*6]; const float* pcol=&pAb[0];
            // uk_c = x_c[d+i] - dot6(S[d+i], pAb_col_c)
            #pragma unroll
            for(int c=0;c<K;++c){
                float u=rhs[c*G1_NV + d + i];
                float s=0;
                #pragma unroll
                for(int j=0;j<6;++j) s+=Sdi[j]*pAb[j*K+c];
                qdd[i*K+c]=u-s;
            }
        }
        // solve Dfac6 (chol L) L L^T qdd = rhs, for each column independently.
        const float* L=fac->Dfac6;
        // forward: qdd[i] = (rhs[i] - sum_{k<i} L[i,k] qdd[k]) / L[i,i]
        for(int i=0;i<6;++i){ float Lii=L[i*6+i];
            #pragma unroll
            for(int c=0;c<K;++c){
                float s=qdd[i*K+c];
                for(int k=0;k<i;++k) s-=L[i*6+k]*qdd[k*K+c];
                qdd[i*K+c]=s/Lii;
            }
        }
        // backward: qdd[i] = (qdd[i] - sum_{k>i} L[k,i] qdd[k]) / L[i,i]
        for(int i=5;i>=0;--i){ float Lii=L[i*6+i];
            #pragma unroll
            for(int c=0;c<K;++c){
                float s=qdd[i*K+c];
                for(int k=i+1;k<6;++k) s-=L[k*6+i]*qdd[k*K+c];
                qdd[i*K+c]=s/Lii;
            }
        }
        #pragma unroll
        for(int c=0;c<K;++c) for(int k=0;k<6;++k) out[c*G1_NV + d + k]=qdd[k*K+c];
        // a6[b] = ap(=0) + S*qdd
        for(int i=0;i<6;++i){
            #pragma unroll
            for(int c=0;c<K;++c){
                float Sq=0;
                #pragma unroll
                for(int k=0;k<6;++k) Sq+=S[(d+k)*6+i]*qdd[k*K+c];
                a6b[i*K+c]=Sq;
            }
        }
    }
    for (int b=2;b<NB;++b){
        int par=body_parentid[b], d=body_dofadr[b];
        const float* U=&fac->U[b*6];
        const float* Sd=&S[d*6];
        const float invD=fac->invD[b];
        const float* pAb=&pA[b*6*K];
        const float* a6p=&a6[par*6*K];
        float* a6b=&a6[b*6*K];
        // qdd_c = (x_c[d] - S^T pAb_c - U^T ap_c) * invD
        float qdd[K];
        #pragma unroll
        for(int c=0;c<K;++c) qdd[c]=rhs[c*G1_NV + d];
        for(int i=0;i<6;++i){ float si=Sd[i]; float Ui=U[i];
            #pragma unroll
            for(int c=0;c<K;++c) qdd[c]-=si*pAb[i*K+c]+Ui*a6p[i*K+c]; }
        #pragma unroll
        for(int c=0;c<K;++c){ qdd[c]*=invD; out[c*G1_NV + d]=qdd[c]; }
        for(int i=0;i<6;++i){ float si=Sd[i];
            #pragma unroll
            for(int c=0;c<K;++c) a6b[i*K+c]=a6p[i*K+c]+si*qdd[c]; }
    }
}
