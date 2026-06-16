// Articulated-Body Algorithm (Featherstone) forward dynamics for the G1, fp32.
// O(nbody) instead of dense CRBA + O(nv^3) Cholesky. Computes qacc = M^-1(-bias)
// to the same answer as src/dynamics.cuh (validated vs MuJoCo) but with a much
// smaller working set and far shorter dependency chains -> less DRAM spill.
//
// Common-frame formulation (world axes about the pelvis origin), same basis as the
// CRBA/RNE we validated, so qacc lands in MuJoCo's generalized-velocity basis.
// Only ONE matrix solve: a 6x6 at the floating base. Every hinge is scalar.
#pragma once
#include "dynamics.cuh"

__device__ __forceinline__ void mv6(const float* M, const float* x, float* o) {
    for (int i=0;i<6;++i){ float s=0; for (int j=0;j<6;++j) s+=M[i*6+j]*x[j]; o[i]=s; }
}
__device__ __forceinline__ float dot6(const float* a, const float* b){
    float s=0; for(int i=0;i<6;++i) s+=a[i]*b[i]; return s;
}
// packed upper-triangular index for a symmetric 6x6 (21 unique). spatial inertia is
// symmetric, so storing 21 instead of 36 cuts the ABA working set ~20% (DRAM-bound win).
__device__ __forceinline__ int symidx(int i, int j){
    if (i>j){ int t=i; i=j; j=t; }
    return i*6 - (i*(i-1))/2 + (j-i);
}
// symmetric 6x6 (21 packed) times vector
__device__ __forceinline__ void symmv6(const float* P, const float* x, float* o){
    for (int i=0;i<6;++i){ float s=0; for(int j=0;j<6;++j) s+=P[symidx(i,j)]*x[j]; o[i]=s; }
}
// spatial force cross: v=[w;vv] x* f=[n;f0] = [w x n + vv x f0 ; w x f0]
__device__ __forceinline__ void crossf6(const float* v, const float* f, float* o){
    V3 w={v[0],v[1],v[2]}, vv={v[3],v[4],v[5]}, n={f[0],f[1],f[2]}, f0={f[3],f[4],f[5]};
    V3 a=cross(w,n), b=cross(vv,f0), c=cross(w,f0);
    o[0]=a.x+b.x; o[1]=a.y+b.y; o[2]=a.z+b.z; o[3]=c.x; o[4]=c.y; o[5]=c.z;
}
// 6x6 SPD solve D x = b (in place factor of Dwork)
__device__ void chol6_solve(float* D, const float* b, float* x){
    for (int i=0;i<6;++i){
        for (int j=0;j<=i;++j){
            float s=D[i*6+j]; for(int k=0;k<j;++k) s-=D[i*6+k]*D[j*6+k];
            if(i==j) D[i*6+i]=sqrtf(s); else D[i*6+j]=s/D[j*6+j];
        }
    }
    for (int i=0;i<6;++i){ float s=b[i]; for(int k=0;k<i;++k) s-=D[i*6+k]*x[k]; x[i]=s/D[i*6+i]; }
    for (int i=5;i>=0;--i){ float s=x[i]; for(int k=i+1;k<6;++k) s-=D[k*6+i]*x[k]; x[i]=s/D[i*6+i]; }
}

// scratch size for aba_qacc working arrays (IA[symmetric 21] + vel + cb + pA + a6)
#define ABA_SCR (G1_NBODY*21 + 4*G1_NBODY*6)
// qacc = M^-1(-bias) via ABA. xpos/xquat/xipos from forward_kinematics. `scr` (ABA_SCR
// floats) and `S` may point at local OR shared memory -- lets the caller choose residency.
__device__ void aba_qacc(const float* qpos, const float* qvel,
                         const float* xpos, const float* xquat, const float* xipos,
                         float* S, float* scr, float* qacc) {
    const int NB=G1_NBODY, NV=G1_NV;
    V3 p={xpos[3],xpos[4],xpos[5]};
    Q4 q1={xquat[4],xquat[5],xquat[6],xquat[7]};
    M3 R1=quat2mat(q1);

    // ---- motion axes S (about p, world axes) ----
    for (int i=0;i<NV*6;++i) S[i]=0.f;
    for (int j=0;j<G1_NJNT;++j){
        int t=jnt_type[j], b=jnt_bodyid[j], d=jnt_dofadr[j];
        if (t==0){
            S[(d+0)*6+3]=1; S[(d+1)*6+4]=1; S[(d+2)*6+5]=1;
            for (int k=0;k<3;++k){ V3 ax=col(R1,k); S[(d+3+k)*6+0]=ax.x; S[(d+3+k)*6+1]=ax.y; S[(d+3+k)*6+2]=ax.z; }
        } else if (t==3){
            Q4 qb={xquat[4*b],xquat[4*b+1],xquat[4*b+2],xquat[4*b+3]};
            V3 aw=qrot(qb,(V3){jnt_axis[3*j],jnt_axis[3*j+1],jnt_axis[3*j+2]});
            V3 anchor=vadd((V3){xpos[3*b],xpos[3*b+1],xpos[3*b+2]}, qrot(qb,(V3){jnt_pos[3*j],jnt_pos[3*j+1],jnt_pos[3*j+2]}));
            V3 r=vsub(anchor,p), rxa=cross(r,aw);
            S[d*6+0]=aw.x; S[d*6+1]=aw.y; S[d*6+2]=aw.z; S[d*6+3]=rxa.x; S[d*6+4]=rxa.y; S[d*6+5]=rxa.z;
        }
    }

    // ---- per-body spatial inertia (6x6 about p) -> init IA ; also keep for pA init ----
    float* IA = scr;   // [NB*21] symmetric-packed spatial/articulated inertia
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
        float* X=&IA[b*21];                          // pack symmetric upper triangle
        for(int i=0;i<6;++i)for(int j=i;j<6;++j) X[symidx(i,j)]=M6[i*6+j];
    }

    // ---- pass 1: velocities v[b] (about p), bias accel c[b] = sum Sdot_a qd_a ----
    float* vel=scr+G1_NBODY*21; float* cb=vel+G1_NBODY*6; float* pA=cb+G1_NBODY*6;
    for (int i=0;i<6;++i){ vel[i]=0; cb[i]=0; }
    for (int b=1;b<NB;++b){
        int par=body_parentid[b], da=body_dofadr[b], dn=body_dofnum[b];
        float w[6]; for(int i=0;i<6;++i) w[i]=vel[par*6+i];
        float c6[6]={0,0,0,0,0,0};
        // helper: accumulate cdof_dot using free-joint-aware partial velocity
        auto add_sdot=[&](int a, const float* wpart){
            V3 pw={wpart[0],wpart[1],wpart[2]}, pv={wpart[3],wpart[4],wpart[5]};
            V3 sw={S[a*6+0],S[a*6+1],S[a*6+2]}, sv={S[a*6+3],S[a*6+4],S[a*6+5]};
            V3 ow,ov; crossm(pw,pv,sw,sv,ow,ov);
            float qd=qvel[a];
            c6[0]+=ow.x*qd; c6[1]+=ow.y*qd; c6[2]+=ow.z*qd;
            c6[3]+=ov.x*qd; c6[4]+=ov.y*qd; c6[5]+=ov.z*qd;
        };
        if (dn==6){
            for(int a=da;a<da+3;++a){ add_sdot(a,w); float qd=qvel[a]; for(int k=0;k<6;++k) w[k]+=S[a*6+k]*qd; }
            float wr[6]; for(int k=0;k<6;++k) wr[k]=w[k];
            for(int a=da+3;a<da+6;++a) add_sdot(a,wr);
            for(int a=da+3;a<da+6;++a){ float qd=qvel[a]; for(int k=0;k<6;++k) w[k]+=S[a*6+k]*qd; }
        } else {
            for(int a=da;a<da+dn;++a){ add_sdot(a,w); float qd=qvel[a]; for(int k=0;k<6;++k) w[k]+=S[a*6+k]*qd; }
        }
        for(int k=0;k<6;++k){ vel[b*6+k]=w[k]; cb[b*6+k]=c6[k]; }
    }
    // pA init = v x* (I_body v)  (body inertia = IA before accumulation)
    for (int b=1;b<NB;++b){
        float Iv[6]; symmv6(&IA[b*21], &vel[b*6], Iv);
        crossf6(&vel[b*6], Iv, &pA[b*6]);
    }

    // ---- pass 2: leaf->root, propagate articulated inertia/bias (hinges only; base is root) ----
    for (int b=NB-1;b>=2;--b){
        int par=body_parentid[b], d=body_dofadr[b];   // hinge: single dof
        float U[6]; symmv6(&IA[b*21], &S[d*6], U);
        float D=dot6(&S[d*6], U) + dof_armature[d];   // + reflected rotor inertia
        float u=-dot6(&S[d*6], &pA[b*6]);
        float invD=1.f/D;
        // Ia = IA - U U^T / D ; pa = pA + Ia*c + U*(u/D)  (symmetric, 21 packed)
        float Ia[21];
        for(int i=0;i<6;++i)for(int j=i;j<6;++j) Ia[symidx(i,j)]=IA[b*21+symidx(i,j)]-U[i]*U[j]*invD;
        float Iac[6]; symmv6(Ia,&cb[b*6],Iac);
        float pa[6]; for(int k=0;k<6;++k) pa[k]=pA[b*6+k]+Iac[k]+U[k]*(u*invD);
        for(int k=0;k<21;++k) IA[par*21+k]+=Ia[k];
        for(int k=0;k<6;++k)  pA[par*6+k]+=pa[k];
    }

    // ---- pass 3: root->leaf accelerations ----
    float* a6=pA+G1_NBODY*6;
    float a0[6]={0,0,0,-G1_GRAVITY[0],-G1_GRAVITY[1],-G1_GRAVITY[2]}; // world accel = -g
    // base (body 1, free joint): solve 6x6
    {
        int b=1, d=body_dofadr[b];
        float ap[6]; for(int k=0;k<6;++k) ap[k]=a0[k]+cb[b*6+k];   // a' = a_parent + c
        // U columns U_l = IA*S_l ; D = S^T U (6x6) ; rhs = u - U^T a'
        float U[36], Dm[36], rhs[6];
        for(int l=0;l<6;++l){ float Ul[6]; symmv6(&IA[b*21],&S[(d+l)*6],Ul); for(int i=0;i<6;++i) U[i*6+l]=Ul[i]; }
        for(int k=0;k<6;++k)for(int l=0;l<6;++l){ float s=0; for(int i=0;i<6;++i) s+=S[(d+k)*6+i]*U[i*6+l]; Dm[k*6+l]=s + (k==l?dof_armature[d+k]:0.f); }
        for(int k=0;k<6;++k){ float uk=-dot6(&S[(d+k)*6],&pA[b*6]); float UTa=0; for(int i=0;i<6;++i) UTa+=U[i*6+k]*ap[i]; rhs[k]=uk-UTa; }
        float qdd[6]; chol6_solve(Dm, rhs, qdd);
        for(int k=0;k<6;++k) qacc[d+k]=qdd[k];
        for(int i=0;i<6;++i){ float Sqdd=0; for(int k=0;k<6;++k) Sqdd+=S[(d+k)*6+i]*qdd[k]; a6[b*6+i]=ap[i]+Sqdd; }
    }
    for (int b=2;b<NB;++b){
        int par=body_parentid[b], d=body_dofadr[b];
        float ap[6]; for(int k=0;k<6;++k) ap[k]=a6[par*6+k]+cb[b*6+k];
        float U[6]; symmv6(&IA[b*21],&S[d*6],U);
        float D=dot6(&S[d*6],U) + dof_armature[d];
        float u=-dot6(&S[d*6],&pA[b*6]);
        float qdd=(u - dot6(U,ap))/D;
        qacc[d]=qdd;
        for(int i=0;i<6;++i) a6[b*6+i]=ap[i]+S[d*6+i]*qdd;
    }
}
