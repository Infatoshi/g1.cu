// Experiment: ABA with ALL working-set scratch in fp16 (IA + vel/cb/pA/a6); compute fp32.
// Halves the local-memory traffic of the entire spilled working set. Accuracy is RL-grade
// (measured), not bit-match. fp32 default stays in aba.cuh.
#pragma once
#include "aba.cuh"
#include <cuda_fp16.h>

__device__ __forceinline__ void symmv6_h(const __half* P, const float* x, float* o){
    for(int i=0;i<6;++i){ float s=0; for(int j=0;j<6;++j) s+=__half2float(P[symidx(i,j)])*x[j]; o[i]=s; }
}
__device__ __forceinline__ void ld6(const __half* p, float* o){ for(int k=0;k<6;++k) o[k]=__half2float(p[k]); }
__device__ __forceinline__ void st6(__half* p, const float* x){ for(int k=0;k<6;++k) p[k]=__float2half(x[k]); }

#define ABA_SCR_H (4*G1_NBODY*6)   // vel+cb+pA+a6 as __half (IA is separate __half buffer)

__device__ void aba_qacc_h(const float* qpos, const float* qvel,
                           const float* xpos, const float* xquat, const float* xipos,
                           float* S, __half* scr, __half* IA, float* qacc) {
    const int NB=G1_NBODY, NV=G1_NV;
    V3 p={xpos[3],xpos[4],xpos[5]};
    Q4 q1={xquat[4],xquat[5],xquat[6],xquat[7]};
    M3 R1=quat2mat(q1);

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

    // ---- per-body spatial inertia -> IA (fp16) ----
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
        __half* X=&IA[b*21];
        for(int i=0;i<6;++i)for(int j=i;j<6;++j) X[symidx(i,j)]=__float2half(M6[i*6+j]);
    }

    // scratch (fp16)
    __half* vel=scr; __half* cb=vel+G1_NBODY*6; __half* pA=cb+G1_NBODY*6; __half* a6=pA+G1_NBODY*6;

    // ---- pass 1 ----
    for(int i=0;i<6;++i){ vel[i]=__float2half(0.f); cb[i]=__float2half(0.f); }
    for (int b=1;b<NB;++b){
        int par=body_parentid[b], da=body_dofadr[b], dn=body_dofnum[b];
        float w[6]; ld6(&vel[par*6], w);
        float c6[6]={0,0,0,0,0,0};
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
        st6(&vel[b*6], w); st6(&cb[b*6], c6);
    }
    for (int b=1;b<NB;++b){
        float v6[6]; ld6(&vel[b*6], v6);
        float Iv[6]; symmv6_h(&IA[b*21], v6, Iv);
        float pa[6]; crossf6(v6, Iv, pa); st6(&pA[b*6], pa);
    }

    // ---- pass 2 ----
    for (int b=NB-1;b>=2;--b){
        int par=body_parentid[b], d=body_dofadr[b];
        float U[6]; symmv6_h(&IA[b*21], &S[d*6], U);
        float D=dot6(&S[d*6], U) + dof_armature[d];
        float pAb[6]; ld6(&pA[b*6], pAb);
        float u=-dot6(&S[d*6], pAb);
        float invD=1.f/D;
        float Ia[21];
        for(int i=0;i<6;++i)for(int j=i;j<6;++j){ int s=symidx(i,j); Ia[s]=__half2float(IA[b*21+s])-U[i]*U[j]*invD; }
        float cbb[6]; ld6(&cb[b*6], cbb);
        float Iac[6]; for(int i=0;i<6;++i){ float ss=0; for(int j=0;j<6;++j) ss+=Ia[symidx(i,j)]*cbb[j]; Iac[i]=ss; }
        float pa[6]; for(int k=0;k<6;++k) pa[k]=pAb[k]+Iac[k]+U[k]*(u*invD);
        for(int k=0;k<21;++k) IA[par*21+k]=__float2half(__half2float(IA[par*21+k])+Ia[k]);
        float pAp[6]; ld6(&pA[par*6], pAp); for(int k=0;k<6;++k) pAp[k]+=pa[k]; st6(&pA[par*6], pAp);
    }

    // ---- pass 3 ----
    float a0[6]={0,0,0,-G1_GRAVITY[0],-G1_GRAVITY[1],-G1_GRAVITY[2]};
    {
        int b=1, d=body_dofadr[b];
        float cbb[6]; ld6(&cb[6], cbb);
        float ap[6]; for(int k=0;k<6;++k) ap[k]=a0[k]+cbb[k];
        float pAb[6]; ld6(&pA[6], pAb);
        float U[36], Dm[36], rhs[6];
        for(int l=0;l<6;++l){ float Ul[6]; symmv6_h(&IA[21],&S[(d+l)*6],Ul); for(int i=0;i<6;++i) U[i*6+l]=Ul[i]; }
        for(int k=0;k<6;++k)for(int l=0;l<6;++l){ float s=0; for(int i=0;i<6;++i) s+=S[(d+k)*6+i]*U[i*6+l]; Dm[k*6+l]=s + (k==l?dof_armature[d+k]:0.f); }
        for(int k=0;k<6;++k){ float uk=-dot6(&S[(d+k)*6],pAb); float UTa=0; for(int i=0;i<6;++i) UTa+=U[i*6+k]*ap[i]; rhs[k]=uk-UTa; }
        float qdd[6]; chol6_solve(Dm, rhs, qdd);
        for(int k=0;k<6;++k) qacc[d+k]=qdd[k];
        float aw[6]; for(int i=0;i<6;++i){ float Sq=0; for(int k=0;k<6;++k) Sq+=S[(d+k)*6+i]*qdd[k]; aw[i]=ap[i]+Sq; } st6(&a6[6], aw);
    }
    for (int b=2;b<NB;++b){
        int par=body_parentid[b], d=body_dofadr[b];
        float ap6[6], cbb[6]; ld6(&a6[par*6], ap6); ld6(&cb[b*6], cbb);
        float ap[6]; for(int k=0;k<6;++k) ap[k]=ap6[k]+cbb[k];
        float U[6]; symmv6_h(&IA[b*21],&S[d*6],U);
        float D=dot6(&S[d*6],U) + dof_armature[d];
        float pAb[6]; ld6(&pA[b*6], pAb);
        float u=-dot6(&S[d*6],pAb);
        float qdd=(u - dot6(U,ap))/D;
        qacc[d]=qdd;
        float aw[6]; for(int i=0;i<6;++i) aw[i]=ap[i]+S[d*6+i]*qdd; st6(&a6[b*6], aw);
    }
}
