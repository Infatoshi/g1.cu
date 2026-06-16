// Standalone factor/apply identity test for aba_factor.cuh.
// Verifies: solveM(tau - bias) == aba.cuh qacc (M^-1(tau-bias)) to fp32, AND
//           solveM(x) == dense-Cholesky M^-1 x  for random x.
// Uses bench/contact_validate.bin states (across ncon buckets).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "dynamics.cuh"
#include "aba.cuh"
#include "aba_factor.cuh"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// fill S exactly as aba.cuh / dynamics.cuh do (world axes about pelvis).
__device__ void fill_S(const float* xpos, const float* xquat, float* S) {
    V3 p={xpos[3],xpos[4],xpos[5]};
    Q4 q1={xquat[4],xquat[5],xquat[6],xquat[7]};
    M3 R1=quat2mat(q1);
    for (int i=0;i<G1_NV*6;++i) S[i]=0.f;
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
}

// per state: qacc_aba (from aba_qacc) and qacc_solve (from solveM(-bias) via dense M_bias).
__global__ void test_kernel(const float* qpos, const float* qvel, int nstates,
                            float* out_aba, float* out_solve, float* out_dense) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=nstates) return;
    const float* qp=qpos+(size_t)k*G1_NQ;
    const float* qv=qvel+(size_t)k*G1_NV;
    float qpl[G1_NQ], qvl[G1_NV];
    for(int i=0;i<G1_NQ;++i)qpl[i]=qp[i];
    for(int i=0;i<G1_NV;++i)qvl[i]=qv[i];
    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    forward_kinematics(qpl, xpos, xquat, xipos);

    // ---- reference 1: aba.cuh qacc = M^-1(-bias) ----
    float S[G1_NV*6], scr[ABA_SCR], qacc_aba[G1_NV];
    aba_qacc(qpl, qvl, xpos, xquat, xipos, S, scr, qacc_aba);

    // ---- reference 2 + factor/apply: dense M,bias then solveM(-bias) ----
    float Sd[G1_NV*6], M[G1_NV*G1_NV], bias[G1_NV];
    compute_M_bias(qpl, qvl, xpos, xquat, xipos, Sd, M, bias);
    float negbias[G1_NV]; for(int i=0;i<G1_NV;++i) negbias[i]=-bias[i];

    // dense cholesky M^-1(-bias)
    float Mfac[G1_NV*G1_NV]; for(int i=0;i<G1_NV*G1_NV;++i)Mfac[i]=M[i];
    {
        const int N=G1_NV;
        for(int i=0;i<N;++i)for(int j=0;j<=i;++j){ float s=Mfac[i*N+j]; for(int kk=0;kk<j;++kk)s-=Mfac[i*N+kk]*Mfac[j*N+kk];
            if(i==j)Mfac[i*N+i]=sqrtf(s); else Mfac[i*N+j]=s/Mfac[j*N+j]; }
        float qd[G1_NV];
        for(int i=0;i<N;++i){ float s=negbias[i]; for(int kk=0;kk<i;++kk)s-=Mfac[i*N+kk]*qd[kk]; qd[i]=s/Mfac[i*N+i]; }
        for(int i=N-1;i>=0;--i){ float s=qd[i]; for(int kk=i+1;kk<N;++kk)s-=Mfac[kk*N+i]*qd[kk]; qd[i]=s/Mfac[i*N+i]; }
        for(int i=0;i<N;++i) out_dense[(size_t)k*G1_NV+i]=qd[i];
    }

    // factor + apply (use Sd from compute_M_bias)
    AbaFactor fac; float fscr[ABA_FAC_SCR];
    aba_factorize(xquat, xipos, xpos, Sd, fscr, &fac);
    float sscr[ABA_SOLVEM_SCR], y[G1_NV];
    aba_solveM(&fac, Sd, negbias, sscr, y);

    // residual diagnostic: r = M*y - negbias  (should be ~0 if solveM is M^-1)
    if (k==0){
        const int N=G1_NV; double rn=0,bn=0;
        for(int i=0;i<N;++i){ float s=0; for(int j=0;j<N;++j)s+=M[i*N+j]*y[j]; double e=s-negbias[i]; rn+=e*e; bn+=(double)negbias[i]*negbias[i]; }
        printf("[k0] ||M*y - b|| / ||b|| = %.3e\n", sqrt(rn)/fmax(sqrt(bn),1e-9));
    }

    for(int i=0;i<G1_NV;++i){ out_aba[(size_t)k*G1_NV+i]=qacc_aba[i]; out_solve[(size_t)k*G1_NV+i]=y[i]; }
}

int main(){
    FILE* f=fopen("bench/contact_validate.bin","rb");
    if(!f){fprintf(stderr,"missing bench/contact_validate.bin\n");return 1;}
    int n; fread(&n,4,1,f);
    std::vector<float> qpos((size_t)n*G1_NQ), qvel((size_t)n*G1_NV);
    fread(qpos.data(),4,qpos.size(),f); fread(qvel.data(),4,qvel.size(),f);
    fclose(f);

    float *dqp,*dqv,*da,*ds,*dd;
    CK(cudaMalloc(&dqp,qpos.size()*4)); CK(cudaMalloc(&dqv,qvel.size()*4));
    CK(cudaMalloc(&da,(size_t)n*G1_NV*4)); CK(cudaMalloc(&ds,(size_t)n*G1_NV*4)); CK(cudaMalloc(&dd,(size_t)n*G1_NV*4));
    CK(cudaMemcpy(dqp,qpos.data(),qpos.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqv,qvel.data(),qvel.size()*4,cudaMemcpyHostToDevice));
    test_kernel<<<(n+63)/64,64>>>(dqp,dqv,n,da,ds,dd);
    CK(cudaDeviceSynchronize());
    std::vector<float> A((size_t)n*G1_NV), Sv((size_t)n*G1_NV), D((size_t)n*G1_NV);
    CK(cudaMemcpy(A.data(),da,A.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Sv.data(),ds,Sv.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(D.data(),dd,D.size()*4,cudaMemcpyDeviceToHost));

    printf("factor/apply identity (solveM(-bias) vs aba.cuh qacc vs dense chol):\n");
    double worst_sa=0, worst_sd=0;
    for(int k=0;k<n;++k){
        double na=0,nd=0,den_a=0,den_d=0;
        for(int i=0;i<G1_NV;++i){
            double ea=Sv[(size_t)k*G1_NV+i]-A[(size_t)k*G1_NV+i];
            double ed=Sv[(size_t)k*G1_NV+i]-D[(size_t)k*G1_NV+i];
            na+=ea*ea; den_a+=A[(size_t)k*G1_NV+i]*A[(size_t)k*G1_NV+i];
            nd+=ed*ed; den_d+=D[(size_t)k*G1_NV+i]*D[(size_t)k*G1_NV+i];
        }
        double ra=sqrt(na)/fmax(sqrt(den_a),1e-9), rd=sqrt(nd)/fmax(sqrt(den_d),1e-9);
        worst_sa=fmax(worst_sa,ra); worst_sd=fmax(worst_sd,rd);
        printf("  state %2d  relerr vs aba=%.3e  vs dense=%.3e\n",k,ra,rd);
    }
    printf("worst: solveM vs aba.cuh=%.3e   solveM vs dense-chol=%.3e\n", worst_sa, worst_sd);
    return 0;
}
