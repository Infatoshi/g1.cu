// Equivalence gate: aba_solveM_multi<K>(fac,S,rhs) must reproduce K sequential aba_solveM
// calls bit-for-bit (<=1e-6). Builds REAL damped factors from random physical states (same
// path env_qacc uses), then random RHS columns.
//   nvcc -arch=sm_86 -O3 src/test_solvem_multi.cu -lcurand -o build/test_solvem_multi
#include <cstdio>
#include <cmath>
#include <curand_kernel.h>
#include "dynamics.cuh"
#include "aba.cuh"
#include "aba_factor.cuh"
#include "aba_bias.cuh"
#include "aba_solvem_multi.cuh"

#define KK 3
#define ACT_DIM 29

__global__ void k(int Nstate, unsigned long seed, float* maxabs, float* maxrel) {
    int w = blockIdx.x*blockDim.x + threadIdx.x; if (w >= Nstate) return;
    curandState st; curand_init(seed, w, 0, &st);
    const int NV=G1_NV;
    float qpos[G1_NQ], qvel[G1_NV];
    for (int i=0;i<G1_NQ;++i) qpos[i] = g1_qpos_stand[i] + 0.3f*(curand_uniform(&st)*2.f-1.f);
    float n=0; for(int i=3;i<7;++i) n+=qpos[i]*qpos[i]; n=sqrtf(n);
    for(int i=3;i<7;++i) qpos[i]/=n;
    for (int i=0;i<NV;++i) qvel[i] = 1.5f*(curand_uniform(&st)*2.f-1.f);

    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    forward_kinematics(qpos, xpos, xquat, xipos);

    float Bdt[G1_NV]; for (int i=0;i<NV;++i) Bdt[i]=0.f;
    for (int i=0;i<ACT_DIM;++i) Bdt[g1_act_dof[i]] = G1_DT*g1_act_kd[i];

    // Fill S in the exact basis the factor consumes (same call env_qacc uses); bias discarded.
    float S[G1_NV*6], bias[G1_NV];
    compute_bias(qpos, qvel, xpos, xquat, xipos, S, bias);

    AbaFactor fac; float facscr[ABA_FAC_SCR];
    aba_factorize_damped(xquat, xipos, xpos, S, Bdt, facscr, &fac);

    // random RHS, K columns
    float rhs[KK*G1_NV];
    for (int i=0;i<KK*NV;++i) rhs[i]=2.0f*(curand_uniform(&st)*2.f-1.f);

    // sequential reference
    float seqscr[ABA_SOLVEM_SCR], out_seq[KK*G1_NV];
    for (int c=0;c<KK;++c) aba_solveM(&fac, S, &rhs[c*NV], seqscr, &out_seq[c*NV]);

    // multi-RHS
    float mscr[ABA_SOLVEM_MULTI_SCR(KK)], out_multi[KK*G1_NV];
    aba_solveM_multi<KK>(&fac, S, rhs, mscr, out_multi);

    float ma=0, sd=0, sr=0;
    for (int i=0;i<KK*NV;++i){
        float e=fabsf(out_multi[i]-out_seq[i]); if(e>ma) ma=e;
        sd+=(out_multi[i]-out_seq[i])*(out_multi[i]-out_seq[i]);
        sr+=out_seq[i]*out_seq[i];
    }
    float rel = sqrtf(sd)/(sqrtf(sr)+1e-12f);
    atomicMax((int*)maxabs, __float_as_int(ma));
    atomicMax((int*)maxrel, __float_as_int(rel));
}

int main(){
    int N=4096;
    float *dma,*dmr; cudaMalloc(&dma,4); cudaMalloc(&dmr,4);
    float z=0; cudaMemcpy(dma,&z,4,cudaMemcpyHostToDevice); cudaMemcpy(dmr,&z,4,cudaMemcpyHostToDevice);
    k<<<(N+63)/64,64>>>(N, 4242UL, dma, dmr);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("CUDA err %s\n", cudaGetErrorString(e)); return 1; }
    float ma,mr; cudaMemcpy(&ma,dma,4,cudaMemcpyDeviceToHost); cudaMemcpy(&mr,dmr,4,cudaMemcpyDeviceToHost);
    printf("aba_solveM_multi<%d> vs %d sequential aba_solveM over %d states:\n", KK,KK,N);
    printf("  max abs err = %.3e   max relerr = %.3e\n", ma, mr);
    printf("  %s\n", (mr < 1e-6f) ? "PASS (<=1e-6)" : "FAIL");
    return (mr < 1e-6f) ? 0 : 1;
}
