// Numerical-equivalence gate for env_qacc optimization. Over a fixed batch of random
// (qpos,qvel) states with a FIXED seed, fixed zero tau, and the standard implicit Bdt, run
// forward_kinematics then env_qacc and dump the resulting qacc for all states to a binary file.
//
//   build:  /opt/cuda/bin/nvcc -arch=sm_86 -O3 src/test_envqacc.cu -lcurand -o build/test_envqacc
//   ref:    ./build/test_envqacc gen     -> writes build/envqacc_ref.bin (run on KNOWN-GOOD code)
//   check:  ./build/test_envqacc          -> loads ref, prints max abs err + max relerr vs ref
//
// Gate: after every env_qacc change, max relerr vs ref must stay <= 1e-5 (the current env_qacc
// is itself validated to the MuJoCo oracle at qacc relerr ~2.24e-5, so reproducing it to 1e-5
// preserves that validation).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <curand_kernel.h>
#include "dynamics.cuh"
#include "aba.cuh"
#include "aba_factor.cuh"
#include "aba_bias.cuh"
#include "contact.cuh"
#include "env_qacc.cuh"
#include "g1_model.h"

#define NSTATE 4096
#define ACT_DIM 29

// mirrors env_advance's Bdt build (implicit kd damping diagonal) and zero tau.
__global__ void k(int Nstate, unsigned long seed, float* qacc_out /*[Nstate*G1_NV]*/) {
    int w = blockIdx.x*blockDim.x + threadIdx.x; if (w >= Nstate) return;
    curandState st; curand_init(seed, w, 0, &st);
    float qpos[G1_NQ], qvel[G1_NV];
    // perturb the standing pose so a meaningful fraction of states have foot contact
    for (int i=0;i<G1_NQ;++i) qpos[i] = g1_qpos_stand[i] + 0.20f*(curand_uniform(&st)*2.f-1.f);
    // drop the base a little so feet penetrate the ground in many samples (exercise contact path)
    qpos[2] -= 0.15f*curand_uniform(&st);
    float n=0; for(int i=3;i<7;++i) n+=qpos[i]*qpos[i]; n=sqrtf(n);
    for(int i=3;i<7;++i) qpos[i]/=n;
    for (int i=0;i<G1_NV;++i) qvel[i] = 1.0f*(curand_uniform(&st)*2.f-1.f);

    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    forward_kinematics(qpos, xpos, xquat, xipos);

    float tau[G1_NV]; for (int i=0;i<G1_NV;++i) tau[i]=0.f;
    float Bdt[G1_NV]; for (int i=0;i<G1_NV;++i) Bdt[i]=0.f;
    for (int i=0;i<ACT_DIM;++i) Bdt[g1_act_dof[i]] = G1_DT*g1_act_kd[i];

    float Sm[G1_NV*6], qacc[G1_NV];
    env_qacc(qpos, qvel, xpos, xquat, xipos, tau, Bdt, Sm, qacc, 8);
    for (int i=0;i<G1_NV;++i) qacc_out[(size_t)w*G1_NV+i] = qacc[i];
}

int main(int argc, char** argv){
    bool gen = (argc>1 && strcmp(argv[1],"gen")==0);
    int N=NSTATE;
    float* dq; cudaMalloc(&dq,(size_t)N*G1_NV*4);
    k<<<(N+63)/64,64>>>(N, 999UL, dq);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("CUDA err %s\n", cudaGetErrorString(e)); return 1; }
    float* hq=(float*)malloc((size_t)N*G1_NV*4);
    cudaMemcpy(hq,dq,(size_t)N*G1_NV*4,cudaMemcpyDeviceToHost);

    const char* path="build/envqacc_ref.bin";
    if (gen){
        FILE* f=fopen(path,"wb"); if(!f){ printf("cannot write %s\n",path); return 1; }
        fwrite(hq,4,(size_t)N*G1_NV,f); fclose(f);
        printf("wrote reference %s (%d states x %d dof)\n", path, N, G1_NV);
        return 0;
    }
    FILE* f=fopen(path,"rb");
    if(!f){ printf("no reference %s -- run './build/test_envqacc gen' first\n",path); return 1; }
    float* ref=(float*)malloc((size_t)N*G1_NV*4);
    size_t rd=fread(ref,4,(size_t)N*G1_NV,f); fclose(f);
    if (rd != (size_t)N*G1_NV){ printf("ref size mismatch\n"); return 1; }
    // per-state relerr = ||new-ref||_2 / (||ref||_2 + eps), report the worst state
    float maxabs=0.f, maxrel=0.f;
    for (int w=0;w<N;++w){
        float sd=0.f, sr=0.f;
        for (int i=0;i<G1_NV;++i){
            float a=hq[(size_t)w*G1_NV+i], b=ref[(size_t)w*G1_NV+i];
            float er=fabsf(a-b); if(er>maxabs) maxabs=er;
            sd+=(a-b)*(a-b); sr+=b*b;
        }
        float rel=sqrtf(sd)/(sqrtf(sr)+1e-12f);
        if(rel>maxrel) maxrel=rel;
    }
    printf("env_qacc vs reference over %d random states:\n", N);
    printf("  max abs err = %.3e   max relerr = %.3e\n", maxabs, maxrel);
    printf("  %s\n", (maxrel < 1e-5f) ? "PASS (<=1e-5)" : "FAIL (>1e-5)");
    return (maxrel < 1e-5f) ? 0 : 1;
}
