// Validate compute_bias (RNE-only, no dense M) == compute_M_bias bias (the MuJoCo-validated
// path in dynamics.cuh) over random states. If they match, compute_bias inherits the M0
// MuJoCo validation. Prints max abs + relerr of (compute_bias - compute_M_bias) bias.
//   nvcc -arch=sm_86 -O3 src/test_bias.cu -o build/test_bias && ./build/test_bias
#include <cstdio>
#include <cmath>
#include <curand_kernel.h>
#include "dynamics.cuh"
#include "aba_bias.cuh"

__global__ void k(int Nstate, unsigned long seed, float* maxabs, float* maxrel) {
    int w = blockIdx.x*blockDim.x + threadIdx.x; if (w >= Nstate) return;
    curandState st; curand_init(seed, w, 0, &st);
    float qpos[G1_NQ], qvel[G1_NV];
    for (int i=0;i<G1_NQ;++i) qpos[i] = g1_qpos_stand[i] + 0.3f*(curand_uniform(&st)*2.f-1.f);
    // renormalize base quat (qpos[3:7])
    float n=0; for(int i=3;i<7;++i) n+=qpos[i]*qpos[i]; n=sqrtf(n);
    for(int i=3;i<7;++i) qpos[i]/=n;
    for (int i=0;i<G1_NV;++i) qvel[i] = 1.5f*(curand_uniform(&st)*2.f-1.f);

    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    forward_kinematics(qpos, xpos, xquat, xipos);

    float S1[G1_NV*6], S2[G1_NV*6];
    float M[G1_NV*G1_NV], bias_ref[G1_NV], bias_new[G1_NV];
    compute_M_bias(qpos, qvel, xpos, xquat, xipos, S1, M, bias_ref);
    compute_bias(qpos, qvel, xpos, xquat, xipos, S2, bias_new);

    float ma=0, sd=0, sr=0;
    for (int i=0;i<G1_NV;++i){
        float e=fabsf(bias_new[i]-bias_ref[i]); if(e>ma) ma=e;
        sd+=(bias_new[i]-bias_ref[i])*(bias_new[i]-bias_ref[i]);
        sr+=bias_ref[i]*bias_ref[i];
    }
    float rel = sqrtf(sd)/(sqrtf(sr)+1e-12f);
    atomicMax((int*)maxabs, __float_as_int(ma));
    atomicMax((int*)maxrel, __float_as_int(rel));
}

int main(){
    int N=4096;
    float *dma,*dmr; cudaMalloc(&dma,4); cudaMalloc(&dmr,4);
    float z=0; cudaMemcpy(dma,&z,4,cudaMemcpyHostToDevice); cudaMemcpy(dmr,&z,4,cudaMemcpyHostToDevice);
    k<<<(N+63)/64,64>>>(N, 12345UL, dma, dmr);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("CUDA err %s\n", cudaGetErrorString(e)); return 1; }
    float ma,mr; cudaMemcpy(&ma,dma,4,cudaMemcpyDeviceToHost); cudaMemcpy(&mr,dmr,4,cudaMemcpyDeviceToHost);
    printf("compute_bias vs compute_M_bias over %d random states:\n", N);
    printf("  max abs err = %.3e   max relerr = %.3e\n", ma, mr);
    printf("  %s\n", (mr < 1e-5f) ? "PASS (matches validated bias)" : "FAIL");
    return (mr < 1e-5f) ? 0 : 1;
}
