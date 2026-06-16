// M1 -- bulk-synchronous batched baseline. THE number to beat.
//
// One thread = one world. State (qpos,qvel) lives in global memory; the host issues
// one step-kernel launch per timestep (the bulk-synchronous pattern whose launch +
// global-traffic overhead the later persistent/fused kernel is trying to eliminate).
// Reuses the M0-validated device dynamics verbatim (src/dynamics.cuh).
//
// Usage: ./sim [nsteps] [nworlds]   (no nworlds -> throughput sweep)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "dynamics.cuh"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef BLOCK
#define BLOCK 64
#endif
#ifdef LB
__launch_bounds__(BLOCK)
#endif
// one bulk-synchronous physics step for all worlds
__global__ void step_kernel(float* qpos, float* qvel, int nworlds) {
    int w = blockIdx.x*blockDim.x + threadIdx.x;
    if (w >= nworlds) return;
    float* qp = qpos + (size_t)w*G1_NQ;
    float* qv = qvel + (size_t)w*G1_NV;

    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    float S[G1_NV*6], M[G1_NV*G1_NV], bias[G1_NV], nbias[G1_NV], qacc[G1_NV];
    const float dt = G1_DT;

    forward_kinematics(qp, xpos, xquat, xipos);
    compute_M_bias(qp, qv, xpos, xquat, xipos, S, M, bias);
    for (int i=0;i<G1_NV;++i) nbias[i] = -bias[i];
    chol_solve(M, nbias, qacc);
    for (int i=0;i<G1_NV;++i) qv[i] += dt*qacc[i];
    integrate_pos(qp, qv, dt);
}

static void read_init(double* qpos_d, double* qvel_d) {
    FILE* f = fopen("bench/init_state.bin", "rb");
    if (!f) { fprintf(stderr, "cannot open bench/init_state.bin (run `just ref`)\n"); exit(1); }
    if (fread(qpos_d, sizeof(double), G1_NQ, f) != G1_NQ ||
        fread(qvel_d, sizeof(double), G1_NV, f) != G1_NV) { fprintf(stderr,"short read\n"); exit(1); }
    fclose(f);
}

// fill N worlds with identical init state
static void init_worlds(float* d_qpos, float* d_qvel, int N,
                        const float* qpos0, const float* qvel0) {
    std::vector<float> hp((size_t)N*G1_NQ), hv((size_t)N*G1_NV);
    for (int w=0; w<N; ++w) {
        for (int i=0;i<G1_NQ;++i) hp[(size_t)w*G1_NQ+i]=qpos0[i];
        for (int i=0;i<G1_NV;++i) hv[(size_t)w*G1_NV+i]=qvel0[i];
    }
    CK(cudaMemcpy(d_qpos, hp.data(), hp.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_qvel, hv.data(), hv.size()*sizeof(float), cudaMemcpyHostToDevice));
}

// run nsteps, return elapsed ms (host issues one launch per step = bulk-synchronous)
static float run(float* d_qpos, float* d_qvel, int N, int nsteps) {
    const int TPB = BLOCK;
    int blocks = (N + TPB - 1) / TPB;
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int s=0; s<nsteps; ++s) step_kernel<<<blocks, TPB>>>(d_qpos, d_qvel, N);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    CK(cudaGetLastError());
    return ms;
}

int main(int argc, char** argv) {
    int nsteps = (argc>1) ? atoi(argv[1]) : 300;
    double qpos_d[G1_NQ], qvel_d[G1_NV];
    read_init(qpos_d, qvel_d);
    float qpos0[G1_NQ], qvel0[G1_NV];
    for (int i=0;i<G1_NQ;++i) qpos0[i]=(float)qpos_d[i];
    for (int i=0;i<G1_NV;++i) qvel0[i]=(float)qvel_d[i];

    if (argc > 2) {
        // single run: validate world 0 + determinism, dump final state for the test
        int N = atoi(argv[2]);
        float *d_qpos, *d_qvel;
        CK(cudaMalloc(&d_qpos, (size_t)N*G1_NQ*sizeof(float)));
        CK(cudaMalloc(&d_qvel, (size_t)N*G1_NV*sizeof(float)));
        init_worlds(d_qpos, d_qvel, N, qpos0, qvel0);
        float ms = run(d_qpos, d_qvel, N, nsteps);
        std::vector<float> hp((size_t)N*G1_NQ);
        CK(cudaMemcpy(hp.data(), d_qpos, hp.size()*sizeof(float), cudaMemcpyDeviceToHost));
        // determinism: all worlds equal world 0
        double maxdiff = 0;
        for (int w=1; w<N; ++w)
            for (int i=0;i<G1_NQ;++i)
                maxdiff = fmax(maxdiff, fabs(hp[(size_t)w*G1_NQ+i]-hp[i]));
        FILE* o = fopen("bench/sim_final.bin", "wb");  // world-0 final qpos
        fwrite(hp.data(), sizeof(float), G1_NQ, o); fclose(o);
        double esps = (double)N*nsteps/(ms/1e3);
        printf("N=%d steps=%d  %.2f ms  %.3e env-steps/s  world-determinism maxdiff=%.2e\n",
               N, nsteps, ms, esps, maxdiff);
        cudaFree(d_qpos); cudaFree(d_qvel);
        return 0;
    }

    // throughput sweep
    int Ns[] = {1024, 4096, 16384, 65536, 262144};
    printf("bulk-synchronous baseline (1 thread/world, per-step launch), %d steps:\n", nsteps);
    for (int N : Ns) {
        float *d_qpos, *d_qvel;
        CK(cudaMalloc(&d_qpos, (size_t)N*G1_NQ*sizeof(float)));
        CK(cudaMalloc(&d_qvel, (size_t)N*G1_NV*sizeof(float)));
        init_worlds(d_qpos, d_qvel, N, qpos0, qvel0);
        run(d_qpos, d_qvel, N, 20);                  // warmup
        init_worlds(d_qpos, d_qvel, N, qpos0, qvel0);
        float ms = run(d_qpos, d_qvel, N, nsteps);
        double esps = (double)N*nsteps/(ms/1e3);
        printf("  N=%7d  %8.2f ms  %.3e env-steps/s  (%.1f us/step)\n",
               N, ms, esps, ms*1e3/nsteps);
        cudaFree(d_qpos); cudaFree(d_qvel);
    }
    return 0;
}
