// M0 single-world harness: rolls out the validated device dynamics for one world
// and dumps step-0 M/bias/qacc + the qpos trajectory for tests/test_m0.py.
#include <cstdio>
#include <cstdlib>
#include "dynamics.cuh"

// ---------------- M0 kernel: full smooth-dynamics rollout ----------------
__global__ void m0_kernel(const float* qpos_in, const float* qvel_in, int nsteps,
                          float* Mout0, float* bias0, float* qacc0, float* traj) {
    if (threadIdx.x || blockIdx.x) return;
    float qpos[G1_NQ], qvel[G1_NV];
    for (int i=0;i<G1_NQ;++i) qpos[i]=qpos_in[i];
    for (int i=0;i<G1_NV;++i) qvel[i]=qvel_in[i];

    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    __shared__ float S[G1_NV*6];
    float M[G1_NV*G1_NV], bias[G1_NV], nbias[G1_NV], qacc[G1_NV];
    const float dt=G1_DT;

    for (int i=0;i<G1_NQ;++i) traj[i]=qpos[i];   // step 0
    for (int step=0; step<nsteps; ++step){
        forward_kinematics(qpos, xpos, xquat, xipos);
        compute_M_bias(qpos, qvel, xpos, xquat, xipos, S, M, bias);
        if (step==0){ for(int i=0;i<G1_NV*G1_NV;++i)Mout0[i]=M[i]; for(int i=0;i<G1_NV;++i)bias0[i]=bias[i]; }
        for (int i=0;i<G1_NV;++i) nbias[i]=-bias[i];
        chol_solve(M, nbias, qacc);                 // qacc = M^-1 (-bias)
        if (step==0){ for(int i=0;i<G1_NV;++i)qacc0[i]=qacc[i]; }
        for (int i=0;i<G1_NV;++i) qvel[i]+=dt*qacc[i];   // semi-implicit Euler
        integrate_pos(qpos, qvel, dt);
        for (int i=0;i<G1_NQ;++i) traj[(step+1)*G1_NQ+i]=qpos[i];
    }
}

// ---------------- host harness ----------------
#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

int main(int argc, char** argv) {
    int nsteps = (argc>1) ? atoi(argv[1]) : 300;
    // read init state (float64: qpos[NQ], qvel[NV])
    FILE* f = fopen("bench/init_state.bin", "rb");
    if (!f) { fprintf(stderr, "cannot open bench/init_state.bin\n"); return 1; }
    double qpos_d[G1_NQ], qvel_d[G1_NV];
    if (fread(qpos_d, sizeof(double), G1_NQ, f) != G1_NQ ||
        fread(qvel_d, sizeof(double), G1_NV, f) != G1_NV) {
        fprintf(stderr, "short read on init_state.bin\n"); return 1;
    }
    fclose(f);
    float qpos_h[G1_NQ], qvel_h[G1_NV];
    for (int i = 0; i < G1_NQ; ++i) qpos_h[i] = (float)qpos_d[i];
    for (int i = 0; i < G1_NV; ++i) qvel_h[i] = (float)qvel_d[i];

    float *d_qpos, *d_qvel, *d_M, *d_bias, *d_qacc, *d_traj;
    CK(cudaMalloc(&d_qpos, G1_NQ*sizeof(float)));
    CK(cudaMalloc(&d_qvel, G1_NV*sizeof(float)));
    CK(cudaMalloc(&d_M, G1_NV*G1_NV*sizeof(float)));
    CK(cudaMalloc(&d_bias, G1_NV*sizeof(float)));
    CK(cudaMalloc(&d_qacc, G1_NV*sizeof(float)));
    CK(cudaMalloc(&d_traj, (size_t)(nsteps+1)*G1_NQ*sizeof(float)));
    CK(cudaMemcpy(d_qpos, qpos_h, G1_NQ*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_qvel, qvel_h, G1_NV*sizeof(float), cudaMemcpyHostToDevice));

    m0_kernel<<<1,1>>>(d_qpos, d_qvel, nsteps, d_M, d_bias, d_qacc, d_traj);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    float M_h[G1_NV*G1_NV], bias_h[G1_NV], qacc_h[G1_NV];
    float* traj_h = (float*)malloc((size_t)(nsteps+1)*G1_NQ*sizeof(float));
    CK(cudaMemcpy(M_h, d_M, sizeof(M_h), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bias_h, d_bias, sizeof(bias_h), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(qacc_h, d_qacc, sizeof(qacc_h), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(traj_h, d_traj, (size_t)(nsteps+1)*G1_NQ*sizeof(float), cudaMemcpyDeviceToHost));

    FILE* o = fopen("bench/cuda_fk.bin", "wb");   // step-0 diagnostics
    fwrite(M_h, sizeof(float), G1_NV*G1_NV, o);
    fwrite(bias_h, sizeof(float), G1_NV, o);
    fwrite(qacc_h, sizeof(float), G1_NV, o);
    fclose(o);
    FILE* t = fopen("bench/cuda_traj.bin", "wb");  // qpos trajectory
    fwrite(traj_h, sizeof(float), (size_t)(nsteps+1)*G1_NQ, t);
    fclose(t);
    free(traj_h);
    printf("M0 rollout done: %d steps -> bench/cuda_traj.bin\n", nsteps);
    return 0;
}
