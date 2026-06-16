// Stage A accuracy harness: replay a controlled trajectory through the IMPLICIT integrator
// (env_qacc's implicit-damping path) for ONE world with contacts inactive (base in air), and
// dump qpos/qvel per step. scripts/test_implicit.py compares against MuJoCo implicitfast.
//
// Reuses env_qacc verbatim from g1_env.cu logic; we re-declare the minimal pieces here to
// avoid the extern-C env wrapper. PD torque = kp*(target-q) - kd*qd, dt*kd folded into D.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "dynamics.cuh"
#include "aba.cuh"
#include "aba_factor.cuh"
#include "contact.cuh"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define ACT_DIM 29

// --- copy of g1_env.cu env_qacc (implicit-damping path) ---
__device__ void env_qacc_t(const float* qp, const float* qv,
                           const float* xpos, const float* xquat, const float* xipos,
                           const float* tau, const float* Bdt, float* S, float* qacc, int pgs_iters){
    const int NV=G1_NV;
    float abascr[ABA_SCR], qacc_s[G1_NV];
    aba_qacc(qp, qv, xpos, xquat, xipos, S, abascr, qacc_s);
    float negbias[G1_NV]; float mulscr[ABA_MULM_SCR];
    aba_mulM(xquat, xipos, xpos, S, qacc_s, mulscr, negbias);
    AbaFactor fac; float facscr[ABA_FAC_SCR];
    aba_factorize_damped(xquat, xipos, xpos, S, Bdt, facscr, &fac);
    float solvescr[ABA_SOLVEM_SCR];
    float rhs[G1_NV];
    for (int i=0;i<NV;++i) rhs[i]=negbias[i]+tau[i];
    aba_solveM(&fac, S, rhs, solvescr, qacc_s);
    Contact cons[G1_MAX_CONTACT];
    int ncon = detect_foot_contacts(xpos, xquat, cons);
    if (ncon==0){ for(int i=0;i<NV;++i) qacc[i]=qacc_s[i]; return; }
    // (contacts not expected in the air test; full path lives in g1_env.cu)
    for(int i=0;i<NV;++i) qacc[i]=qacc_s[i];
}

__global__ void k_traj(float* qp0, float* qv0, const float* targets, int nsteps,
                       const float* Bdt, float* qp_out, float* qv_out){
    float lp[G1_NQ], lv[G1_NV];
    for(int i=0;i<G1_NQ;++i) lp[i]=qp0[i];
    for(int i=0;i<G1_NV;++i) lv[i]=qv0[i];
    for(int i=0;i<G1_NQ;++i) qp_out[i]=lp[i];
    for(int i=0;i<G1_NV;++i) qv_out[i]=lv[i];
    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    float Sm[G1_NV*6], qacc[G1_NV], tau[G1_NV];
    const float dt=G1_DT;
    for(int t=0;t<nsteps;++t){
        const float* tgt=&targets[t*ACT_DIM];
        forward_kinematics(lp, xpos, xquat, xipos);
        for(int i=0;i<G1_NV;++i) tau[i]=0.f;
        for(int i=0;i<ACT_DIM;++i){
            int d=g1_act_dof[i];
            float pdt=g1_act_kp[i]*(tgt[i]-lp[g1_act_qadr[i]]) - g1_act_kd[i]*lv[d];
            if (pdt<g1_act_frc_lo[i]) pdt=g1_act_frc_lo[i];
            else if (pdt>g1_act_frc_hi[i]) pdt=g1_act_frc_hi[i];
            tau[d]=pdt;
        }
        env_qacc_t(lp, lv, xpos, xquat, xipos, tau, Bdt, Sm, qacc, 0);
        for(int i=0;i<G1_NV;++i) lv[i]+=dt*qacc[i];
        integrate_pos(lp, lv, dt);
        for(int i=0;i<G1_NQ;++i) qp_out[(t+1)*G1_NQ+i]=lp[i];
        for(int i=0;i<G1_NV;++i) qv_out[(t+1)*G1_NV+i]=lv[i];
    }
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s implicit_ref.bin\n",argv[0]); return 1; }
    FILE* f=fopen(argv[1],"rb"); if(!f){ perror("open"); return 1; }
    int nsteps; double dt_d;
    fread(&nsteps,sizeof(int),1,f); fread(&dt_d,sizeof(double),1,f);
    // read qpos0 (NQ f64), qvel0 (NV f64), Bdiag (NV f64), targets (nsteps*ACT f64)
    double* buf=(double*)malloc(sizeof(double)*(G1_NQ+G1_NV+G1_NV+nsteps*ACT_DIM));
    fread(buf,sizeof(double),G1_NQ+G1_NV+G1_NV+nsteps*ACT_DIM,f); fclose(f);
    float hqp[G1_NQ],hqv[G1_NV],hBdt[G1_NV];
    int o=0;
    for(int i=0;i<G1_NQ;++i) hqp[i]=(float)buf[o++];
    for(int i=0;i<G1_NV;++i) hqv[i]=(float)buf[o++];
    for(int i=0;i<G1_NV;++i) hBdt[i]=(float)(buf[o++]*dt_d);  // Bdiag*dt
    float* htg=(float*)malloc(sizeof(float)*nsteps*ACT_DIM);
    for(int i=0;i<nsteps*ACT_DIM;++i) htg[i]=(float)buf[o++];

    float *dqp0,*dqv0,*dtg,*dBdt,*dqpo,*dqvo;
    CK(cudaMalloc(&dqp0,sizeof(float)*G1_NQ)); CK(cudaMalloc(&dqv0,sizeof(float)*G1_NV));
    CK(cudaMalloc(&dtg,sizeof(float)*nsteps*ACT_DIM)); CK(cudaMalloc(&dBdt,sizeof(float)*G1_NV));
    CK(cudaMalloc(&dqpo,sizeof(float)*(nsteps+1)*G1_NQ)); CK(cudaMalloc(&dqvo,sizeof(float)*(nsteps+1)*G1_NV));
    CK(cudaMemcpy(dqp0,hqp,sizeof(float)*G1_NQ,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqv0,hqv,sizeof(float)*G1_NV,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtg,htg,sizeof(float)*nsteps*ACT_DIM,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBdt,hBdt,sizeof(float)*G1_NV,cudaMemcpyHostToDevice));
    k_traj<<<1,1>>>(dqp0,dqv0,dtg,nsteps,dBdt,dqpo,dqvo);
    CK(cudaDeviceSynchronize());
    float* hqpo=(float*)malloc(sizeof(float)*(nsteps+1)*G1_NQ);
    float* hqvo=(float*)malloc(sizeof(float)*(nsteps+1)*G1_NV);
    CK(cudaMemcpy(hqpo,dqpo,sizeof(float)*(nsteps+1)*G1_NQ,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hqvo,dqvo,sizeof(float)*(nsteps+1)*G1_NV,cudaMemcpyDeviceToHost));
    FILE* of=fopen("bench/implicit_cuda.bin","wb");
    fwrite(&nsteps,sizeof(int),1,of);
    fwrite(hqpo,sizeof(float),(nsteps+1)*G1_NQ,of);
    fwrite(hqvo,sizeof(float),(nsteps+1)*G1_NV,of);
    fclose(of);
    printf("wrote bench/implicit_cuda.bin: %d steps\n", nsteps);
    return 0;
}
