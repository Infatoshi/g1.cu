// 1-thread/world ABA (full SIMT) with the per-thread WORKING SET in shared memory.
// Not cooperative -- each thread still does one whole world (SIMT preserved). The big
// ABA scratch (IA/vel/cb/pA/a6) + S live in SMEM instead of local memory, so the
// per-step recompute never spills to DRAM. Trade-off: SMEM caps threads/SM (occupancy)
// vs on-chip speed. Multi-step keeps qpos/qvel resident across steps. Measures whether
// on-chip-at-low-occupancy beats DRAM-bound-at-high-occupancy. Usage: same as sim_aba.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "aba.cuh"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef TPB
#define TPB 64
#endif
// per-thread SMEM footprint: scratch + S (xpos/xquat/xipos stay local)
#define PERTHREAD (ABA_SCR + G1_NV*6)

__global__ void step_kernel_smem(float* qpos, float* qvel, int nworlds, int ksteps) {
    extern __shared__ float smem[];
    int w = blockIdx.x*blockDim.x + threadIdx.x;
    float* myscr = smem + (size_t)threadIdx.x*PERTHREAD;
    float* scr = myscr;            // ABA_SCR floats
    float* S   = myscr + ABA_SCR;  // NV*6 floats
    if (w >= nworlds) return;

    float* gqp = qpos + (size_t)w*G1_NQ;
    float* gqv = qvel + (size_t)w*G1_NV;
    float qp[G1_NQ], qv[G1_NV];
    for (int i=0;i<G1_NQ;++i) qp[i]=gqp[i];
    for (int i=0;i<G1_NV;++i) qv[i]=gqv[i];
    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3], qacc[G1_NV];
    const float dt=G1_DT;
    for (int s=0;s<ksteps;++s){
        forward_kinematics(qp, xpos, xquat, xipos);
        aba_qacc(qp, qv, xpos, xquat, xipos, S, scr, qacc);
        for (int i=0;i<G1_NV;++i) qv[i]+=dt*qacc[i];
        integrate_pos(qp, qv, dt);
    }
    for (int i=0;i<G1_NQ;++i) gqp[i]=qp[i];
    for (int i=0;i<G1_NV;++i) gqv[i]=qv[i];
}

static void read_init(double* qp,double* qv){ FILE* f=fopen("bench/init_state.bin","rb");
    if(!f){fprintf(stderr,"run just ref\n");exit(1);} fread(qp,8,G1_NQ,f); fread(qv,8,G1_NV,f); fclose(f); }
static void init_worlds(float* dqp,float* dqv,int N,const float* q0,const float* v0){
    std::vector<float> hp((size_t)N*G1_NQ),hv((size_t)N*G1_NV);
    for(int w=0;w<N;++w){ for(int i=0;i<G1_NQ;++i)hp[(size_t)w*G1_NQ+i]=q0[i]; for(int i=0;i<G1_NV;++i)hv[(size_t)w*G1_NV+i]=v0[i]; }
    CK(cudaMemcpy(dqp,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqv,hv.data(),hv.size()*4,cudaMemcpyHostToDevice)); }

static float run(float* dqp,float* dqv,int N,int nsteps,int ksteps,int tpb){
    int blocks=(N+tpb-1)/tpb; int sm=tpb*PERTHREAD*sizeof(float);
    CK(cudaFuncSetAttribute(step_kernel_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); CK(cudaEventRecord(a));
    for(int s=0;s<nsteps;s+=ksteps){ int k=min(ksteps,nsteps-s); step_kernel_smem<<<blocks,tpb,sm>>>(dqp,dqv,N,k); }
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b)); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); CK(cudaGetLastError());
    return ms; }

int main(int argc,char**argv){
    int nsteps=(argc>1)?atoi(argv[1]):300;
    double qpd[G1_NQ],qvd[G1_NV]; read_init(qpd,qvd);
    float q0[G1_NQ],v0[G1_NV]; for(int i=0;i<G1_NQ;++i)q0[i]=(float)qpd[i]; for(int i=0;i<G1_NV;++i)v0[i]=(float)qvd[i];
    printf("smem-scratch: %d floats/thread (%lu B)\n", PERTHREAD, PERTHREAD*sizeof(float));
    if (argc>2){ int N=atoi(argv[2]); int ks=(argc>3)?atoi(argv[3]):1; int tpb=(argc>4)?atoi(argv[4]):TPB;
        float *dqp,*dqv; CK(cudaMalloc(&dqp,(size_t)N*G1_NQ*4)); CK(cudaMalloc(&dqv,(size_t)N*G1_NV*4));
        init_worlds(dqp,dqv,N,q0,v0); float ms=run(dqp,dqv,N,nsteps,ks,tpb);
        std::vector<float> hp((size_t)N*G1_NQ); CK(cudaMemcpy(hp.data(),dqp,hp.size()*4,cudaMemcpyDeviceToHost));
        double md=0; for(int w=1;w<N;++w)for(int i=0;i<G1_NQ;++i) md=fmax(md,fabs(hp[(size_t)w*G1_NQ+i]-hp[i]));
        FILE* o=fopen("bench/sim_smem_final.bin","wb"); fwrite(hp.data(),4,G1_NQ,o); fclose(o);
        printf("smem N=%d steps=%d ksteps=%d tpb=%d  %.2f ms  %.3e env-steps/s  determinism=%.1e\n",N,nsteps,ks,tpb,ms,(double)N*nsteps/(ms/1e3),md);
        return 0; }
    // sweep threads/block (occupancy vs SMEM) x ksteps
    int tpbs[]={4,8,12}; int Ks[]={1,16};
    int N=65536;
    printf("smem-scratch 1-thread/world, N=%d, %d steps:\n",N,nsteps);
    float *dqp,*dqv; CK(cudaMalloc(&dqp,(size_t)N*G1_NQ*4)); CK(cudaMalloc(&dqv,(size_t)N*G1_NV*4));
    for(int tpb:tpbs){ size_t sm=(size_t)tpb*PERTHREAD*4; if(sm>101376){printf("  tpb=%d SMEM %luB too big, skip\n",tpb,sm);continue;}
        for(int ks:Ks){ init_worlds(dqp,dqv,N,q0,v0); run(dqp,dqv,N,20,ks,tpb);
            init_worlds(dqp,dqv,N,q0,v0); float ms=run(dqp,dqv,N,nsteps,ks,tpb);
            printf("  tpb=%3d ksteps=%2d  %8.2f ms  %.3e env-steps/s\n",tpb,ks,ms,(double)N*nsteps/(ms/1e3)); } }
    return 0;
}
