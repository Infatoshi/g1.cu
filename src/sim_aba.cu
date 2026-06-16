// Batched ABA sim with multi-step on-chip looping.
// One thread = one world. Each launch: load world state once, run KSTEPS steps with
// state resident in registers/L1 (FK -> ABA -> integrate), write back once. ABA's
// O(N) recursion has a ~35% smaller working set than the dense kernel and far shorter
// dependency chains. Validated vs MuJoCo (world 0). Usage: ./sim_aba [nsteps] [nworlds]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "aba.cuh"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef BLOCK
#define BLOCK 64
#endif

// run ksteps in-kernel, world state resident in local registers/L1
__global__ void step_kernel_aba(float* qpos, float* qvel, int nworlds, int ksteps) {
    int w = blockIdx.x*blockDim.x + threadIdx.x;
    if (w >= nworlds) return;
    float* gqp = qpos + (size_t)w*G1_NQ;
    float* gqv = qvel + (size_t)w*G1_NV;
    float qp[G1_NQ], qv[G1_NV];
    for (int i=0;i<G1_NQ;++i) qp[i]=gqp[i];
    for (int i=0;i<G1_NV;++i) qv[i]=gqv[i];

    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    float S[G1_NV*6], qacc[G1_NV], scr[ABA_SCR];
    const float dt = G1_DT;
    for (int s=0;s<ksteps;++s){
        forward_kinematics(qp, xpos, xquat, xipos);
        aba_qacc(qp, qv, xpos, xquat, xipos, S, scr, qacc);
        for (int i=0;i<G1_NV;++i) qv[i] += dt*qacc[i];
        integrate_pos(qp, qv, dt);
    }
    for (int i=0;i<G1_NQ;++i) gqp[i]=qp[i];
    for (int i=0;i<G1_NV;++i) gqv[i]=qv[i];
}

static void read_init(double* qpos_d, double* qvel_d){
    FILE* f=fopen("bench/init_state.bin","rb");
    if(!f){fprintf(stderr,"run `just ref`\n");exit(1);}
    fread(qpos_d,8,G1_NQ,f); fread(qvel_d,8,G1_NV,f); fclose(f);
}
static void init_worlds(float* dqp,float* dqv,int N,const float* q0,const float* v0){
    std::vector<float> hp((size_t)N*G1_NQ),hv((size_t)N*G1_NV);
    for(int w=0;w<N;++w){ for(int i=0;i<G1_NQ;++i)hp[(size_t)w*G1_NQ+i]=q0[i];
                          for(int i=0;i<G1_NV;++i)hv[(size_t)w*G1_NV+i]=v0[i]; }
    CK(cudaMemcpy(dqp,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqv,hv.data(),hv.size()*4,cudaMemcpyHostToDevice));
}
// run nsteps total via launches of KSTEPS each; return elapsed ms
static float run(float* dqp,float* dqv,int N,int nsteps,int ksteps){
    int blocks=(N+BLOCK-1)/BLOCK;
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for(int s=0;s<nsteps;s+=ksteps){ int k=min(ksteps,nsteps-s); step_kernel_aba<<<blocks,BLOCK>>>(dqp,dqv,N,k); }
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); CK(cudaGetLastError());
    return ms;
}

int main(int argc,char**argv){
    int nsteps=(argc>1)?atoi(argv[1]):300;
    double qpd[G1_NQ],qvd[G1_NV]; read_init(qpd,qvd);
    float q0[G1_NQ],v0[G1_NV];
    for(int i=0;i<G1_NQ;++i)q0[i]=(float)qpd[i]; for(int i=0;i<G1_NV;++i)v0[i]=(float)qvd[i];

    if (argc>2){ // single run: validate world 0 + determinism
        int N=atoi(argv[2]); int ks=(argc>3)?atoi(argv[3]):1;
        float *dqp,*dqv; CK(cudaMalloc(&dqp,(size_t)N*G1_NQ*4)); CK(cudaMalloc(&dqv,(size_t)N*G1_NV*4));
        init_worlds(dqp,dqv,N,q0,v0);
        float ms=run(dqp,dqv,N,nsteps,ks);
        std::vector<float> hp((size_t)N*G1_NQ); CK(cudaMemcpy(hp.data(),dqp,hp.size()*4,cudaMemcpyDeviceToHost));
        double md=0; for(int w=1;w<N;++w) for(int i=0;i<G1_NQ;++i) md=fmax(md,fabs(hp[(size_t)w*G1_NQ+i]-hp[i]));
        FILE* o=fopen("bench/sim_aba_final.bin","wb"); fwrite(hp.data(),4,G1_NQ,o); fclose(o);
        printf("ABA N=%d steps=%d ksteps=%d  %.2f ms  %.3e env-steps/s  determinism=%.1e\n",
               N,nsteps,ks,ms,(double)N*nsteps/(ms/1e3),md);
        return 0;
    }
    // sweep: world counts x ksteps
    int Ns[]={4096,16384,65536,262144};
    int Ks[]={1,4,16};
    printf("ABA batched sim (1 thread/world, multi-step on-chip), %d steps:\n",nsteps);
    for(int N:Ns){
        float *dqp,*dqv; CK(cudaMalloc(&dqp,(size_t)N*G1_NQ*4)); CK(cudaMalloc(&dqv,(size_t)N*G1_NV*4));
        for(int ks:Ks){
            init_worlds(dqp,dqv,N,q0,v0); run(dqp,dqv,N,20,ks);          // warmup
            init_worlds(dqp,dqv,N,q0,v0); float ms=run(dqp,dqv,N,nsteps,ks);
            printf("  N=%7d ksteps=%2d  %8.2f ms  %.3e env-steps/s\n",N,ks,ms,(double)N*nsteps/(ms/1e3));
        }
        cudaFree(dqp); cudaFree(dqv);
    }
    return 0;
}
