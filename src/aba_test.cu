// Validate ABA forward dynamics vs MuJoCo oracle: dump qacc[0] + qpos trajectory.
#include <cstdio>
#include <cstdlib>
#include "aba.cuh"

#define CK(c) do{cudaError_t e=(c); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void aba_roll(const float* qpos_in, const float* qvel_in, int nsteps,
                         float* qacc0, float* traj) {
    if (threadIdx.x||blockIdx.x) return;
    float qpos[G1_NQ], qvel[G1_NV], xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    float S[G1_NV*6], qacc[G1_NV], scr[ABA_SCR];
    for(int i=0;i<G1_NQ;++i) qpos[i]=qpos_in[i];
    for(int i=0;i<G1_NV;++i) qvel[i]=qvel_in[i];
    const float dt=G1_DT;
    for(int i=0;i<G1_NQ;++i) traj[i]=qpos[i];
    for(int s=0;s<nsteps;++s){
        forward_kinematics(qpos,xpos,xquat,xipos);
        aba_qacc(qpos,qvel,xpos,xquat,xipos,S,scr,qacc);
        if(s==0) for(int i=0;i<G1_NV;++i) qacc0[i]=qacc[i];
        for(int i=0;i<G1_NV;++i) qvel[i]+=dt*qacc[i];
        integrate_pos(qpos,qvel,dt);
        for(int i=0;i<G1_NQ;++i) traj[(s+1)*G1_NQ+i]=qpos[i];
    }
}

int main(int argc,char**argv){
    int nsteps=(argc>1)?atoi(argv[1]):300;
    FILE* f=fopen("bench/init_state.bin","rb"); if(!f){fprintf(stderr,"no init_state.bin\n");return 1;}
    double qp[G1_NQ],qv[G1_NV]; fread(qp,8,G1_NQ,f); fread(qv,8,G1_NV,f); fclose(f);
    float qph[G1_NQ],qvh[G1_NV];
    for(int i=0;i<G1_NQ;++i)qph[i]=(float)qp[i]; for(int i=0;i<G1_NV;++i)qvh[i]=(float)qv[i];
    float *dqp,*dqv,*dqa,*dtr;
    CK(cudaMalloc(&dqp,G1_NQ*4)); CK(cudaMalloc(&dqv,G1_NV*4));
    CK(cudaMalloc(&dqa,G1_NV*4)); CK(cudaMalloc(&dtr,(size_t)(nsteps+1)*G1_NQ*4));
    CK(cudaMemcpy(dqp,qph,G1_NQ*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqv,qvh,G1_NV*4,cudaMemcpyHostToDevice));
    aba_roll<<<1,1>>>(dqp,dqv,nsteps,dqa,dtr);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float qa[G1_NV]; float* tr=(float*)malloc((size_t)(nsteps+1)*G1_NQ*4);
    CK(cudaMemcpy(qa,dqa,G1_NV*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tr,dtr,(size_t)(nsteps+1)*G1_NQ*4,cudaMemcpyDeviceToHost));
    FILE* o=fopen("bench/aba_qacc.bin","wb"); fwrite(qa,4,G1_NV,o); fclose(o);
    FILE* t=fopen("bench/aba_traj.bin","wb"); fwrite(tr,4,(size_t)(nsteps+1)*G1_NQ,t); fclose(t);
    printf("ABA rollout done: %d steps\n",nsteps);
    return 0;
}
