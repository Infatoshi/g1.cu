// Cooperative warp-per-world ABA, state in shared memory (on-chip, no DRAM spill).
// warp = world, lane = body (lanes 0..30 = bodies, lane 31 idle). The articulated
// tree is walked in depth-waves: at each wave, all bodies at that depth run in
// parallel on their lanes; parent<->child data flows through SMEM; pass2 accumulation
// uses atomicAdd to the parent's SMEM slot (handles sibling races). One 6x6 base solve
// on lane 1. Multi-step loop keeps the world resident on-chip across timesteps.
//
// Hypothesis: removing the DRAM spill (the dense/ABA bottleneck) wins despite low lane
// utilization (G1's tree is narrow). Validated vs MuJoCo oracle. Usage:
//   ./sim_aba_coop [nsteps] [nworlds] [ksteps]   (no nworlds -> sweep)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "aba.cuh"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef WPB
#define WPB 8           // warps (worlds) per block
#endif
#define MAXDEPTH 12

// per-world SMEM layout (floats)
#define O_XPOS  0
#define O_XQUAT (O_XPOS  + G1_NBODY*3)
#define O_XIPOS (O_XQUAT + G1_NBODY*4)
#define O_S     (O_XIPOS + G1_NBODY*3)
#define O_IA    (O_S     + G1_NV*6)
#define O_VEL   (O_IA    + G1_NBODY*36)
#define O_CB    (O_VEL   + G1_NBODY*6)
#define O_PA    (O_CB    + G1_NBODY*6)
#define O_A     (O_PA    + G1_NBODY*6)
#define O_QP    (O_A     + G1_NBODY*6)   // qpos (NQ)
#define O_QV    (O_QP    + G1_NQ)        // qvel (NV)
#define PERWORLD (O_QV + G1_NV)

__global__ void coop_kernel(float* qpos, float* qvel, int nworlds, int ksteps) {
    extern __shared__ float smem[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int world = blockIdx.x*WPB + warp;
    float* W = smem + warp*PERWORLD;
    int b = lane;                       // lane = body
    bool act = (b >= 1 && b < G1_NBODY);
    int par = act ? body_parentid[b] : 0;
    // depth of body b (walk to root)
    int depth = 0; { int j=b; while (j>0){ j=body_parentid[j]; depth++; } }

    float* sxpos=W+O_XPOS; float* sxquat=W+O_XQUAT; float* sxipos=W+O_XIPOS;
    float* sS=W+O_S; float* sIA=W+O_IA; float* sVEL=W+O_VEL; float* sCB=W+O_CB;
    float* sPA=W+O_PA; float* sA=W+O_A; float* sQP=W+O_QP; float* sQV=W+O_QV;

    // load world state into SMEM (lane-strided)
    if (world < nworlds) {
        for (int i=lane;i<G1_NQ;i+=32) sQP[i]=qpos[(size_t)world*G1_NQ+i];
        for (int i=lane;i<G1_NV;i+=32) sQV[i]=qvel[(size_t)world*G1_NV+i];
    }
    if (lane==0){ sxpos[0]=0;sxpos[1]=0;sxpos[2]=0; sxquat[0]=1;sxquat[1]=0;sxquat[2]=0;sxquat[3]=0; }
    __syncwarp();

    const float dt=G1_DT;
    for (int step=0; step<ksteps; ++step) {
        // ---- FK (depth waves) ----
        for (int L=1; L<=MAXDEPTH; ++L){
            if (act && depth==L){
                Q4 qp={sxquat[4*par],sxquat[4*par+1],sxquat[4*par+2],sxquat[4*par+3]};
                V3 pp={sxpos[3*par],sxpos[3*par+1],sxpos[3*par+2]};
                V3 bpos={body_pos[3*b],body_pos[3*b+1],body_pos[3*b+2]};
                Q4 bquat={body_quat[4*b],body_quat[4*b+1],body_quat[4*b+2],body_quat[4*b+3]};
                V3 pos=vadd(pp,qrot(qp,bpos)); Q4 quat=qmul(qp,bquat);
                int ja=body_jntadr[b], jn=body_jntnum[b];
                for (int jj=0;jj<jn;++jj){ int j=ja+jj; int qa=jnt_qposadr[j]; int t=jnt_type[j];
                    if (t==0){ pos=v3(sQP[qa],sQP[qa+1],sQP[qa+2]);
                        quat=qnorm((Q4){sQP[qa+3],sQP[qa+4],sQP[qa+5],sQP[qa+6]}); }
                    else if (t==3){ V3 ax={jnt_axis[3*j],jnt_axis[3*j+1],jnt_axis[3*j+2]};
                        V3 an={jnt_pos[3*j],jnt_pos[3*j+1],jnt_pos[3*j+2]};
                        V3 aw=vadd(pos,qrot(quat,an)); quat=qmul(quat,axisangle(ax,sQP[qa]));
                        pos=vsub(aw,qrot(quat,an)); }
                }
                sxpos[3*b]=pos.x;sxpos[3*b+1]=pos.y;sxpos[3*b+2]=pos.z;
                sxquat[4*b]=quat.w;sxquat[4*b+1]=quat.x;sxquat[4*b+2]=quat.y;sxquat[4*b+3]=quat.z;
                V3 ip={body_ipos[3*b],body_ipos[3*b+1],body_ipos[3*b+2]};
                V3 cw=vadd(pos,qrot(quat,ip)); sxipos[3*b]=cw.x;sxipos[3*b+1]=cw.y;sxipos[3*b+2]=cw.z;
            }
            __syncwarp();
        }
        // pelvis ref
        V3 p={sxpos[3],sxpos[4],sxpos[5]};
        Q4 q1={sxquat[4],sxquat[5],sxquat[6],sxquat[7]}; M3 R1=quat2mat(q1);

        // ---- S axes + spatial inertia (parallel per body) ----
        if (act){
            // S for this body's dofs
            int jadr=body_jntadr[b];
            for (int jj=0;jj<body_jntnum[b];++jj){ int j=jadr+jj; int d=jnt_dofadr[j], t=jnt_type[j];
                if (t==0){ for(int k=0;k<6;++k)for(int c=0;c<6;++c) sS[(d+k)*6+c]=0;
                    sS[(d+0)*6+3]=1;sS[(d+1)*6+4]=1;sS[(d+2)*6+5]=1;
                    for(int k=0;k<3;++k){V3 axc=col(R1,k); sS[(d+3+k)*6+0]=axc.x;sS[(d+3+k)*6+1]=axc.y;sS[(d+3+k)*6+2]=axc.z;} }
                else if (t==3){ for(int c=0;c<6;++c) sS[d*6+c]=0;
                    Q4 qb={sxquat[4*b],sxquat[4*b+1],sxquat[4*b+2],sxquat[4*b+3]};
                    V3 aw=qrot(qb,(V3){jnt_axis[3*j],jnt_axis[3*j+1],jnt_axis[3*j+2]});
                    V3 anc=vadd((V3){sxpos[3*b],sxpos[3*b+1],sxpos[3*b+2]},qrot(qb,(V3){jnt_pos[3*j],jnt_pos[3*j+1],jnt_pos[3*j+2]}));
                    V3 r=vsub(anc,p), rxa=cross(r,aw);
                    sS[d*6]=aw.x;sS[d*6+1]=aw.y;sS[d*6+2]=aw.z;sS[d*6+3]=rxa.x;sS[d*6+4]=rxa.y;sS[d*6+5]=rxa.z; }
            }
            // spatial inertia 6x6 about p -> sIA[b]
            Q4 qb={sxquat[4*b],sxquat[4*b+1],sxquat[4*b+2],sxquat[4*b+3]};
            Q4 qi={body_iquat[4*b],body_iquat[4*b+1],body_iquat[4*b+2],body_iquat[4*b+3]};
            M3 Ri=quat2mat(qmul(qb,qi)); V3 di={body_inertia[3*b],body_inertia[3*b+1],body_inertia[3*b+2]};
            M3 Iw=world_inertia(Ri,di); float m=body_mass[b];
            V3 c={sxipos[3*b]-p.x,sxipos[3*b+1]-p.y,sxipos[3*b+2]-p.z}; float cc=dot3(c,c);
            float A[9]; for(int i=0;i<9;++i)A[i]=Iw.m[i];
            A[0]+=m*(cc-c.x*c.x);A[1]+=m*(-c.x*c.y);A[2]+=m*(-c.x*c.z);
            A[3]+=m*(-c.y*c.x);A[4]+=m*(cc-c.y*c.y);A[5]+=m*(-c.y*c.z);
            A[6]+=m*(-c.z*c.x);A[7]+=m*(-c.z*c.y);A[8]+=m*(cc-c.z*c.z);
            float Bx[9]={0,-c.z,c.y, c.z,0,-c.x, -c.y,c.x,0};
            float* X=&sIA[b*36];
            for(int i=0;i<3;++i)for(int j=0;j<3;++j){ X[6*i+j]=A[3*i+j]; X[6*i+(j+3)]=m*Bx[3*i+j];
                X[6*(i+3)+j]=-m*Bx[3*i+j]; X[6*(i+3)+(j+3)]=(i==j)?m:0.f; }
        }
        __syncwarp();

        // ---- pass1: velocities + bias accel (depth waves) ----
        for (int L=1; L<=MAXDEPTH; ++L){
            if (act && depth==L){
                float w6[6]; for(int k=0;k<6;++k) w6[k]=sVEL[par*6+k];
                float c6[6]={0,0,0,0,0,0};
                int da=body_dofadr[b], dn=body_dofnum[b];
                if (dn==6){
                    for(int a=da;a<da+3;++a){ V3 sw={sS[a*6],sS[a*6+1],sS[a*6+2]},sv={sS[a*6+3],sS[a*6+4],sS[a*6+5]};
                        V3 ow,ov; crossm((V3){w6[0],w6[1],w6[2]},(V3){w6[3],w6[4],w6[5]},sw,sv,ow,ov);
                        float qd=sQV[a]; c6[0]+=ow.x*qd;c6[1]+=ow.y*qd;c6[2]+=ow.z*qd;c6[3]+=ov.x*qd;c6[4]+=ov.y*qd;c6[5]+=ov.z*qd;
                        for(int k=0;k<6;++k) w6[k]+=sS[a*6+k]*qd; }
                    float wr[6]; for(int k=0;k<6;++k) wr[k]=w6[k];
                    for(int a=da+3;a<da+6;++a){ V3 sw={sS[a*6],sS[a*6+1],sS[a*6+2]},sv={sS[a*6+3],sS[a*6+4],sS[a*6+5]};
                        V3 ow,ov; crossm((V3){wr[0],wr[1],wr[2]},(V3){wr[3],wr[4],wr[5]},sw,sv,ow,ov);
                        float qd=sQV[a]; c6[0]+=ow.x*qd;c6[1]+=ow.y*qd;c6[2]+=ow.z*qd;c6[3]+=ov.x*qd;c6[4]+=ov.y*qd;c6[5]+=ov.z*qd; }
                    for(int a=da+3;a<da+6;++a){ float qd=sQV[a]; for(int k=0;k<6;++k) w6[k]+=sS[a*6+k]*qd; }
                } else {
                    for(int a=da;a<da+dn;++a){ V3 sw={sS[a*6],sS[a*6+1],sS[a*6+2]},sv={sS[a*6+3],sS[a*6+4],sS[a*6+5]};
                        V3 ow,ov; crossm((V3){w6[0],w6[1],w6[2]},(V3){w6[3],w6[4],w6[5]},sw,sv,ow,ov);
                        float qd=sQV[a]; c6[0]+=ow.x*qd;c6[1]+=ow.y*qd;c6[2]+=ow.z*qd;c6[3]+=ov.x*qd;c6[4]+=ov.y*qd;c6[5]+=ov.z*qd;
                        for(int k=0;k<6;++k) w6[k]+=sS[a*6+k]*qd; }
                }
                for(int k=0;k<6;++k){ sVEL[b*6+k]=w6[k]; sCB[b*6+k]=c6[k]; }
            }
            __syncwarp();
        }
        // pA init = v x* (I v)
        if (act){ float Iv[6]; mv6(&sIA[b*36],&sVEL[b*6],Iv); crossf6(&sVEL[b*6],Iv,&sPA[b*6]); }
        __syncwarp();

        // ---- pass2 (leaf->root): reduce, atomicAdd to parent ----
        for (int L=MAXDEPTH; L>=2; --L){
            if (act && depth==L){
                int d=body_dofadr[b];
                float U[6]; mv6(&sIA[b*36],&sS[d*6],U);
                float D=dot6(&sS[d*6],U)+dof_armature[d];
                float u=-dot6(&sS[d*6],&sPA[b*6]); float invD=1.f/D;
                float cbl[6]; for(int k=0;k<6;++k) cbl[k]=sCB[b*6+k];
                // Ia = IA - UU^T/D ; pa = pA + Ia c + U u/D
                float Ia[36]; for(int i=0;i<6;++i)for(int j=0;j<6;++j) Ia[i*6+j]=sIA[b*36+i*6+j]-U[i]*U[j]*invD;
                float Iac[6]; mv6(Ia,cbl,Iac);
                for(int i=0;i<6;++i) atomicAdd(&sPA[par*6+i], sPA[b*6+i]+Iac[i]+U[i]*(u*invD));
                for(int i=0;i<36;++i) atomicAdd(&sIA[par*36+i], Ia[i]);
            }
            __syncwarp();
        }

        // ---- pass3: base (lane 1) then depth waves ----
        if (lane==1){
            int d=body_dofadr[1];
            float ap[6]; for(int k=0;k<6;++k) ap[k]=sCB[6+k]; ap[3]-=G1_GRAVITY[0];ap[4]-=G1_GRAVITY[1];ap[5]-=G1_GRAVITY[2];
            float Umat[36], Dm[36], rhs[6];
            for(int l=0;l<6;++l){ float Ul[6]; mv6(&sIA[36],&sS[(d+l)*6],Ul); for(int i=0;i<6;++i) Umat[i*6+l]=Ul[i]; }
            for(int k=0;k<6;++k)for(int l=0;l<6;++l){ float s=0; for(int i=0;i<6;++i) s+=sS[(d+k)*6+i]*Umat[i*6+l]; Dm[k*6+l]=s+(k==l?dof_armature[d+k]:0.f); }
            for(int k=0;k<6;++k){ float uk=-dot6(&sS[(d+k)*6],&sPA[6]); float UTa=0; for(int i=0;i<6;++i) UTa+=Umat[i*6+k]*ap[i]; rhs[k]=uk-UTa; }
            float qdd[6]; chol6_solve(Dm,rhs,qdd);
            for(int k=0;k<6;++k) sQV[d+k] += dt*qdd[k];                  // integrate base vel now uses qdd
            for(int i=0;i<6;++i){ float Sq=0; for(int k=0;k<6;++k) Sq+=sS[(d+k)*6+i]*qdd[k]; sA[6+i]=ap[i]+Sq; }
            // stash base qdd in sA? we already advanced sQV; store qacc via sA not needed
        }
        __syncwarp();
        for (int L=2; L<=MAXDEPTH; ++L){
            if (act && depth==L){
                int d=body_dofadr[b];
                float ap[6]; for(int k=0;k<6;++k) ap[k]=sA[par*6+k]+sCB[b*6+k];
                float U[6]; mv6(&sIA[b*36],&sS[d*6],U);
                float D=dot6(&sS[d*6],U)+dof_armature[d];
                float u=-dot6(&sS[d*6],&sPA[b*6]);
                float qdd=(u-dot6(U,ap))/D;
                sQV[d] += dt*qdd;
                for(int i=0;i<6;++i) sA[b*6+i]=ap[i]+sS[d*6+i]*qdd;
            }
            __syncwarp();
        }

        // ---- integrate positions (lane 0 does it serially; cheap) ----
        if (lane==0){
            for (int j=0;j<G1_NJNT;++j){ int t=jnt_type[j], qa=jnt_qposadr[j], dd=jnt_dofadr[j];
                if (t==0){ sQP[qa]+=dt*sQV[dd]; sQP[qa+1]+=dt*sQV[dd+1]; sQP[qa+2]+=dt*sQV[dd+2];
                    quat_integrate(&sQP[qa+3],(V3){sQV[dd+3],sQV[dd+4],sQV[dd+5]},dt); }
                else if (t==3){ sQP[qa]+=dt*sQV[dd]; }
            }
        }
        __syncwarp();
        // zero IA/pA for next step's accumulation handled by S/inertia rewrite (IA overwritten) and pA init (overwritten);
        // but pass2 atomicAdds accumulate into IA -> must reset IA before next inertia write. Inertia write overwrites sIA fully, pA overwritten in init. OK.
    }

    // write back
    if (world < nworlds){
        for (int i=lane;i<G1_NQ;i+=32) qpos[(size_t)world*G1_NQ+i]=sQP[i];
        for (int i=lane;i<G1_NV;i+=32) qvel[(size_t)world*G1_NV+i]=sQV[i];
    }
}

static void read_init(double* qp,double* qv){ FILE* f=fopen("bench/init_state.bin","rb");
    if(!f){fprintf(stderr,"run just ref\n");exit(1);} fread(qp,8,G1_NQ,f); fread(qv,8,G1_NV,f); fclose(f); }
static void init_worlds(float* dqp,float* dqv,int N,const float* q0,const float* v0){
    std::vector<float> hp((size_t)N*G1_NQ),hv((size_t)N*G1_NV);
    for(int w=0;w<N;++w){ for(int i=0;i<G1_NQ;++i)hp[(size_t)w*G1_NQ+i]=q0[i]; for(int i=0;i<G1_NV;++i)hv[(size_t)w*G1_NV+i]=v0[i]; }
    CK(cudaMemcpy(dqp,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqv,hv.data(),hv.size()*4,cudaMemcpyHostToDevice)); }

static int smem_bytes(){ return WPB*PERWORLD*sizeof(float); }
static float run(float* dqp,float* dqv,int N,int nsteps,int ksteps){
    int blocks=(N+WPB-1)/WPB; int sm=smem_bytes();
    CK(cudaFuncSetAttribute(coop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); CK(cudaEventRecord(a));
    for(int s=0;s<nsteps;s+=ksteps){ int k=min(ksteps,nsteps-s); coop_kernel<<<blocks,WPB*32,sm>>>(dqp,dqv,N,k); }
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b)); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); CK(cudaGetLastError());
    return ms; }

int main(int argc,char**argv){
    int nsteps=(argc>1)?atoi(argv[1]):300;
    double qpd[G1_NQ],qvd[G1_NV]; read_init(qpd,qvd);
    float q0[G1_NQ],v0[G1_NV]; for(int i=0;i<G1_NQ;++i)q0[i]=(float)qpd[i]; for(int i=0;i<G1_NV;++i)v0[i]=(float)qvd[i];
    printf("coop: WPB=%d  SMEM/block=%d B (%d floats/world)\n",WPB,smem_bytes(),PERWORLD);
    if (argc>2){ int N=atoi(argv[2]); int ks=(argc>3)?atoi(argv[3]):1;
        float *dqp,*dqv; CK(cudaMalloc(&dqp,(size_t)N*G1_NQ*4)); CK(cudaMalloc(&dqv,(size_t)N*G1_NV*4));
        init_worlds(dqp,dqv,N,q0,v0); float ms=run(dqp,dqv,N,nsteps,ks);
        std::vector<float> hp((size_t)N*G1_NQ); CK(cudaMemcpy(hp.data(),dqp,hp.size()*4,cudaMemcpyDeviceToHost));
        double md=0; for(int w=1;w<N;++w)for(int i=0;i<G1_NQ;++i) md=fmax(md,fabs(hp[(size_t)w*G1_NQ+i]-hp[i]));
        FILE* o=fopen("bench/sim_coop_final.bin","wb"); fwrite(hp.data(),4,G1_NQ,o); fclose(o);
        printf("coop N=%d steps=%d ksteps=%d  %.2f ms  %.3e env-steps/s  determinism=%.1e\n",N,nsteps,ks,ms,(double)N*nsteps/(ms/1e3),md);
        return 0; }
    int Ns[]={4096,16384,65536}; int Ks[]={1,16};
    printf("cooperative ABA (warp/world, SMEM), %d steps:\n",nsteps);
    for(int N:Ns){ float *dqp,*dqv; CK(cudaMalloc(&dqp,(size_t)N*G1_NQ*4)); CK(cudaMalloc(&dqv,(size_t)N*G1_NV*4));
        for(int ks:Ks){ init_worlds(dqp,dqv,N,q0,v0); run(dqp,dqv,N,20,ks);
            init_worlds(dqp,dqv,N,q0,v0); float ms=run(dqp,dqv,N,nsteps,ks);
            printf("  N=%7d ksteps=%2d  %8.2f ms  %.3e env-steps/s\n",N,ks,ms,(double)N*nsteps/(ms/1e3)); }
        cudaFree(dqp); cudaFree(dqv); }
    return 0;
}
