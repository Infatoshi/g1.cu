// Batched full-physics sim: smooth ABA dynamics + foot-ground contact solve per world.
// One thread = one world (matches the winning smooth-kernel mapping). Per step:
//   FK -> compute_M_bias (dense M, S, bias) -> qacc_smooth = M^-1(-bias)
//   -> detect 8 foot-sphere contacts -> build pyramidal J -> reconstruct R/aref
//   -> A = J M^-1 J^T + diag(R), b = J qacc_smooth - aref -> PGS for f
//   -> qacc = qacc_smooth + M^-1 (J^T f) -> integrate.
// All from scratch (no oracle). Validated vs bench/contact_solve_ref.npz first, then benched.
//
// Modes:
//   ./sim_contact validate            -> per-state qacc/qfrc error vs oracle (states from .bin)
//   ./sim_contact [nsteps] [nworlds]   -> single batched run, env-steps/s + determinism
//   ./sim_contact                      -> sweep N x nsteps benchmark vs MJX/Warp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "dynamics.cuh"
#include "aba.cuh"
#include "aba_factor.cuh"
#include "contact.cuh"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef BLOCK
#define BLOCK 64
#endif

// ---- the full smooth+contact qacc for one world. Writes qacc[NV]. ----
// M^-1 via ABA articulated-inertia factor (aba_factor.cuh) -- NO dense CRBA/Cholesky.
// `S` scratch (NV*6) is caller-provided; PGS iters fixed.
__device__ void full_qacc(const float* qpos, const float* qvel,
                          const float* xpos, const float* xquat, const float* xipos,
                          float* S, float* /*unused*/, float* qacc, int pgs_iters) {
    const int NV = G1_NV;

    // qacc_smooth = M^-1(-bias) via the validated ABA smooth pass; it also fills S.
    float abascr[ABA_SCR], qacc_smooth[G1_NV];
    aba_qacc(qpos, qvel, xpos, xquat, xipos, S, abascr, qacc_smooth);

    // --- contacts ---
    Contact cons[G1_MAX_CONTACT];
    int ncon = detect_foot_contacts(xpos, xquat, cons);
    if (ncon == 0) {
        for (int i=0;i<NV;++i) qacc[i]=qacc_smooth[i];
        return;
    }

    // Factorize M ONCE (ABA articulated inertias). Reuse for every M^-1 J^T apply + the
    // final M^-1 J^T f. aba_qacc already left S in the correct (world-about-pelvis) basis.
    AbaFactor fac; float facscr[ABA_FAC_SCR];
    aba_factorize(xquat, xipos, xpos, S, facscr, &fac);
    float solvescr[ABA_SOLVEM_SCR];

    V3 p = {xpos[3],xpos[4],xpos[5]};   // pelvis origin (body 1) = S reference point
    float J[G1_MAX_EFC*G1_NV];
    int nefc = build_contact_jac(cons, ncon, S, p, J);

    // per-contact invweight (foot body), floor contributes 0. detect_foot_contacts appends
    // contacts in sphere order, so re-walk the spheres and pick the penetrating ones to map
    // each contact -> its sphere's body_invweight0.
    float invw0[G1_MAX_CONTACT];
    {
        int idx=0;
        for (int s=0; s<G1_NFOOT_SPHERE; ++s) {
            int b = g1_foot_body[s];
            Q4 qb = {xquat[4*b],xquat[4*b+1],xquat[4*b+2],xquat[4*b+3]};
            V3 lp = {g1_foot_lpos[3*s],g1_foot_lpos[3*s+1],g1_foot_lpos[3*s+2]};
            V3 c  = vadd((V3){xpos[3*b],xpos[3*b+1],xpos[3*b+2]}, qrot(qb,lp));
            float pen = c.z - G1_FOOT_RADIUS;
            if (pen < 0.f) { invw0[idx++] = g1_foot_body_invweight0[s]; }
        }
    }

    float R[G1_MAX_EFC], aref[G1_MAX_EFC];
    build_R_aref(cons, ncon, invw0, J, qvel, R, aref);

    // A = J M^-1 J^T + diag(R).  Columns of MinvJT[r] = M^-1 (J row r)^T via ABA solveM.
    float MinvJT[G1_MAX_EFC*G1_NV];
    for (int r=0;r<nefc;++r) aba_solveM(&fac, S, &J[r*NV], solvescr, &MinvJT[r*NV]);
    float A[G1_MAX_EFC*G1_MAX_EFC], bvec[G1_MAX_EFC];
    for (int i=0;i<nefc;++i){
        // b_i = J_i . qacc_smooth - aref_i
        float bi=0.f; for(int a=0;a<NV;++a) bi += J[i*NV+a]*qacc_smooth[a];
        bvec[i] = bi - aref[i];
        for (int j=0;j<nefc;++j){
            float s=0.f; const float* Ji=&J[i*NV]; const float* Mj=&MinvJT[j*NV];
            for (int a=0;a<NV;++a) s += Ji[a]*Mj[a];
            A[i*nefc+j] = s + (i==j ? R[i] : 0.f);
        }
    }

    float f[G1_MAX_EFC];
    solve_pgs(A, bvec, f, nefc, pgs_iters);

    // qacc = qacc_smooth + M^-1 (J^T f)  (one more ABA solveM apply)
    float Jtf[G1_NV], dq[G1_NV];
    contact_qfrc(J, f, nefc, Jtf);
    aba_solveM(&fac, S, Jtf, solvescr, dq);
    for (int i=0;i<NV;++i) qacc[i] = qacc_smooth[i] + dq[i];
}

// ---- batched step kernel: ksteps in-kernel, one thread/world ----
__global__ void step_kernel_contact(float* qpos, float* qvel, int nworlds, int ksteps, int pgs_iters) {
    int w = blockIdx.x*blockDim.x + threadIdx.x;
    if (w >= nworlds) return;
    float* gqp = qpos + (size_t)w*G1_NQ;
    float* gqv = qvel + (size_t)w*G1_NV;
    float qp[G1_NQ], qv[G1_NV];
    for (int i=0;i<G1_NQ;++i) qp[i]=gqp[i];
    for (int i=0;i<G1_NV;++i) qv[i]=gqv[i];

    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    float S[G1_NV*6], qacc[G1_NV];
    const float dt = G1_DT;
    for (int s=0;s<ksteps;++s){
        forward_kinematics(qp, xpos, xquat, xipos);
        full_qacc(qp, qv, xpos, xquat, xipos, S, nullptr, qacc, pgs_iters);
        for (int i=0;i<G1_NV;++i) qv[i] += dt*qacc[i];
        integrate_pos(qp, qv, dt);
    }
    for (int i=0;i<G1_NQ;++i) gqp[i]=qp[i];
    for (int i=0;i<G1_NV;++i) gqv[i]=qv[i];
}

// ---- validation kernel: one thread per state, compute qacc once, write out ----
__global__ void validate_kernel(const float* qpos, const float* qvel, float* qacc_out,
                                 int nstates, int pgs_iters) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nstates) return;
    const float* qp = qpos + (size_t)k*G1_NQ;
    const float* qv = qvel + (size_t)k*G1_NV;
    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    float S[G1_NV*6], qacc[G1_NV];
    float qpl[G1_NQ], qvl[G1_NV];
    for(int i=0;i<G1_NQ;++i)qpl[i]=qp[i];
    for(int i=0;i<G1_NV;++i)qvl[i]=qv[i];
    forward_kinematics(qpl, xpos, xquat, xipos);
    full_qacc(qpl, qvl, xpos, xquat, xipos, S, nullptr, qacc, pgs_iters);
    for(int i=0;i<G1_NV;++i) qacc_out[(size_t)k*G1_NV+i]=qacc[i];
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
static float run(float* dqp,float* dqv,int N,int nsteps,int ksteps,int pgs_iters){
    int blocks=(N+BLOCK-1)/BLOCK;
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for(int s=0;s<nsteps;s+=ksteps){ int k=min(ksteps,nsteps-s); step_kernel_contact<<<blocks,BLOCK>>>(dqp,dqv,N,k,pgs_iters); }
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); CK(cudaGetLastError());
    return ms;
}

// ---- validation: read bench/contact_validate.bin (written by gen_contact_validate.py) ----
static int do_validate(int pgs_iters){
    FILE* f=fopen("bench/contact_validate.bin","rb");
    if(!f){fprintf(stderr,"missing bench/contact_validate.bin (run scripts/gen_contact_validate.py)\n");return 1;}
    int nstates; fread(&nstates,4,1,f);
    std::vector<float> qpos((size_t)nstates*G1_NQ), qvel((size_t)nstates*G1_NV), qacc_ref((size_t)nstates*G1_NV);
    std::vector<int> labels(nstates);
    fread(qpos.data(),4,qpos.size(),f);
    fread(qvel.data(),4,qvel.size(),f);
    fread(qacc_ref.data(),4,qacc_ref.size(),f);
    fread(labels.data(),4,nstates,f);
    fclose(f);

    float *dqp,*dqv,*dqa;
    CK(cudaMalloc(&dqp,qpos.size()*4)); CK(cudaMalloc(&dqv,qvel.size()*4)); CK(cudaMalloc(&dqa,qacc_ref.size()*4));
    CK(cudaMemcpy(dqp,qpos.data(),qpos.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqv,qvel.data(),qvel.size()*4,cudaMemcpyHostToDevice));
    int blocks=(nstates+BLOCK-1)/BLOCK;
    validate_kernel<<<blocks,BLOCK>>>(dqp,dqv,dqa,nstates,pgs_iters);
    CK(cudaDeviceSynchronize());
    std::vector<float> qacc((size_t)nstates*G1_NV);
    CK(cudaMemcpy(qacc.data(),dqa,qacc.size()*4,cudaMemcpyDeviceToHost));

    printf("CUDA full-physics validation vs oracle (fp32, %d PGS iters):\n", pgs_iters);
    printf("  %-5s %12s %12s\n","state","qacc_relerr","qacc_maxerr");
    double worst=0;
    for(int k=0;k<nstates;++k){
        double num=0,den=0,mx=0;
        for(int i=0;i<G1_NV;++i){
            double e=qacc[(size_t)k*G1_NV+i]-qacc_ref[(size_t)k*G1_NV+i];
            num+=e*e; den+=qacc_ref[(size_t)k*G1_NV+i]*qacc_ref[(size_t)k*G1_NV+i];
            mx=fmax(mx,fabs(e));
        }
        double rel=sqrt(num)/fmax(sqrt(den),1e-9);
        worst=fmax(worst,rel);
        printf("  %-5d %12.3e %12.3e\n",k,rel,mx);
    }
    printf("worst qacc relerr: %.3e\n", worst);
    return 0;
}

int main(int argc,char**argv){
    int pgs_iters = 2000;
    if (argc>1 && strcmp(argv[1],"validate")==0) return do_validate(pgs_iters);

    int nsteps=(argc>1)?atoi(argv[1]):300;
    double qpd[G1_NQ],qvd[G1_NV]; read_init(qpd,qvd);
    float q0[G1_NQ],v0[G1_NV];
    for(int i=0;i<G1_NQ;++i)q0[i]=(float)qpd[i]; for(int i=0;i<G1_NV;++i)v0[i]=(float)qvd[i];
    // drop start so contacts engage during the run
    q0[2]+=0.05f;

    if (argc>2){
        int N=atoi(argv[2]); int ks=(argc>3)?atoi(argv[3]):1;
        float *dqp,*dqv; CK(cudaMalloc(&dqp,(size_t)N*G1_NQ*4)); CK(cudaMalloc(&dqv,(size_t)N*G1_NV*4));
        init_worlds(dqp,dqv,N,q0,v0);
        float ms=run(dqp,dqv,N,nsteps,ks,pgs_iters);
        std::vector<float> hp((size_t)N*G1_NQ); CK(cudaMemcpy(hp.data(),dqp,hp.size()*4,cudaMemcpyDeviceToHost));
        double md=0; for(int w=1;w<N;++w) for(int i=0;i<G1_NQ;++i) md=fmax(md,fabs(hp[(size_t)w*G1_NQ+i]-hp[i]));
        printf("CONTACT N=%d steps=%d ksteps=%d  %.2f ms  %.3e env-steps/s  determinism=%.1e\n",
               N,nsteps,ks,ms,(double)N*nsteps/(ms/1e3),md);
        return 0;
    }
    int Ns[]={4096,16384,65536};
    printf("Full-physics (ABA + foot-ground contacts) batched sim, %d steps, PGS=%d:\n",nsteps,pgs_iters);
    for(int N:Ns){
        float *dqp,*dqv; CK(cudaMalloc(&dqp,(size_t)N*G1_NQ*4)); CK(cudaMalloc(&dqv,(size_t)N*G1_NV*4));
        init_worlds(dqp,dqv,N,q0,v0); run(dqp,dqv,N,20,1,pgs_iters);          // warmup
        init_worlds(dqp,dqv,N,q0,v0); float ms=run(dqp,dqv,N,nsteps,1,pgs_iters);
        double eps=(double)N*nsteps/(ms/1e3);
        printf("  N=%7d  %8.2f ms  %.3e env-steps/s  (%.1fx MJX 3.1e5, %.1fx Warp 2.2e5)\n",
               N,ms,eps,eps/3.1e5,eps/2.2e5);
        cudaFree(dqp); cudaFree(dqv);
    }
    return 0;
}
