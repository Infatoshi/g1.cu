// g1_env.cu -- batched G1 walking RL environment as a ctypes shared lib.
// All N worlds stay resident on the GPU; only obs/reward/done buffers are copied
// host<->device per policy step. PPO (scripts/ppo.py) owns the update math on host.
//
// Actuators: position (PD). policy outputs joint targets (29), converted to torque
//   tau = kp*(target - q) - kd*qd  (kp/kd from g1_model.h, exported additively).
// Physics: reuses the validated ABA + foot-contact path. PD torque enters qacc as the
//   extra term M^-1 tau (one more aba_solveM apply on the already-built factor) -- this
//   does NOT modify aba.cuh / aba_factor.cuh / contact.cuh; it composes them.
// Cadence: S physics substeps per policy step (control dt = S * G1_DT).
//
// extern "C" API (ctypes):
//   void* g1_env_create(int n_worlds, int substeps, int pgs_iters, unsigned long seed);
//   void  g1_env_reset (void* h, float* obs_out);                 // obs_out [N x OBS]
//   void  g1_env_step  (void* h, const float* act, float* obs,    // act [N x ACT]
//                       float* rew, float* done);                 // rew/done [N]
//   void  g1_env_destroy(void* h);
//   void  g1_env_qpos  (void* h, float* out);                     // out [NQ] (world 0)
//   int   g1_env_obs_dim(void); int g1_env_act_dim(void);
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <curand_kernel.h>
#include "dynamics.cuh"
#include "aba.cuh"
#include "aba_factor.cuh"
#include "aba_bias.cuh"
#include "contact.cuh"
#include "env_qacc.cuh"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef BLOCK
#define BLOCK 64
#endif
// min blocks/SM target for the fused rollout kernel. 1 = bound local-mem reservation (safe on
// a shared GPU); >1 trades local-mem footprint for occupancy (the kernel is latency-bound at
// 1 block/SM). Override at compile time: -DROLLOUT_MINBLK=3
#ifndef ROLLOUT_MINBLK
#define ROLLOUT_MINBLK 1
#endif

// obs layout (OBS_DIM=100):
//  [0:3]   base linear vel (world)         qvel[0:3]
//  [3:6]   base angular vel (body)         qvel[3:6]
//  [6:9]   projected gravity (body frame)  R_base^T * [0,0,-1]
//  [9:38]  joint pos (29 actuated)         qpos[g1_act_qadr]
//  [38:67] joint vel (29 actuated)         qvel[g1_act_dof]
//  [67:96] prev action (29)
//  [96:100] commands: [vx_cmd, vy_cmd, wz_cmd, 0]
#define OBS_DIM 100
#define ACT_DIM 29
#define CMD_VX 0.8f   // forward velocity command (m/s) toward the finish line (+x)
// finish-line task: robot starts at x=0 (g1_qpos_stand), must walk to x>=X_FIN upright.
// The dense forward reward (r_vel) drives it toward the line ("rewarded by how close it gets");
// crossing while upright is a WIN -> big terminal bonus + episode success.
#define X_FIN 5.0f
#define FINISH_BONUS 50.0f
// NOTE: the old KP_SCALE=0.2 explicit-Euler stability hack is REMOVED. Full kp=500 is now
// stable because the kd velocity-feedback is integrated IMPLICITLY (see env_qacc: dt*kd
// folded into the ABA articulated-inertia diagonal D = MuJoCo implicitfast).

// env_qacc moved verbatim to env_qacc.cuh (included above) so it can be unit-tested and
// optimized under a numerical-equivalence gate. No behavior change.

__device__ __forceinline__ void compute_obs(const float* qp, const float* qv,
                                            const float* prev_act, float* obs){
    for (int i=0;i<3;++i) obs[i]=qv[i];          // base lin vel (world)
    for (int i=0;i<3;++i) obs[3+i]=qv[3+i];      // base ang vel (body)
    // projected gravity: R_base^T * (0,0,-1) using base quat qpos[3:7] (wxyz)
    Q4 qb={qp[3],qp[4],qp[5],qp[6]};
    M3 Rb=quat2mat(qb);
    obs[6]=-Rb.m[6]; obs[7]=-Rb.m[7]; obs[8]=-Rb.m[8];   // -third row = R^T*(0,0,-1)
    for (int i=0;i<ACT_DIM;++i) obs[9+i]=qp[g1_act_qadr[i]];
    for (int i=0;i<ACT_DIM;++i) obs[38+i]=qv[g1_act_dof[i]];
    for (int i=0;i<ACT_DIM;++i) obs[67+i]=prev_act[i];
    obs[96]=CMD_VX; obs[97]=0.f; obs[98]=0.f; obs[99]=0.f;
}

struct EnvState {
    int N, S, pgs;
    float *qp, *qv, *prev_act;   // device persistent state [N*NQ],[N*NV],[N*ACT]
    int   *steps;                // episode step counter [N]
    curandState* rng;
    // persistent IO staging (avoid per-step cudaMalloc/Free -- it dominated wall time)
    float *dact, *dobs, *drew, *ddone;
};

__global__ void k_init_rng(curandState* st, unsigned long seed, int N){
    int w=blockIdx.x*blockDim.x+threadIdx.x; if(w>=N) return;
    curand_init(seed, w, 0, &st[w]);
}

__global__ void k_reset(float* qp, float* qv, float* prev_act, int* steps,
                        curandState* rng, int N, float* obs){
    int w=blockIdx.x*blockDim.x+threadIdx.x; if(w>=N) return;
    float* gp=qp+(size_t)w*G1_NQ; float* gv=qv+(size_t)w*G1_NV;
    float* pa=prev_act+(size_t)w*ACT_DIM;
    curandState* st=&rng[w];
    for (int i=0;i<G1_NQ;++i) gp[i]=g1_qpos_stand[i];  // stable bent-knee standing pose
    // small joint randomization on actuated dofs
    for (int i=0;i<ACT_DIM;++i) gp[g1_act_qadr[i]] += 0.02f*(curand_uniform(st)*2.f-1.f);
    for (int i=0;i<G1_NV;++i) gv[i]=0.f;
    for (int i=0;i<ACT_DIM;++i) pa[i]=0.f;
    steps[w]=0;
    float o[OBS_DIM]; compute_obs(gp, gv, pa, o);
    for (int i=0;i<OBS_DIM;++i) obs[(size_t)w*OBS_DIM+i]=o[i];
}

// Advance ONE world by one control step (S physics substeps) given an action.
// State (lp/lv/pa/step) is in/out via local arrays so the caller decides whether it lives
// in global (k_step) or stays register-resident across a whole rollout (k_rollout). This is
// the single source of truth for the physics + reward + done + auto-reset; both kernels call
// it so the host-step path and the fused on-GPU rollout share identical numerics.
__device__ void env_advance(float* lp, float* lv, float* pa, int* pstep, curandState* st,
                            const float* a, int S, int pgs, float* r_out, int* done_out){
    // policy action = residual (scaled) on the default standing pose -> joint target,
    // clamped to ctrl range. Residual convention keeps target near a stable hold so
    // an untrained (~zero) policy does not immediately destabilize the robot.
    float target[ACT_DIM];
    const float ACT_SCALE=0.5f;
    for (int i=0;i<ACT_DIM;++i){
        float t=g1_qpos_stand[g1_act_qadr[i]] + ACT_SCALE*a[i];
        if (t<g1_act_ctrl_lo[i]) t=g1_act_ctrl_lo[i];
        if (t>g1_act_ctrl_hi[i]) t=g1_act_ctrl_hi[i];
        target[i]=t;
    }

    float xpos[G1_NBODY*3], xquat[G1_NBODY*4], xipos[G1_NBODY*3];
    float Sm[G1_NV*6], qacc[G1_NV], tau[G1_NV];
    const float dt=G1_DT;
    // implicit-damping diagonal (dt*B): kd at each actuated dof (dof_damping is 0 for G1).
    // Folded into the ABA factor -> MuJoCo implicitfast-equivalent stable PD at full kp=500.
    float Bdt[G1_NV];
    for (int i=0;i<G1_NV;++i) Bdt[i]=0.f;
    for (int i=0;i<ACT_DIM;++i) Bdt[g1_act_dof[i]] = dt*g1_act_kd[i];
    float vx_sum=0.f, energy=0.f; int alive=1;
    for (int s=0;s<S;++s){
        forward_kinematics(lp, xpos, xquat, xipos);
        for (int i=0;i<G1_NV;++i) tau[i]=0.f;
        for (int i=0;i<ACT_DIM;++i){
            int d=g1_act_dof[i];
            float pdt = g1_act_kp[i]*(target[i]-lp[g1_act_qadr[i]]) - g1_act_kd[i]*lv[d];
            // MuJoCo clamps qfrc_actuator to the joint's actuatorfrcrange (kd derivative
            // stays implicit/full -> matches implicitfast). Realistic + aids stability.
            if (pdt<g1_act_frc_lo[i]) pdt=g1_act_frc_lo[i];
            else if (pdt>g1_act_frc_hi[i]) pdt=g1_act_frc_hi[i];
            tau[d]=pdt; energy+=fabsf(pdt*lv[d]);
        }
        env_qacc(lp, lv, xpos, xquat, xipos, tau, Bdt, Sm, qacc, pgs);
        for (int i=0;i<G1_NV;++i){
            float v=lv[i]+dt*qacc[i];
            // safety clamp: a diverging contact solve must not NaN-poison the batch / PPO.
            if (!isfinite(v)) v=0.f;
            if (v> 50.f) v= 50.f; else if (v<-50.f) v=-50.f;
            lv[i]=v;
        }
        integrate_pos(lp, lv, dt);
        vx_sum += lv[0];
    }
    float vx = vx_sum/S;                 // mean forward velocity over the control step
    float height = lp[2];
    // upright: base z-axis vs world up. Rb third column = base z in world.
    Q4 qb={lp[3],lp[4],lp[5],lp[6]}; M3 Rb=quat2mat(qb);
    float up = Rb.m[8];                  // world-z component of body-z axis
    if (height<0.45f || up<0.5f) alive=0; // fallen (tighter: must stay reasonably upright)

    // locomotion reward v2 (anti-"dive-and-fall"): run #1 hacked the v1 reward by lunging
    // forward and faceplanting (vx~0.45, ep_len~16, 100% fall) because forward-velocity reward
    // accrued every step while the fall cost was a single -2.0. Two structural fixes:
    //  - strong per-step ALIVE bonus so long upright episodes dominate short diving ones
    //  - forward reward GATED by uprightness (upc) so a pitched-over lunge earns ~0 velocity
    //  - falling TERMINATES with a large penalty and no positive terms (strictly bad)
    // ordering by design: walk-upright (~2.1/step) > stand (~1.3) > dive (negative return).
    float upc = up<0.f ? 0.f : (up>1.f ? 1.f : up); // uprightness gate 0..1
    float vx_c = vx>CMD_VX ? CMD_VX : vx;           // reward up to the command, no overspeed bonus
    if (vx_c<0.f) vx_c=0.f;                          // no reward for moving backward
    float r_vel  = 1.0f * vx_c * upc;               // forward progress, killed when not upright
    float vy_lat = lv[1];                            // lateral base vel (world y)
    float wz_yaw = lv[5];                            // base yaw rate (body z)
    float r_lat  = -0.2f*fabsf(vy_lat) - 0.05f*fabsf(wz_yaw);
    float r_up   = 0.3f*upc;                          // upright bonus (0..0.3)
    float r_h    = -0.5f*fabsf(height-0.78f);         // hold near standing base height (stand z=0.79)
    float act_pen=0.f; for(int i=0;i<ACT_DIM;++i){ float da=a[i]-pa[i]; act_pen+=da*da; }
    if (act_pen>10.f) act_pen=10.f;
    float r;
    if (alive) r = 1.0f + r_vel + r_lat + r_up + r_h - 0.005f*act_pen - 1e-4f*energy;
    else       r = -5.0f;                            // terminal: falling is strictly bad

    *pstep += 1;
    // WIN: crossed the finish line upright. Big terminal bonus, episode ends as success.
    int won = (alive && lp[0] >= X_FIN);
    if (won) r += FINISH_BONUS;
    int d_flag = (!alive || won || *pstep>=1000) ? 1 : 0;

    // store prev action
    for (int i=0;i<ACT_DIM;++i) pa[i]=a[i];

    if (d_flag){
        // auto-reset: respawn at the SAME stable bent-knee stand pose as k_reset (consistent
        // start distribution; the straight-leg qpos0 was less stable and mismatched).
        for (int i=0;i<G1_NQ;++i) lp[i]=g1_qpos_stand[i];
        for (int i=0;i<ACT_DIM;++i) lp[g1_act_qadr[i]] += 0.05f*(curand_uniform(st)*2.f-1.f);
        for (int i=0;i<G1_NV;++i) lv[i]=0.f;
        for (int i=0;i<ACT_DIM;++i) pa[i]=0.f;
        *pstep=0;
    }
    *r_out=r; *done_out=d_flag;
}

// Inline actor-critic forward (NO cuBLAS): one thread = one world. Matches scripts/ppo.py AC:
//   Linear(OBS->256) tanh, Linear(256->256) tanh, mu=Linear(256->ACT), v=Linear(256->1),
//   plus a per-action log_std vector. Weights are a single packed buffer in torch param order;
//   nn.Linear weight is [out,in] row-major so row o lives at W+o*in (y[o]=sum_i W[o,i]*x[i]).
//   The whole net is ~100K MACs/world/step (~2% of physics) -- reads broadcast through L2.
#define H1 256
#define H2 256
__device__ void mlp_forward(const float* W, const float* nobs,
                            float* mu, float* logstd, float* val){
    const int I=OBS_DIM, A=ACT_DIM;
    const float* W0=W;            const float* b0=W0+(size_t)H1*I;
    const float* W1=b0+H1;        const float* b1=W1+(size_t)H2*H1;
    const float* Wmu=b1+H2;       const float* bmu=Wmu+(size_t)A*H2;
    const float* Wv=bmu+A;        const float* bv=Wv+H2;
    const float* ls=bv+1;
    float z0[H1];
    for (int o=0;o<H1;++o){ float s=b0[o]; const float* row=W0+(size_t)o*I;
        for (int i=0;i<I;++i) s+=row[i]*nobs[i]; z0[o]=tanhf(s); }
    float z1[H2];
    for (int o=0;o<H2;++o){ float s=b1[o]; const float* row=W1+(size_t)o*H1;
        for (int i=0;i<H1;++i) s+=row[i]*z0[i]; z1[o]=tanhf(s); }
    for (int o=0;o<A;++o){ float s=bmu[o]; const float* row=Wmu+(size_t)o*H2;
        for (int i=0;i<H2;++i) s+=row[i]*z1[i]; mu[o]=s; }
    float sv=bv[0]; for (int i=0;i<H2;++i) sv+=Wv[i]*z1[i]; *val=sv;
    for (int o=0;o<A;++o) logstd[o]=ls[o];
}

// __launch_bounds__: this kernel has a ~32KB/thread local frame (big per-world physics scratch).
// CUDA reserves local-mem backing store for the MAX resident threads of the launch config,
// independent of N. Capping to BLOCK threads * 1 block/SM bounds that reservation (else ~4GB,
// which OOMs on a shared GPU). Occupancy is already register-limited (255 regs), so no loss.
__global__ void __launch_bounds__(BLOCK, ROLLOUT_MINBLK)
              k_step(float* qp, float* qv, float* prev_act, int* steps,
                       curandState* rng, int N, int S, int pgs,
                       const float* act, float* obs, float* rew, float* done){
    int w=blockIdx.x*blockDim.x+threadIdx.x; if(w>=N) return;
    float* gp=qp+(size_t)w*G1_NQ; float* gv=qv+(size_t)w*G1_NV;
    float* pa=prev_act+(size_t)w*ACT_DIM;
    const float* a=act+(size_t)w*ACT_DIM;

    float lp[G1_NQ], lv[G1_NV], pal[ACT_DIM];
    for (int i=0;i<G1_NQ;++i) lp[i]=gp[i];
    for (int i=0;i<G1_NV;++i) lv[i]=gv[i];
    for (int i=0;i<ACT_DIM;++i) pal[i]=pa[i];
    int step=steps[w];
    float r; int d_flag;
    env_advance(lp, lv, pal, &step, &rng[w], a, S, pgs, &r, &d_flag);
    steps[w]=step;
    for (int i=0;i<G1_NQ;++i) gp[i]=lp[i];
    for (int i=0;i<G1_NV;++i) gv[i]=lv[i];
    for (int i=0;i<ACT_DIM;++i) pa[i]=pal[i];
    float o[OBS_DIM]; compute_obs(lp, lv, pal, o);
    for (int i=0;i<OBS_DIM;++i) obs[(size_t)w*OBS_DIM+i]=o[i];
    rew[w]=r; done[w]=(float)d_flag;
}

// FUSED on-GPU rollout: one launch runs the ENTIRE H-step PPO rollout for all worlds with
// ZERO host round trips. Per-world state stays register/local-resident across all H control
// steps (only loaded/stored to global once, at the ends). Each step: build obs -> normalize
// (frozen pre-rollout mean/var) -> inline MLP -> sample action via the per-world curand ->
// physics advance. Trajectory tensors O/A/LP/V/R/D are torch-owned GPU buffers ([H,N,...]);
// the host never touches obs/act/reward. PPO update math (torch autograd) reads them in place.
// O stores RAW obs; the host renormalizes with the same frozen stats (bit-consistent) for
// training and updates the running normalizer from the raw obs afterwards.
__global__ void __launch_bounds__(BLOCK, ROLLOUT_MINBLK)
              k_rollout(float* qp, float* qv, float* prev_act, int* steps,
                        curandState* rng, int N, int S, int pgs, int H,
                        const float* W, const float* mean, const float* var,
                        float* O, float* A, float* LP, float* V, float* Rb,
                        float* Db, float* Vlast){
    int w=blockIdx.x*blockDim.x+threadIdx.x; if(w>=N) return;
    float* gp=qp+(size_t)w*G1_NQ; float* gv=qv+(size_t)w*G1_NV;
    float* gpa=prev_act+(size_t)w*ACT_DIM;
    curandState* st=&rng[w];

    float lp[G1_NQ], lv[G1_NV], pa[ACT_DIM];
    for (int i=0;i<G1_NQ;++i) lp[i]=gp[i];
    for (int i=0;i<G1_NV;++i) lv[i]=gv[i];
    for (int i=0;i<ACT_DIM;++i) pa[i]=gpa[i];
    int step=steps[w];
    const float LOG2PI=1.8378770664093453f;

    for (int t=0;t<H;++t){
        float o[OBS_DIM]; compute_obs(lp, lv, pa, o);
        float no[OBS_DIM];
        for (int i=0;i<OBS_DIM;++i){
            float z=(o[i]-mean[i])/sqrtf(var[i]+1e-8f);
            if (z<-10.f) z=-10.f; else if (z>10.f) z=10.f; no[i]=z;
        }
        size_t ob=((size_t)t*N+w)*OBS_DIM;
        for (int i=0;i<OBS_DIM;++i) O[ob+i]=o[i];   // RAW obs (host renormalizes/updates norm)

        float mu[ACT_DIM], ls[ACT_DIM], val;
        mlp_forward(W, no, mu, ls, &val);
        float a[ACT_DIM]; float lpsum=0.f;
        for (int i=0;i<ACT_DIM;++i){
            float sd=expf(ls[i]); float nz=curand_normal(st);
            float ai=mu[i]+sd*nz; a[i]=ai;
            // log N(ai; mu, sd) = -0.5*((ai-mu)/sd)^2 - log sd - 0.5 log(2pi); (ai-mu)/sd == nz
            lpsum += -0.5f*nz*nz - ls[i] - 0.5f*LOG2PI;
        }
        size_t ab=((size_t)t*N+w)*ACT_DIM;
        for (int i=0;i<ACT_DIM;++i) A[ab+i]=a[i];
        size_t sb=(size_t)t*N+w;
        LP[sb]=lpsum; V[sb]=val;

        float r; int d_flag;
        env_advance(lp, lv, pa, &step, st, a, S, pgs, &r, &d_flag);
        Rb[sb]=r; Db[sb]=(float)d_flag;
    }

    // bootstrap value for the final obs (GAE needs V(s_H))
    float o[OBS_DIM]; compute_obs(lp, lv, pa, o);
    float no[OBS_DIM];
    for (int i=0;i<OBS_DIM;++i){
        float z=(o[i]-mean[i])/sqrtf(var[i]+1e-8f);
        if (z<-10.f) z=-10.f; else if (z>10.f) z=10.f; no[i]=z;
    }
    float mu[ACT_DIM], ls[ACT_DIM], val; mlp_forward(W, no, mu, ls, &val); Vlast[w]=val;

    for (int i=0;i<G1_NQ;++i) gp[i]=lp[i];
    for (int i=0;i<G1_NV;++i) gv[i]=lv[i];
    for (int i=0;i<ACT_DIM;++i) gpa[i]=pa[i];
    steps[w]=step;
}

// Physics-only throughput bench: K control steps (each S substeps) of env_advance with ZERO
// action, fully on-device, no per-step host I/O. Isolates the env physics rate (the apples-to-
// apples unit vs MJX's mjx.step rollout) from the policy/sampling/writeback overhead.
__global__ void __launch_bounds__(BLOCK, ROLLOUT_MINBLK)
              k_phys_bench(float* qp, float* qv, float* prev_act, int* steps,
                           curandState* rng, int N, int S, int pgs, int K){
    int w=blockIdx.x*blockDim.x+threadIdx.x; if(w>=N) return;
    float lp[G1_NQ], lv[G1_NV], pa[ACT_DIM];
    for (int i=0;i<G1_NQ;++i) lp[i]=qp[(size_t)w*G1_NQ+i];
    for (int i=0;i<G1_NV;++i) lv[i]=qv[(size_t)w*G1_NV+i];
    for (int i=0;i<ACT_DIM;++i) pa[i]=prev_act[(size_t)w*ACT_DIM+i];
    int step=steps[w];
    float a[ACT_DIM]; for(int i=0;i<ACT_DIM;++i) a[i]=0.f;
    float r; int dfl;
    for (int k=0;k<K;++k) env_advance(lp, lv, pa, &step, &rng[w], a, S, pgs, &r, &dfl);
    for (int i=0;i<G1_NQ;++i) qp[(size_t)w*G1_NQ+i]=lp[i];
    for (int i=0;i<G1_NV;++i) qv[(size_t)w*G1_NV+i]=lv[i];
    for (int i=0;i<ACT_DIM;++i) prev_act[(size_t)w*ACT_DIM+i]=pa[i];
    steps[w]=step;
}

extern "C" {

// Authoritative occupancy report for the actual kernels via the CUDA Occupancy API
// (accounts for regs, local frame, launch_bounds, smem -- not hand arithmetic).
void g1_occupancy_report(){
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("SM count %d, regs/SM %d, maxThreads/SM %d (%d warps), maxBlocks/SM %d\n",
           p.multiProcessorCount, p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor,
           p.maxThreadsPerMultiProcessor/32, p.maxBlocksPerMultiProcessor);
    struct { const char* nm; const void* fn; } ks[] = {
        {"k_phys_bench",(const void*)k_phys_bench},
        {"k_rollout",   (const void*)k_rollout},
        {"k_step",      (const void*)k_step}};
    for (auto& k : ks){
        cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, k.fn);
        int mb=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k.fn, BLOCK, 0);
        int warps = mb*BLOCK/32;
        printf("  %-13s regs=%d localFrame=%zuB maxActiveBlocks/SM=%d  -> %d warps  %.1f%% occupancy\n",
               k.nm, fa.numRegs, fa.localSizeBytes, mb, warps,
               100.0*mb*BLOCK/p.maxThreadsPerMultiProcessor);
    }
}

// run K physics control steps on-device (zero action), blocking. For throughput timing.
void g1_env_bench_phys(void* h, int K){
    EnvState* E=(EnvState*)h; int gr=(E->N+BLOCK-1)/BLOCK;
    k_phys_bench<<<gr,BLOCK>>>(E->qp,E->qv,E->prev_act,E->steps,E->rng,E->N,E->S,E->pgs,K);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}

int g1_env_obs_dim(){ return OBS_DIM; }
int g1_env_act_dim(){ return ACT_DIM; }

void* g1_env_create(int N, int substeps, int pgs_iters, unsigned long seed){
    EnvState* E=new EnvState();
    E->N=N; E->S=substeps; E->pgs=pgs_iters;
    CK(cudaMalloc(&E->qp,(size_t)N*G1_NQ*4));
    CK(cudaMalloc(&E->qv,(size_t)N*G1_NV*4));
    CK(cudaMalloc(&E->prev_act,(size_t)N*ACT_DIM*4));
    CK(cudaMalloc(&E->steps,(size_t)N*4));
    CK(cudaMalloc(&E->rng,(size_t)N*sizeof(curandState)));
    CK(cudaMalloc(&E->dact,(size_t)N*ACT_DIM*4));
    CK(cudaMalloc(&E->dobs,(size_t)N*OBS_DIM*4));
    CK(cudaMalloc(&E->drew,(size_t)N*4));
    CK(cudaMalloc(&E->ddone,(size_t)N*4));
    int gr=(N+BLOCK-1)/BLOCK;
    k_init_rng<<<gr,BLOCK>>>(E->rng, seed, N);
    CK(cudaDeviceSynchronize());
    return E;
}

void g1_env_reset(void* h, float* obs_out){
    EnvState* E=(EnvState*)h; int N=E->N;
    int gr=(N+BLOCK-1)/BLOCK;
    k_reset<<<gr,BLOCK>>>(E->qp,E->qv,E->prev_act,E->steps,E->rng,N,E->dobs);
    CK(cudaMemcpy(obs_out,E->dobs,(size_t)N*OBS_DIM*4,cudaMemcpyDeviceToHost));
}

void g1_env_step(void* h, const float* act, float* obs, float* rew, float* done){
    EnvState* E=(EnvState*)h; int N=E->N;
    CK(cudaMemcpy(E->dact,act,(size_t)N*ACT_DIM*4,cudaMemcpyHostToDevice));
    int gr=(N+BLOCK-1)/BLOCK;
    k_step<<<gr,BLOCK>>>(E->qp,E->qv,E->prev_act,E->steps,E->rng,N,E->S,E->pgs,
                         E->dact,E->dobs,E->drew,E->ddone);
    CK(cudaGetLastError());   // surface launch failures (e.g. local-mem OOM) instead of zeros
    CK(cudaMemcpy(obs,E->dobs,(size_t)N*OBS_DIM*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rew,E->drew,(size_t)N*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(done,E->ddone,(size_t)N*4,cudaMemcpyDeviceToHost));
}

// FUSED rollout: run the whole H-step PPO rollout on-GPU in one launch, writing the
// trajectory directly into caller-provided DEVICE buffers (torch tensors' data_ptr). No
// host<->device copies at all. Pointers W/mean/var/O/A/LP/V/R/D/Vlast are device addresses.
void g1_rollout(void* h, int H, const float* W, const float* mean, const float* var,
                float* O, float* A, float* LP, float* V, float* R, float* D, float* Vlast){
    EnvState* E=(EnvState*)h; int N=E->N;
    int gr=(N+BLOCK-1)/BLOCK;
    k_rollout<<<gr,BLOCK>>>(E->qp,E->qv,E->prev_act,E->steps,E->rng,N,E->S,E->pgs,H,
                            W,mean,var,O,A,LP,V,R,D,Vlast);
    CK(cudaGetLastError());   // surface launch failures (local-mem OOM) instead of silent zeros
}

// Copy world-0's qpos (G1_NQ floats) to host. Additive accessor for faithful playback
// (scripts/play.py records world-0's trajectory, replays kinematically in mujoco.viewer).
void g1_env_qpos(void* h, float* out){
    EnvState* E=(EnvState*)h;
    CK(cudaMemcpy(out, E->qp, (size_t)G1_NQ*4, cudaMemcpyDeviceToHost));
}

// Copy ALL N worlds' qpos ([N x G1_NQ]) to host -- lets playback pick a clean winning world
// (one that walks to the finish line without falling) for the demo render.
void g1_env_qpos_all(void* h, float* out){
    EnvState* E=(EnvState*)h;
    CK(cudaMemcpy(out, E->qp, (size_t)E->N*G1_NQ*4, cudaMemcpyDeviceToHost));
}

void g1_env_destroy(void* h){
    EnvState* E=(EnvState*)h;
    cudaFree(E->qp); cudaFree(E->qv); cudaFree(E->prev_act); cudaFree(E->steps); cudaFree(E->rng);
    cudaFree(E->dact); cudaFree(E->dobs); cudaFree(E->drew); cudaFree(E->ddone);
    delete E;
}

} // extern "C"
