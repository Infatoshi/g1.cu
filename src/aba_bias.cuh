// Direct bias force (qfrc_bias = Coriolis + gravity) via RNE with qacc=0, world axes about
// the pelvis origin -- the SAME validated recursion as compute_M_bias() in dynamics.cuh, but
// WITHOUT forming the dense mass matrix (no CRBA). MuJoCo computes qfrc_bias exactly this way.
//
// Why this exists: env_qacc previously recovered -bias the roundabout way -- run ABA forward
// dynamics (a factorize + solve, qacc_plain = M^-1(-bias)) then multiply back (M * qacc_plain).
// That is two O(N) passes (one of them division-heavy) plus their roundoff to recover a
// quantity RNE gives directly in one pass. This removes a full factorization + the mulM and
// shrinks the kernel's local working set (the actual bandwidth bottleneck).
//
// S (motion subspace, [NV*6]) is filled here identically to aba_qacc/compute_M_bias so the
// downstream aba_factorize_damped consumes the same basis. The factorization is velocity-
// independent, so S is the only shared dependency.
#pragma once
#include "dynamics.cuh"

__device__ void compute_bias(const float* qpos, const float* qvel,
                             const float* xpos, const float* xquat, const float* xipos,
                             float* S, float* bias) {
    const int NV = G1_NV, NB = G1_NBODY;
    V3 p = {xpos[3], xpos[4], xpos[5]};                 // pelvis origin (ref point), body 1
    Q4 q1 = {xquat[4], xquat[5], xquat[6], xquat[7]};
    M3 R1 = quat2mat(q1);

    // ---- motion axes S per dof, about p, world axes (identical to aba_qacc) ----
    for (int i = 0; i < NV*6; ++i) S[i] = 0.f;
    for (int j = 0; j < G1_NJNT; ++j) {
        int t = jnt_type[j], b = jnt_bodyid[j], d = jnt_dofadr[j];
        if (t == 0) {                       // free joint, dofs d..d+5
            S[(d+0)*6+3]=1; S[(d+1)*6+4]=1; S[(d+2)*6+5]=1;
            for (int k=0;k<3;++k){ V3 ax=col(R1,k); S[(d+3+k)*6+0]=ax.x; S[(d+3+k)*6+1]=ax.y; S[(d+3+k)*6+2]=ax.z; }
        } else if (t == 3) {                // hinge
            Q4 qb={xquat[4*b],xquat[4*b+1],xquat[4*b+2],xquat[4*b+3]};
            V3 axl={jnt_axis[3*j],jnt_axis[3*j+1],jnt_axis[3*j+2]};
            V3 aw = qrot(qb, axl);          // world axis
            V3 anchl={jnt_pos[3*j],jnt_pos[3*j+1],jnt_pos[3*j+2]};
            V3 anchor = vadd((V3){xpos[3*b],xpos[3*b+1],xpos[3*b+2]}, qrot(qb, anchl));
            V3 r = vsub(anchor, p);
            V3 rxa = cross(r, aw);
            S[d*6+0]=aw.x; S[d*6+1]=aw.y; S[d*6+2]=aw.z;
            S[d*6+3]=rxa.x; S[d*6+4]=rxa.y; S[d*6+5]=rxa.z;
        }
    }

    // ---- per-body spatial inertia about p (about-com form, for I a and v x* I v) ----
    SpatInert I[G1_NBODY];
    for (int b = 1; b < NB; ++b) {
        Q4 qb={xquat[4*b],xquat[4*b+1],xquat[4*b+2],xquat[4*b+3]};
        Q4 qi={body_iquat[4*b],body_iquat[4*b+1],body_iquat[4*b+2],body_iquat[4*b+3]};
        M3 Ri = quat2mat(qmul(qb, qi));
        V3 diag={body_inertia[3*b],body_inertia[3*b+1],body_inertia[3*b+2]};
        I[b].Iw = world_inertia(Ri, diag);
        I[b].c  = {xipos[3*b]-p.x, xipos[3*b+1]-p.y, xipos[3*b+2]-p.z};
        I[b].m  = body_mass[b];
    }

    // ---- RNE for bias (qacc=0, gravity via base accel = -g), mirroring MuJoCo ----
    // Incremental velocity + cdof_dot: Sdot[a] = (partial cvel) xm S[a], where the partial
    // velocity is parent + earlier dofs of this body (NOT the full body vel).
    V3 vw[G1_NBODY], vv[G1_NBODY], aw[G1_NBODY], av[G1_NBODY];
    V3 Sdw[G1_NV], Sdv[G1_NV];
    vw[0]={0,0,0}; vv[0]={0,0,0};
    aw[0]={0,0,0}; av[0]={-G1_GRAVITY[0],-G1_GRAVITY[1],-G1_GRAVITY[2]}; // -g
    for (int b=1;b<NB;++b){
        int par=body_parentid[b];
        V3 w=vw[par], v=vv[par];   // partial velocity, starts at parent
        int da=body_dofadr[b], dn=body_dofnum[b];
        if (dn == 6) {
            // free joint: 3 translation dofs accumulate normally; the 3 rotational dofs'
            // cdof_dot all use the post-translation partial velocity -- matches MuJoCo.
            for(int a=da;a<da+3;++a){ float qd=qvel[a];
                V3 sw={S[a*6+0],S[a*6+1],S[a*6+2]}, sv={S[a*6+3],S[a*6+4],S[a*6+5]};
                crossm(w,v, sw,sv, Sdw[a],Sdv[a]);
                w={w.x+sw.x*qd, w.y+sw.y*qd, w.z+sw.z*qd};
                v={v.x+sv.x*qd, v.y+sv.y*qd, v.z+sv.z*qd};
            }
            V3 wr=w, vr=v;  // post-translation partial velocity
            for(int a=da+3;a<da+6;++a){
                V3 sw={S[a*6+0],S[a*6+1],S[a*6+2]}, sv={S[a*6+3],S[a*6+4],S[a*6+5]};
                crossm(wr,vr, sw,sv, Sdw[a],Sdv[a]);
            }
            for(int a=da+3;a<da+6;++a){ float qd=qvel[a];
                w={w.x+S[a*6+0]*qd, w.y+S[a*6+1]*qd, w.z+S[a*6+2]*qd};
                v={v.x+S[a*6+3]*qd, v.y+S[a*6+4]*qd, v.z+S[a*6+5]*qd};
            }
        } else {
            for(int a=da;a<da+dn;++a){ float qd=qvel[a];
                V3 sw={S[a*6+0],S[a*6+1],S[a*6+2]}, sv={S[a*6+3],S[a*6+4],S[a*6+5]};
                crossm(w,v, sw,sv, Sdw[a],Sdv[a]);     // cdof_dot uses partial velocity
                w={w.x+sw.x*qd, w.y+sw.y*qd, w.z+sw.z*qd};
                v={v.x+sv.x*qd, v.y+sv.y*qd, v.z+sv.z*qd};
            }
        }
        vw[b]=w; vv[b]=v;
        // a[b] = a[par] + sum_a (S_a*qddot(=0) + Sdot_a*qdot_a)
        V3 aaw=aw[par], aav=av[par];
        for(int a=da;a<da+dn;++a){ float qd=qvel[a];
            aaw={aaw.x+Sdw[a].x*qd, aaw.y+Sdw[a].y*qd, aaw.z+Sdw[a].z*qd};
            aav={aav.x+Sdv[a].x*qd, aav.y+Sdv[a].y*qd, aav.z+Sdv[a].z*qd};
        }
        aw[b]=aaw; av[b]=aav;
    }
    // forces f[b] = I a + v x* (I v); backward accumulate, project tau=S^T f
    V3 fw[G1_NBODY], fv[G1_NBODY];
    for (int b=1;b<NB;++b){
        V3 Iaw,Iav; apply_inertia(I[b], aw[b], av[b], Iaw, Iav);
        V3 Ivw,Ivv; apply_inertia(I[b], vw[b], vv[b], Ivw, Ivv);
        V3 cw,cv; crossf(vw[b],vv[b], Ivw,Ivv, cw,cv);
        fw[b]={Iaw.x+cw.x, Iaw.y+cw.y, Iaw.z+cw.z};
        fv[b]={Iav.x+cv.x, Iav.y+cv.y, Iav.z+cv.z};
    }
    for (int a=0;a<NV;++a) bias[a]=0.f;
    for (int b=NB-1;b>=1;--b){
        int da=body_dofadr[b], dn=body_dofnum[b];
        for(int a=da;a<da+dn;++a)
            bias[a]=S[a*6+0]*fw[b].x+S[a*6+1]*fw[b].y+S[a*6+2]*fw[b].z
                   +S[a*6+3]*fv[b].x+S[a*6+4]*fv[b].y+S[a*6+5]*fv[b].z;
        int par=body_parentid[b];
        if(par>=1){ fw[par]=vadd(fw[par],fw[b]); fv[par]=vadd(fv[par],fv[b]); }
    }
}
