// g1 device dynamics (fp32) -- validated vs MuJoCo at M0. Shared by dynamics.cu (M0
// single-world harness) and sim.cu (M1 batched baseline). See DEVLOG / SPEC.
//   FK -> CRBA(+armature) -> RNE bias -> Cholesky solve -> semi-implicit Euler.
//   quat (w,x,y,z) Hamilton; free-joint qvel = WORLD linear + BODY-LOCAL angular.
#pragma once
#include <cmath>
#include "g1_model.h"

// ---------------- quaternion / vector helpers (device) ----------------
struct V3 { float x, y, z; };
struct Q4 { float w, x, y, z; };

__device__ __forceinline__ V3 v3(float x, float y, float z) { return {x, y, z}; }
__device__ __forceinline__ V3 vadd(V3 a, V3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ __forceinline__ V3 vsub(V3 a, V3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }

__device__ __forceinline__ Q4 qmul(Q4 a, Q4 b) {
    return {
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}
// rotate vector v by quaternion q:  q * (0,v) * conj(q)
__device__ __forceinline__ V3 qrot(Q4 q, V3 v) {
    // t = 2 * cross(q.xyz, v); v' = v + q.w*t + cross(q.xyz, t)
    V3 u = {q.x, q.y, q.z};
    V3 t = {2.f*(u.y*v.z - u.z*v.y), 2.f*(u.z*v.x - u.x*v.z), 2.f*(u.x*v.y - u.y*v.x)};
    return {
        v.x + q.w*t.x + (u.y*t.z - u.z*t.y),
        v.y + q.w*t.y + (u.z*t.x - u.x*t.z),
        v.z + q.w*t.z + (u.x*t.y - u.y*t.x)
    };
}
__device__ __forceinline__ Q4 qnorm(Q4 q) {
    float n = rsqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    return {q.w*n, q.x*n, q.y*n, q.z*n};
}
__device__ __forceinline__ Q4 axisangle(V3 a, float ang) {
    float h = 0.5f*ang, s = sinf(h);
    return {cosf(h), a.x*s, a.y*s, a.z*s};
}

// ---------------- forward kinematics ----------------
// Fills world body frames xpos[NBODY*3], xquat[NBODY*4], and body-com xipos.
__device__ void forward_kinematics(const float* qpos,
                                   float* xpos, float* xquat, float* xipos) {
    // world body
    xpos[0]=0; xpos[1]=0; xpos[2]=0;
    xquat[0]=1; xquat[1]=0; xquat[2]=0; xquat[3]=0;

    for (int b = 1; b < G1_NBODY; ++b) {
        int p = body_parentid[b];
        Q4 qp = {xquat[4*p], xquat[4*p+1], xquat[4*p+2], xquat[4*p+3]};
        V3 pp = {xpos[3*p], xpos[3*p+1], xpos[3*p+2]};
        // fixed offset from parent
        V3 bpos = {body_pos[3*b], body_pos[3*b+1], body_pos[3*b+2]};
        Q4 bquat = {body_quat[4*b], body_quat[4*b+1], body_quat[4*b+2], body_quat[4*b+3]};
        V3 pos = vadd(pp, qrot(qp, bpos));
        Q4 quat = qmul(qp, bquat);

        // apply this body's joints (G1: exactly one per body)
        int ja = body_jntadr[b], jn = body_jntnum[b];
        for (int jj = 0; jj < jn; ++jj) {
            int j = ja + jj;
            int qadr = jnt_qposadr[j];
            int t = jnt_type[j];
            if (t == 0) {            // free
                pos = v3(qpos[qadr], qpos[qadr+1], qpos[qadr+2]);
                quat = qnorm((Q4){qpos[qadr+3], qpos[qadr+4], qpos[qadr+5], qpos[qadr+6]});
            } else if (t == 3) {     // hinge (qpos0 == 0)
                V3 axis = {jnt_axis[3*j], jnt_axis[3*j+1], jnt_axis[3*j+2]};
                V3 anch = {jnt_pos[3*j], jnt_pos[3*j+1], jnt_pos[3*j+2]};
                V3 anchor_w = vadd(pos, qrot(quat, anch));
                quat = qmul(quat, axisangle(axis, qpos[qadr]));
                pos = vsub(anchor_w, qrot(quat, anch));
            }
        }
        xpos[3*b]=pos.x; xpos[3*b+1]=pos.y; xpos[3*b+2]=pos.z;
        xquat[4*b]=quat.w; xquat[4*b+1]=quat.x; xquat[4*b+2]=quat.y; xquat[4*b+3]=quat.z;
        // body com in world
        V3 ip = {body_ipos[3*b], body_ipos[3*b+1], body_ipos[3*b+2]};
        V3 cw = vadd(pos, qrot(quat, ip));
        xipos[3*b]=cw.x; xipos[3*b+1]=cw.y; xipos[3*b+2]=cw.z;
    }
    xipos[0]=0; xipos[1]=0; xipos[2]=0;
}

// ---------------- spatial algebra helpers ([angular(3); linear(3)] order) ----------------
// 3-vector cross product
__device__ __forceinline__ V3 cross(V3 a, V3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
__device__ __forceinline__ float dot3(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

// 3x3 matrix (row-major) from quaternion
struct M3 { float m[9]; };
__device__ M3 quat2mat(Q4 q) {
    float w=q.w,x=q.x,y=q.y,z=q.z;
    M3 r;
    r.m[0]=1-2*(y*y+z*z); r.m[1]=2*(x*y-w*z);   r.m[2]=2*(x*z+w*y);
    r.m[3]=2*(x*y+w*z);   r.m[4]=1-2*(x*x+z*z); r.m[5]=2*(y*z-w*x);
    r.m[6]=2*(x*z-w*y);   r.m[7]=2*(y*z+w*x);   r.m[8]=1-2*(x*x+y*y);
    return r;
}
__device__ __forceinline__ V3 col(const M3& R, int c) { return {R.m[c], R.m[3+c], R.m[6+c]}; }
// world rotational inertia: R * diag(d) * R^T  (returns 3x3 sym, row-major)
__device__ M3 world_inertia(const M3& R, V3 d) {
    M3 out;
    for (int i=0;i<3;++i)
        for (int j=0;j<3;++j)
            out.m[3*i+j] = R.m[3*i+0]*d.x*R.m[3*j+0] + R.m[3*i+1]*d.y*R.m[3*j+1] + R.m[3*i+2]*d.z*R.m[3*j+2];
    return out;
}
__device__ __forceinline__ V3 mat3vec(const M3& A, V3 v) {
    return {A.m[0]*v.x+A.m[1]*v.y+A.m[2]*v.z,
            A.m[3]*v.x+A.m[4]*v.y+A.m[5]*v.z,
            A.m[6]*v.x+A.m[7]*v.y+A.m[8]*v.z};
}

// A rigid-body spatial inertia about the common ref point, stored as (Iw 3x3, com offset c, mass m).
struct SpatInert { M3 Iw; V3 c; float m; };

// I * [sw; sv]  ->  [A sw + m c×sv ; -m c×sw + m sv]   (A = Iw + m(|c|^2 I - c c^T))
__device__ void apply_inertia(const SpatInert& I, V3 sw, V3 sv, V3& fw, V3& fv) {
    float m = I.m; V3 c = I.c;
    V3 Asw = mat3vec(I.Iw, sw);
    float cc = dot3(c,c);
    // A sw = Iw sw + m(|c|^2 sw - (c.sw) c)
    float csw = dot3(c, sw);
    Asw = {Asw.x + m*(cc*sw.x - csw*c.x),
           Asw.y + m*(cc*sw.y - csw*c.y),
           Asw.z + m*(cc*sw.z - csw*c.z)};
    V3 mc_sv = cross(c, sv);   // c × sv
    fw = {Asw.x + m*mc_sv.x, Asw.y + m*mc_sv.y, Asw.z + m*mc_sv.z};
    V3 c_sw = cross(c, sw);    // c × sw
    fv = {-m*c_sw.x + m*sv.x, -m*c_sw.y + m*sv.y, -m*c_sw.z + m*sv.z};
}

// motion x motion:  [w1;v1] xm [w2;v2] = [w1×w2 ; w1×v2 + v1×w2]
__device__ void crossm(V3 w1,V3 v1, V3 w2,V3 v2, V3& ow,V3& ov){
    ow = cross(w1,w2);
    V3 a=cross(w1,v2), b=cross(v1,w2);
    ov = {a.x+b.x, a.y+b.y, a.z+b.z};
}
// motion x force:  [w1;v1] x* [n;f0] = [w1×n + v1×f0 ; w1×f0]
__device__ void crossf(V3 w1,V3 v1, V3 n,V3 f0, V3& on,V3& of){
    V3 a=cross(w1,n), b=cross(v1,f0);
    on = {a.x+b.x, a.y+b.y, a.z+b.z};
    of = cross(w1,f0);
}

// ---------------- CRBA + RNE: dynamics about pelvis origin, world axes ----------------
// Computes dense mass matrix M[NV*NV] and bias force qfrc_bias[NV] (Coriolis+gravity).
// S[NV*6] are the motion-subspace axes (filled here too). Single thread.
__device__ void compute_M_bias(const float* qpos, const float* qvel,
                               const float* xpos, const float* xquat, const float* xipos,
                               float* S, float* Mout, float* bias) {
    const int NV = G1_NV, NB = G1_NBODY;
    V3 p = {xpos[3], xpos[4], xpos[5]};                 // pelvis origin (ref point), body 1
    Q4 q1 = {xquat[4], xquat[5], xquat[6], xquat[7]};
    M3 R1 = quat2mat(q1);

    // ---- motion axes S per dof, about p, world axes ----
    for (int i = 0; i < NV*6; ++i) S[i] = 0.f;
    for (int j = 0; j < G1_NJNT; ++j) {
        int t = jnt_type[j], b = jnt_bodyid[j], d = jnt_dofadr[j];
        if (t == 0) {                       // free joint, dofs d..d+5
            // linear (world): [0,0,0, e_i]
            S[(d+0)*6+3]=1; S[(d+1)*6+4]=1; S[(d+2)*6+5]=1;
            // angular (local axis in world = columns of R1): [R1 e_i, 0]
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

    // ---- per-body spatial inertia about p ----
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

    // ---- CRBA: build about-p 6x6 spatial inertia per body, sum over subtree, then M = S^T Ic S.
    // (the (Iw,c,m) about-com form is NOT additive across bodies; the about-p 6x6 form is.)
    // 6x6 stored row-major in float[36].
    float Icp[G1_NBODY][36];
    for (int b=1;b<NB;++b){
        // about-p blocks: A = Iw + m(|c|^2 I - c c^T); B = m skew(c); D = m I
        float m=I[b].m; V3 c=I[b].c; float cc=dot3(c,c);
        float A[9];
        for(int i=0;i<9;++i)A[i]=I[b].Iw.m[i];
        A[0]+=m*(cc-c.x*c.x); A[1]+=m*(-c.x*c.y); A[2]+=m*(-c.x*c.z);
        A[3]+=m*(-c.y*c.x);   A[4]+=m*(cc-c.y*c.y);A[5]+=m*(-c.y*c.z);
        A[6]+=m*(-c.z*c.x);   A[7]+=m*(-c.z*c.y); A[8]+=m*(cc-c.z*c.z);
        // skew(c) = [[0,-cz,cy],[cz,0,-cx],[-cy,cx,0]]
        float Bx[9]={0,-c.z,c.y, c.z,0,-c.x, -c.y,c.x,0};
        float* X=Icp[b];
        // rows 0..2 = [A | mB], rows 3..5 = [mB^T | mI]; B^T=-B (skew)
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            X[6*i+j]=A[3*i+j];
            X[6*i+(j+3)]=m*Bx[3*i+j];
            X[6*(i+3)+j]=-m*Bx[3*i+j];
            X[6*(i+3)+(j+3)]=(i==j)?m:0.f;
        }
    }
    for (int b=NB-1;b>=1;--b){ int par=body_parentid[b]; if(par>=1)
        for(int k=0;k<36;++k) Icp[par][k]+=Icp[b][k];
    }
    // M[a][c] = S_a^T Icp[body(a)] S_c, for c in dofs of body(a) and its ancestors
    for (int i=0;i<NV*NV;++i) Mout[i]=0.f;
    for (int b=1;b<NB;++b){
        int da=body_dofadr[b], dn=body_dofnum[b];
        for (int a=da; a<da+dn; ++a){
            // F = Icp[b] * S_a  (6-vec)
            float F[6];
            for(int i=0;i<6;++i){ float s=0; for(int k=0;k<6;++k) s+=Icp[b][6*i+k]*S[a*6+k]; F[i]=s; }
            int j=b;
            while (j>=1){
                int dj=body_dofadr[j], djn=body_dofnum[j];
                for(int cc2=dj; cc2<dj+djn; ++cc2){
                    float v=0; for(int k=0;k<6;++k) v+=S[cc2*6+k]*F[k];
                    Mout[a*NV+cc2]=v; Mout[cc2*NV+a]=v;
                }
                j=body_parentid[j];
            }
        }
    }
    // armature on diagonal
    for (int a=0;a<NV;++a) Mout[a*NV+a]+=dof_armature[a];

    // ---- RNE for bias (qacc=0, gravity via base accel = -g), mirroring MuJoCo ----
    // Incremental velocity + cdof_dot: Sdot[a] = (partial cvel) xm S[a], where the
    // partial velocity is parent + earlier dofs of this body (NOT the full body vel).
    V3 vw[G1_NBODY], vv[G1_NBODY], aw[G1_NBODY], av[G1_NBODY];
    V3 Sdw[G1_NV], Sdv[G1_NV];
    vw[0]={0,0,0}; vv[0]={0,0,0};
    aw[0]={0,0,0}; av[0]={-G1_GRAVITY[0],-G1_GRAVITY[1],-G1_GRAVITY[2]}; // -g
    for (int b=1;b<NB;++b){
        int par=body_parentid[b];
        V3 w=vw[par], v=vv[par];   // partial velocity, starts at parent
        int da=body_dofadr[b], dn=body_dofnum[b];
        if (dn == 6) {
            // free joint: 3 translation dofs accumulate normally; the 3 rotational
            // dofs' cdof_dot all use the post-translation partial velocity (they do
            // NOT self-accumulate among each other before their cdof_dot) -- matches MuJoCo.
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

// ---------------- solve: dense Cholesky M x = b (M SPD), in place ----------------
__device__ void chol_solve(float* M, const float* b, float* x) {
    const int N = G1_NV;
    // factor: lower L in M's lower triangle
    for (int i=0;i<N;++i){
        for (int j=0;j<=i;++j){
            float s=M[i*N+j];
            for (int k=0;k<j;++k) s-=M[i*N+k]*M[j*N+k];
            if (i==j) M[i*N+i]=sqrtf(s);
            else      M[i*N+j]=s/M[j*N+j];
        }
    }
    // L y = b
    for (int i=0;i<N;++i){ float s=b[i]; for(int k=0;k<i;++k) s-=M[i*N+k]*x[k]; x[i]=s/M[i*N+i]; }
    // L^T x = y
    for (int i=N-1;i>=0;--i){ float s=x[i]; for(int k=i+1;k<N;++k) s-=M[k*N+i]*x[k]; x[i]=s/M[i*N+i]; }
}

// ---------------- quaternion integration: q <- q (x) exp(w*dt), w in local frame ----------------
__device__ void quat_integrate(float* q, V3 w, float dt) {
    V3 v={w.x*dt, w.y*dt, w.z*dt};
    float ang=sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);
    Q4 dq;
    if (ang<1e-9f) { dq={1.f, 0.5f*v.x, 0.5f*v.y, 0.5f*v.z}; }
    else { float s=sinf(0.5f*ang)/ang; dq={cosf(0.5f*ang), v.x*s, v.y*s, v.z*s}; }
    Q4 q0={q[0],q[1],q[2],q[3]};
    Q4 qn=qnorm(qmul(q0,dq));
    q[0]=qn.w; q[1]=qn.x; q[2]=qn.y; q[3]=qn.z;
}

// semi-implicit Euler position update (qvel already advanced)
__device__ void integrate_pos(float* qpos, const float* qvel, float dt) {
    for (int j=0;j<G1_NJNT;++j){
        int t=jnt_type[j], qa=jnt_qposadr[j], da=jnt_dofadr[j];
        if (t==0){
            qpos[qa]+=dt*qvel[da]; qpos[qa+1]+=dt*qvel[da+1]; qpos[qa+2]+=dt*qvel[da+2];
            quat_integrate(&qpos[qa+3], (V3){qvel[da+3],qvel[da+4],qvel[da+5]}, dt);
        } else if (t==3){
            qpos[qa]+=dt*qvel[da];
        }
    }
}
