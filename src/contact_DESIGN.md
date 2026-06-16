# contact_DESIGN.md — foot-ground contact solver for g1.cu

Status: single-world numpy prototype (`scripts/contact_np.py`) validated vs the MuJoCo
oracle to **1e-13 qacc** (fp64, foot-ground contacts, PGS to convergence). This doc is the
spec for the CUDA port. The thesis-decider: hold the smooth-dynamics speedup WITH contacts
at matched accuracy vs MJX's full-physics 3.1e5 env-steps/s.

## 1. The contact problem (what the oracle actually is)

MuJoCo config (models/scene.xml, probed empirically — see report):
- solver = **Newton**, cone = **PYRAMIDAL**, integrator = implicitfast, dt = 0.002, impratio = 1.0
- Feet: **8 collision spheres** (4 per foot), radius **5 mm**, on `left_ankle_roll_link`
  (body 7) and `right_ankle_roll_link` (body 13). Local positions:
  `(-0.05,±0.025,-0.03)` and `(0.12,±0.03,-0.03)`.
- Ground = z=0 plane (`floor`, geom 0).
- condim **3** (normal + 2 friction, no torsional/rolling), friction **mu = 0.6**
  (foot geom priority 1 > floor priority 0, so foot params win).
- solref = `[0.02, 1.0]` (timeconst, dampratio), solimp = `[0.9, 0.95, 0.001, 0.5, 2.0]`.

For a humanoid drop-and-settle, the ONLY contacts that matter for locomotion are these
foot-ground ones. (Perturbed reference states can also produce arm/leg self-collisions;
those are OUT OF SCOPE for the foot-ground solver — handle via a separate broadphase later.)

## 2. Collision detection (sphere vs z=0 plane) — from scratch, bit-exact

For each foot sphere with world center `c = xpos[bid] + R[bid] @ local_pos` (FK already
computed by the smooth path), radius `r`:
- penetration  `dist = c_z - r`   (contact iff `dist < 0`)
- contact point `pos = [c_x, c_y, dist/2]`   (MuJoCo midpoint surface<->plane)
- contact frame: `normal=[0,0,1]`, `t1=[0,1,0]`, `t2=[-1,0,0]` (MuJoCo's deterministic
  tangent basis for a +z normal — matters because it sets J row content).

Verified `dist`, `pos`, and the resulting J match MuJoCo's `contact.dist/pos` and `efc_J`
to 0.0 (bit-exact). Cheap, branchless, embarrassingly parallel: 8 spheres/world, no
broadphase needed for foot-ground.

## 3. Contact Jacobian J (pyramidal, condim 3)

Per contact, 4 constraint rows (MuJoCo pyramidal layout):
```
row0 = jn + mu*jt1     row1 = jn - mu*jt1
row2 = jn + mu*jt2     row3 = jn - mu*jt2
```
where `jn = normal @ Jp`, `jt1 = t1 @ Jp`, `jt2 = t2 @ Jp`, and `Jp` (3×nv) is the
**translational point Jacobian** of the contact point fixed to the foot body — the standard
articulated-body point Jacobian (column k = contribution of dof k to world-point velocity).
`Jp` is built from the same per-body motion axes `S` the ABA path already computes (world
axes about pelvis); the point-Jacobian column for dof k is `S_k.ang × (pt - anchor) + S_k.lin`.
=> J reuses ABA primitives, no new heavy kinematics. nefc = 4 * ncon.

## 4. The constraint solve (PGS over the boxed QP)

MuJoCo's contact force solves the regularized problem
```
min_f  0.5 f^T A f + f^T b ,   f >= 0           (pyramidal => each row lower-bounded at 0)
A = J M^-1 J^T + diag(R)        (R = efc_R, soft-constraint regularization)
b = J * qacc_smooth - aref      (aref = efc_aref, reference accel from solref/solimp)
```
Then `qfrc_constraint = J^T f` and `qacc = qacc_smooth + M^-1 J^T f`. Both verified
bit-exact (`J^T f` to 6e-14, `qacc` reconstruction to 3e-13).

**Solver = Projected Gauss-Seidel** (proven to reproduce MuJoCo's Newton solution to 1e-13
at convergence). Per sweep, per row i: `f_i = max(0, -(b_i + A_i·f - A_ii f_i)/A_ii)`.
- mid-bucket (4-8 contacts, light penetration): converges in ~50-200 sweeps.
- hi-bucket (deep penetration, stiff): needs ~1000-5000 sweeps for 1e-8 (see go/no-go).
- M^-1 J^T comes from the ABA factorization (we already have M^-1 action via ABA); A is
  small (nefc ≤ 32 for foot-ground, ncon ≤ 8). Building A = J M^-1 J^T needs ncon back-solves
  through ABA — cheap. PGS on a ≤32×32 SPD-ish matrix is trivial per world.

**Open correctness item (the one remaining gap):** R and aref are currently taken FROM the
oracle. They must be reconstructed from solref/solimp:
- impedance `imp = solimp_sigmoid(|dist|)` (MuJoCo's `mju_makeImpedance`: dmin=0.9, dmax=0.95,
  width=0.001, midpoint=0.5, power=2.0).
- `R = (1 - imp) / imp / diag(J M^-1 J^T)`, with an **impratio** correction on the pyramidal
  friction rows (naive formula was off ~1e3 → the impratio/cone scaling is the missing piece).
- `aref = -B*(J qvel) - K*imp*dist` with K,B from solref (timeconst/dampratio).
First-pass reconstructions did not match (R off ~1e3, aref off) — this is the immediate next
correctness task before the CUDA port. It is pure scalar algebra per contact, no new structure.

## 5. Coupling to ABA forward dynamics

`qacc_smooth` (no contacts) already comes from the validated ABA path. Contact adds a
correction: `qacc = qacc_smooth + M^-1 (J^T f)`. Two clean options for the CUDA kernel:
- (A) form `B = J M^-1` by running ABA's M^-1 action on each J row (ncon×4 solves), build
  `A = B J^T + diag(R)`, PGS for f, then `qacc += B^T f`. Reuses ABA, ~32 extra back-solves.
- (B) treat contact impulses as external spatial forces on the foot bodies and re-run ABA
  with `f_ext` (one extra ABA pass per PGS-outer-iteration) — MuJoCo-style but more passes.
Start with (A): the contact system is tiny (≤32 rows) and decoupled from the per-body recursion.

## 6. SIMT mapping (bucketing) — onto the 3090, from the divergence profile

Profile (bench/contact_ref.npz): ncon mean 5.7, max 25; buckets ballistic 8% / 1-2 9% /
3-4 23% / 5-8 42% / tail>8 17.5%. A 32-world warp's max-ncon averages 14.3 → lane=world
wastes ~2.5x if every lane runs the warp's worst case.

Plan (matches CLAUDE.md design rules):
- **1 world per lane** for collision detection + J build (8 spheres, fully uniform, zero
  divergence — every world checks the same 8 spheres regardless of contact count).
- **Hard-bucket by ncon before the solve.** Fixed PGS iter budget per bucket
  (ballistic: skip; 1-2: ~32 sweeps; 3-4: ~64; 5-8: ~128; tail>8: separate queue with a
  larger budget or 1-warp-per-hard-world). `__ballot_sync` early-exit when a warp's worlds
  all converge. tail>8 is 17.5% (> the 5-10% pathological threshold) → tail worlds get a
  dedicated queue so they don't stall the common case. NOTE: with FOOT-GROUND ONLY contacts,
  ncon ≤ 8 always (8 spheres), so the >8 tail in the profile is dominated by arm/leg self-
  collisions — i.e. the foot-ground solver is naturally bounded at nefc ≤ 32 and the
  pathological tail largely disappears. This is GOOD news for the thesis (see report).
- A is ≤32×32 and lives in registers/SMEM per world; PGS is scalar, no tensor cores.

## 7. Validation ladder (mirror the smooth-dynamics build)

1. [DONE] numpy fp64 single-world, R/aref from oracle: qacc to 1e-13 at PGS convergence.
2. [NEXT] reconstruct R/aref from solref/solimp → fully self-contained numpy (no oracle).
3. CUDA single-world: port collision+J+PGS, validate vs the numpy oracle (fp32 tolerance).
4. CUDA batched + bucketing, full mj_step trajectory vs bench/contact_ref.npz qpos drift.
5. Benchmark env-steps/s vs MJX 3.1e5 full-physics. Go/no-go: ≥15% end-to-end win at
   matched accuracy, no pathological-queue explosion.

## 8. Files
- `scripts/gen_contact_solve_ref.py` — generates `bench/contact_solve_ref.npz` (solver oracle:
  qacc, qfrc_constraint, efc_J/R/aref, contact geometry, per ncon bucket; pure foot-ground via
  FRICTIONLOSS+LIMIT disabled).
- `scripts/contact_np.py` — fp64 numpy prototype (collision + J + PGS from scratch).
- `src/contact.cuh` — (scaffold) device collision + J + PGS, to be wired to aba.cuh.
