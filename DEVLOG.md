# DEVLOG — g1.cu

## 2026-06-13 — Bootstrap, design review, and move to gamer
Idea: GPU humanoid RL is "billions of steps in hours" but far from peak FLOPs (serial physics
critical path, low arithmetic intensity, tiny per-world state ~60-100 floats). Hypothesis: a
persistent warp-specialized kernel fusing physics + policy, with a SMEM ring buffer decoupling
fine-grained physics SIMT from coarse-grained tensor-core batching, could beat bulk-synchronous.

Delegated a design review to codex (gpt-5.5, web-grounded vs PTX ISA + CUTLASS docs). Verdict
in CLAUDE.md "Design rules" + "Go/no-go". Headline: fused single-kernel likely LOSES to
"persistent physics + batched policy microkernel" unless policy inference is 20-40%+ of step
time; build the cheap baseline first; >=15% end-to-end or kill it.

Originally scaffolded on anvil for sm_120 Blackwell. MOVED to gamer (RTX 3090, sm_86, CUDA 13.3)
because it is the only free GPU. Hardware-specific consequences of the move:
- setmaxnreg GONE (sm_90+ only) -> static register control is now mandatory, not a fallback.
- wgmma/tcgen05 GONE -> tensor cores via warp-level mma.sync, Ampere shapes (m16n8k16 bf16).
- cp.async + mbarrier still present (sm_80+) -> the persistent + ring-buffer design survives.
- 24GB VRAM (vs 96GB) and a 3090's weaker tensor cores make the fusion win bar harder, not easier.
The kernel-design rules from the review (two rings, aggregator warp, contact bucketing, policy
cadence) are arch-independent and still hold.

## 2026-06-13 — M0 DONE: single-world fwd dynamics fp32 validated vs MuJoCo
Implemented from-scratch articulated-body smooth dynamics for the G1 (nq=36, nv=35, 31
bodies, free base + 29 hinges) and matched MuJoCo to fp32 precision. Reference: MuJoCo
3.9.0, model `models/g1.xml` (mujoco_menagerie), Euler integrator, CONSTRAINT + ACTUATION
disabled -> pure `M*qacc = -qfrc_bias`. Engine: `src/dynamics.cu` (single-thread kernel),
Featherstone CRBA + RNE in a common world-axes frame referenced at the pelvis origin.

Build order followed (validated each gate before the next): FK -> CRBA (M) -> RNE (bias)
-> solve (dense fp32 Cholesky) -> semi-implicit Euler w/ quaternion integration -> 300-step
rollout. Results vs MuJoCo fp64: M rel 2.5e-7, bias rel 1.2e-7, qacc rel 2.5e-6, and the
qpos trajectory tracks to <=8e-7 (base pos), <=4e-7 (quat), <=5e-7 (joints) over 300 steps,
flat (no divergence). `tests/test_m0.py` (4 tests) is the acceptance gate.

Three gotchas cost the most time (all empirically pinned, not from memory):
- Free-joint qvel basis = WORLD linear + BODY-LOCAL angular. Building the dof motion axes
  this way makes M and bias match MuJoCo element-wise (basis-dependent quantities).
- MuJoCo `cdof_dot` quirk: the free joint's 3 rotational dofs cross with the post-TRANSLATION
  partial velocity (they don't self-accumulate), so their angular part is zero. A naive
  per-dof accumulation corrupts base accel and propagates to every body. See the memory note.
- MuJoCo joint LIMITS are active even mid-air (nefc=29) -> disabling only CONTACT left a
  ~1.33-norm constraint force. Must disable the whole CONSTRAINT solver for clean smooth dyn.
Debug oracle `scripts/ref_dynamics_np.py` (fp64 numpy port) matches MuJoCo to 1e-17 and was
key to separating formula bugs from fp32 noise.

## 2026-06-13 — M1: bulk-synchronous batched baseline (the number to beat)
Refactored the validated device dynamics into `src/dynamics.cuh` (shared header); `dynamics.cu`
is now the M0 harness, `sim.cu` is the M1 batched baseline. Mapping: 1 thread = 1 world, state
(qpos,qvel) in global memory, host issues ONE step-kernel launch per timestep (the
bulk-synchronous pattern). Reuses the M0 device code verbatim.

Correctness: world 0 matches MuJoCo to ~5e-7 (same as M0); all worlds bit-identical
(determinism maxdiff = 0). No contacts yet => zero warp divergence (all worlds same control flow).

Throughput sweep (RTX 3090, 300 steps, fp32), env-steps/s:
  N=1024 1.0e6 | 4096 3.2e6 | 16384 4.9e6 | 65536 7.4e6 (peak) | 262144 6.9e6.
Peaks ~7.4M env-steps/s at N~65k, then drops (local-memory pressure). ptxas: 56 regs, 0 spills,
15648 B stack/thread. NOT register-bound; the bottleneck is 15.6 KB/thread of local memory
(dense 35x35 M = ~4.9 KB) overflowing L1 -> local-mem traffic to L2/DRAM. There is no GEMM/policy
in this baseline yet (NN arrives at M4), so this is the physics-only number.

THE NUMBER TO BEAT: ~7.4e6 env-steps/s. Honest caveat: this naive baseline is local-memory bound;
M2 should optimize it (SMEM-resident M, warp-cooperative solve) before it is a fair fusion target,
else beating it is meaningless.

## 2026-06-13 — Competitive benchmark: ours vs MJX vs MuJoCo Warp vs MuJoCo CPU
Validation strategy = "specialized GPU sim, validated against a trusted oracle". Measured the
real competition on the SAME G1 (env-steps/s, RTX 3090 for GPU rows, Ryzen 7700X for CPU):

  smooth-only (contacts OFF, = our config, the apples-to-apples row):
    ours (naive CUDA fp32)   7.4e6   <- ~3x both production GPU engines
    MJX (jax 0.10)           2.5e6
    MuJoCo Warp (1.14)       2.4e6
    MuJoCo C, 16-thread      3.1e5
    MuJoCo C, 1-core         7.7e4
  full-physics (contacts ON, NOT comparable to ours yet -- we have no contacts):
    MJX                      3.1e5   <- the contact bar to beat
    MuJoCo Warp              2.2e5   (drops to 0.85e5 at N=65k; default nconmax, maybe untuned)
    MuJoCo C, 16-thread      1.5e5
Caveat: Warp numbers use default settings / per-step Python loop, may not be Warp-optimal;
Warp here is ~MJX on smooth and a bit slower on contacts (early mujoco_warp 3.9.0.1).
Scripts: scripts/bench_mjx.py, bench_warp.py, bench_mujoco.py.

Contact benchmark oracle set up (scripts/gen_contact_ref.py -> bench/contact_ref.npz): G1
passive drop-and-settle on a floor (Newton, pyramidal cone). Divergence profile across 4000
mixed-phase states (proxy for parallel worlds): ncon mean 5.7, max 25; buckets ballistic 8% /
1-2 9% / 3-4 23% / 5-8 42% / tail>8 17.5%. A 32-world warp's MAX ncon averages 14.3 -> naive
lane=world SIMT wastes ~2.5x (does max, not mean). Tail >8 at 17.5% exceeds the 5-10%
pathological threshold -> the solve phase likely needs world-sorting / 1-warp=1-hard-world.

STRATEGIC READ: specialization wins ~3x on smooth dynamics (the easy part). The contact bar is
~3.1e5 (MJX). If contacts cost us the same ~8x they cost MJX (2.5e6->3.1e5), we'd land ~9e5,
still ~3x ahead -- but that is an extrapolation; divergence (~2.5x) + matching a tuned contact
solver at accuracy is unproven. THE experiment that decides the company thesis = implement the
specialized contact solver and measure vs MJX at matched accuracy.

## 2026-06-13 — Compiler-tax experiment: specialized dynamics in Warp vs hand-CUDA
Ported our EXACT specialized smooth-dynamics algorithm (FK->CRBA->RNE->Cholesky->Euler,
hardcoded G1, dense, fp32, 1 thread/world) to NVIDIA Warp (scripts/warp_sim.py, ~1 hour vs
~1 day for the CUDA). Warp validates to ~7e-7 vs MuJoCo -- identical accuracy to hand-CUDA.
Throughput (RTX 3090, 300 steps, smooth, env-steps/s):
    specialized hand-CUDA   7.4e6   (peak N=65k)
    specialized Warp        6.0e6   (peak N=262k; 5.5e6 at 65k)
    general MuJoCo-Warp      2.4e6
    MJX                      2.5e6
CUDA-graph capture changed nothing (6.21e5 vs 6.15e5 at N=1k) -> the Warp gap is real kernel
difference, not Python launch overhead (kernel is compute-bound even at N=1k: 15 KB local +
dense Cholesky).

DECOMPOSING THE 3x MOAT: ~2.5x is SPECIALIZATION (specialized Warp 6.0e6 / general 2.4e6),
recoverable in Warp at ~10x less code and validated to 1e-7 in ~1 hour. Only ~1.23x is
hand-writing CUDA on top (7.4e6 / 6.0e6). => ~80% of the moat is accessible WITHOUT writing
CUDA. TOOLING DECISION MADE: build in Warp; drop to hand-CUDA only for the last ~23% if it
ever matters. This empirically confirms the recommendation (prototype in Warp, not raw CUDA).

## 2026-06-14 — Profile-guided optimization of the hand-CUDA baseline (ncu)
Installed nsight-compute (pacman). Profiled src/sim.cu step_kernel.
Speed-of-light: ~tens of KFLOP/world/step -> <0.1% of 3090 fp32 peak. NOT compute-bound.
ncu @ N=8192: compute SM 4%, 86% of warp cycles stalled on long-scoreboard (local-mem) deps.
ncu @ N=65536 (real operating point): DRAM throughput 82%, compute SM 9%, achieved occupancy
51% (24 warps/SM, near the 66% register cap). => DRAM-BANDWIDTH BOUND: the 15.6 KB/thread local
working set x ~64k resident threads (~1 GB) >> 6 MB L2, so it streams through DRAM.
Cheap knobs, all measured, all flat-or-worse (no insane result):
  block size 32/64/128/256 -> 6.7/7.36/7.32/7.35e6 ; __launch_bounds__ no effect
  maxrregcount 32/40/48/64 -> 6.88/6.87/7.23/7.36e6 (forcing occupancy spills more, hurts)
Conclusion: dense per-thread kernel is at its ceiling (~7.4e6), bound by local-mem spill
bandwidth. Inline PTX is the WRONG tool here (compute is 9% idle, not instruction-bound). The
only real lever is shrinking the per-thread working set: (a) pack M lower-tri + symmetric Icp
(~10-15%, mechanical), (b) the real multi-x = sparse LTDL exploiting the tree (M 1225->~200 nnz,
solve O(nv*depth) not O(nv^3)) -> smaller footprint + shorter dep chains. That is an algorithmic
rewrite done ONCE, and it helps in Warp too. sim.cu now has -DBLOCK / -DLB knobs.

### Next (thesis-deciding build, now in Warp)
Implement the specialized foot-ground contact solver IN WARP (validate vs bench/contact_ref.npz),
sort worlds by contact bucket to claw back the ~2.5x SIMT waste, benchmark vs MJX full-physics
(3.1e5). Hold ~3x with contacts -> real moat. Collapse -> cool kernel, no moat.

## 2026-06-14 — ABA (O(N) articulated-body) kernel: 3x over the dense baseline
User opted to pay the CUDA tax and push the smooth-dynamics kernel. Replaced dense CRBA +
O(nv^3) Cholesky with Featherstone ABA in the common world-axes frame (src/aba.cuh): pass1
velocities+bias-accel, pass2 articulated inertia leaf->root (scalar D per hinge), pass3
accelerations with ONE 6x6 solve at the free base. No dense M, no 35x35 Cholesky.

THE BUG (cost most of the session): ABA was missing dof_armature. Symptom: gravity-only matched
oracle to 1e-14 but velocity/force was off ~17%. Isolated via fp64 numpy oracle (scripts/aba_np.py,
aba_solve_test.py): reconstructed M from aba_solve(e_k) and the ONLY error was the hinge-hinge
diagonal, off by exactly 0.01 = the armature. Gravity masked it because it barely excites the
tiny-inertia distal joints where armature dominates. Fix: D_i = S^T I^A S + dof_armature[i] (all
three places: pass2, pass3 hinge, pass3 base diagonal). After fix: numpy 7e-15, CUDA fp32 2.6e-7
qacc, 300-step trajectory <=7.6e-7 vs MuJoCo (matches the dense kernel exactly).

Resources: 9952 B stack/thread (vs dense 15648, -36%), 128 regs, 0 spills.
Throughput (RTX 3090, batched, src/sim_aba.cu, env-steps/s):
  N=16384 ksteps=16 -> 2.27e7 (PEAK). vs dense peak 7.4e6 => ~3.1x. ~9x MJX (2.5e6), ~3.8x
  specialized-Warp (6.0e6). Multi-step on-chip (ksteps 1->16) adds ~7-16% (amortizes global I/O;
  the per-step ~10KB working set still spills, so it's the secondary lever -- ABA footprint +
  short dependency chains are the primary win). `just`-style: ./build/sim_aba [nsteps] [N] [ksteps].

## 2026-06-14 — Cooperative on-chip ABA (warp=world, lane=body, SMEM): MEASURED LOSS
Took the shared-memory shot to kill the DRAM spill. src/sim_aba_coop.cu: warp=world, lane=body,
per-body state in SMEM, tree walked in depth-waves (siblings parallel across lanes, parent<->child
via SMEM, pass2 accumulation via atomicAdd to parent, one 6x6 base solve on lane 1). Correct
(world-0 vs MuJoCo 7.6e-7, deterministic). But PEAK 1.16e7 env-steps/s -- ~half the 1-thread/world
ABA (2.25e7). Why it loses (honest prior held): (1) G1's tree is narrow (2-4 siblings/level) so
~28 of 32 lanes idle within each depth-wave -> poor SIMT utilization; (2) ~9.5 KB SMEM/world caps
occupancy to ~8 worlds/SM (16%), can't hide the serial depth-wave latency; (3) ~11 waves x 3
passes of __syncwarp barriers. Removing the DRAM spill doesn't beat full-SIMT-width 1-thread/world
at saturated DRAM. CONCLUSION: per-world humanoid physics is small+serial enough that 1 world/lane
(SIMT) is the right mapping; cooperation wastes the SIMT width. Best kernel = 1-thread/world ABA,
2.25e7 (3x dense), oracle-matched.

## 2026-06-14 — On-chip take 2: 1-thread/world + working set in SMEM (the right reframing)
Insight (user): keep 1-thread/world (full SIMT) but stage the per-thread working set in SHARED
memory so steps don't spill to DRAM. Refactored aba_qacc to take a `scr` pointer (local OR smem);
src/sim_aba_smem.cu. Correct (3.6e-7). Working set 8280 B/thread -> max ~12 threads/SM.
Result (N=65536): tpb=12 ksteps=16 -> 1.55e7. Multi-step helps MORE at low occupancy (+30% vs
local's +7%) but still BELOW local-scratch 2.25e7.
DEFINITIVE reason (measured across all 3 variants): local-scratch ABA runs ~438 worlds/SM (local
mem is NOT an occupancy limiter -- it's DRAM-backed; registers cap occupancy at 28%=13.7 warps).
ANY on-chip variant collapses to <=12 worlds/SM (SMEM-capacity bound). That 36x parallelism loss
dwarfs the DRAM savings -> "DRAM-bound but massively parallel" BEATS "on-chip but starved".
Ranking: local 2.25e7 > smem-scratch 1.55e7 > cooperative 1.16e7. On-chip residency conclusively
ruled out for this footprint on sm_86. Only lever left = cut LOCAL footprint (keeps occupancy,
cuts DRAM): symmetric IA packing (6x6->21, ~20% -> ~2.7e7).

## 2026-06-14 — Symmetric IA packing: 4x dense (the predicted lever, delivered)
Spatial/articulated inertia is symmetric -> store 21 packed instead of 36 (symidx + symmv6 in
aba.cuh; aba_qacc IA now [NB*21]). Stack 9952->8128 B/thread (-18%), occupancy unchanged (still
register-limited). Correct (qacc 2.6e-7, traj 7.6e-7 -- identical). Throughput PEAK 3.01e7
env-steps/s (N=16384, ksteps=16), up from 2.25e7 = +34% (more than the footprint cut: symmetric
matvec/downdate also do less compute). Multi-step still adds ~18% here (ksteps 1->16).
FINAL smooth-dynamics ranking (RTX 3090, env-steps/s): packed-ABA 3.01e7 > ABA 2.25e7 > dense
7.4e6 > specialized-Warp 6.0e6 > MJX 2.5e6 > MuJoCo-Warp 2.4e6.
=> 4.07x the dense baseline, ~12x MJX, oracle-matched to fp32. This is the smooth-dynamics
ceiling for the 1-thread/world design on sm_86 (DRAM-bound; on-chip ruled out). Best kernel =
src/aba.cuh + src/sim_aba.cu.

## 2026-06-14 — Carmack-mode first-principles pass: SASS + fp16 scratch
Reframed the bottleneck from first principles: the persistent state is only 71 floats
(q,qd); the ~8KB "working set" is TRANSIENT scratch that spills to DRAM because (1) it's
runtime-indexed by the tree (par=body_parentid[b]) -> forced to local memory, and (2) the
~650-float articulated-inertia set exceeds the 255-register file across the leaf->root /
root->leaf pass boundary. SASS (cuobjdump) confirmed: 518 local ops (343 scalar) vs 1323
FFMA -- ~40% of work is spilled-scratch traffic. ABA is FLOP-minimal (CPU objective); on a
GPU with ~10x spare compute and no spare bandwidth it's the wrong objective.
- #pragma unroll the body loops: WORSE (255 regs, +2.3KB new spills) -- compile-time b
  isn't enough; par is runtime. Confirms the tree index is the root cause.
- fp16 storage of IA (compute stays fp32; src/aba_fp16.cuh, sim_aba_fp16.cu): stack
  8128->6832 B, no spills. Accuracy RL-grade (300-step joint drift 2.7e-3 rad ~0.15deg,
  base_pos 7e-6). PEAK 3.90e7 env-steps/s (N=16384) = +29% over fp32-packed 3.01e7.
  ~5.3x dense, ~15.6x MJX. Validated thesis: cut scratch traffic -> proportional speedup.
Remaining ladder: (a) extend fp16 to vel/cb/pA/a6 (more, precision-sensitive); (b) the
structural swing = codegen the kernel with the tree baked as compile-time literals (kills
runtime-indexed local ops entirely) or a matrix-free CG/streaming DFS with register-resident
state. aba.cuh (fp32, bit-accurate) stays the default; fp16 is the RL-throughput mode.

## 2026-06-14 — Full fp16 scratch: ~6.4x dense / ~19x MJX
Extended fp16 from IA to ALL scratch (vel/cb/pA/a6) in src/aba_fp16.cuh (ld6/st6 helpers,
compute fp32). Stack 8128->5344 B. Accuracy barely changed (300-step joint drift 3.3e-3 rad
vs 2.7e-3 for IA-only -- the integration-critical velocity/force arrays tolerate fp16). PEAK
4.71e7 env-steps/s (N=16384) = +20% over fp16-IA, +56% over fp32-packed. ~6.4x dense, ~19x MJX.
fp16 ladder: fp32 3.01e7 -> fp16-IA 3.90e7 -> fp16-all 4.71e7. Stack still 5344B (>1KB reg file)
so it still spills; the remaining structural swing is codegen (tree baked as compile-time
literals -> kill runtime-indexed local ops) -- uncertain payoff now that fp16 halved the traffic.

## 2026-06-14 — Smooth kernel CLOSED + CONTACT THESIS PROVEN (3-way agent fan-out, then contacts)
Optimization past fp16-all (47.1M) settled with two experiments, run in parallel:
- RECOMPUTE-in-pass3 (src/aba_recompute.cuh): fuse pass1a+1b, drop stored `cb` bias-accel
  array, recompute it fp32-from-fp16-vel. VALID (300-step drift 2.93e-3 rad, BEATS fp16-all's
  3.32e-3; stack 5344->4960 B). Only +1.4% peak / +3.6% at N=65k, -12% at small N. NEUTRAL,
  below the 15% go/no-go bar.
- CODEGEN straight-line (scripts/gen_aba_kernel.py -> src/aba_codegen.cuh, 4027 lines): bake
  whole G1 tree as compile-time literals, per-body quantities as named scalars. VALID (bitwise
  identical to fp16). But 0.72x = clear LOSS: 507->5490 SASS local ops, 128->255 regs, spills
  appeared. DEAD END. Decisive lesson: the smooth-kernel wall is the SIZE of the live working
  set (651-elt IA + scratch >> 255-reg file), NOT the runtime-ness of indices. Baking literals
  doesn't shrink the set -> ptxas spills every scalar anyway and loses loop-locality (same
  mechanism as the earlier #pragma-unroll regression). 47.1M is the smooth-kernel ceiling.

CONTACTS (the thesis-decider): MuJoCo foot-ground model pinned (Newton, pyramidal, condim 3,
mu 0.6, solref [0.02,1.0], solimp [0.9,0.95,...], impratio 1). Exported model had ZERO collision
data -> extended scripts/export_model.py (additive foot-geom constants). New solver oracle
bench/contact_solve_ref.npz (scripts/gen_contact_solve_ref.py). fp64 numpy prototype matched
MuJoCo to machine precision. R/aref reconstruction CLOSED (scripts/contact_np2.py): computed
from solref/solimp, match oracle 4e-15/1.8e-14; the missing piece was pyramidal diagApprox =
(invw0_body+invw0_floor)*(1+mu^2)*(2*mu^2/impratio). Batched CUDA full-physics kernel
(src/sim_contact.cu, 1-thread/world): FK -> dense M (CRBA)+Cholesky factor-once -> qacc_smooth
-> 8 sphere-plane detect -> pyramidal J (ANCESTOR dofs only -- bug fix) -> on-device R/aref ->
A=J M^-1 J^T+diag(R) via reused factor -> PGS -> qacc=qacc_smooth+M^-1 J^T f. Validated fp32 vs
oracle worst qacc relerr 2.24e-5, deterministic. pytest 7/7.
FULL-PHYSICS env-steps/s (RTX 3090): N=65536 5.52e6 = 17.8x MJX 3.1e5, 25x MuJoCo-Warp 2.2e5.
THESIS PROVEN (foot-ground). TWO CAVEATS: (1) apples-to-oranges -- we solve only foot-ground;
MJX simulates the full contact set incl. self-collision. (2) the contact kernel uses dense M
(CRBA)+Cholesky, NOT the fast ABA -> abandoned the ABA advantage to get M^-1 for the QP, so
full-physics dropped from 47.1M smooth to 5.52M. Big headroom: apply M^-1 via ABA's articulated
factorization instead of dense CRBA. Next: ABA-based M^-1 in the QP; scope self-collision;
M5 PPO + walk-to-finish-line.

## 2026-06-14 — ABA-M^-1 in the contact QP: full-physics 5x faster (~90x MJX)
Replaced the dense-M wall in the contact solver. The dense CRBA build + O(nv^3) Cholesky was
the bottleneck (PGS was already cheap). New src/aba_factor.cuh exposes aba_factorize() +
aba_solveM(): Featherstone's articulated inertias ARE a factorization of M, independent of
velocity/gravity/force, so factor once (leaf->root, store U/invD/articulated-inertia + the
free-base 6x6 factor) then apply M^-1 to any vector via a zero-bias pass2/pass3 driven by x,
O(nbody) per vector (MuJoCo mj_solveM-equivalent). Used for qacc_smooth, each J^T column of
A=J M^-1 J^T+diag(R) (nefc<=32 applies), and the final M^-1 J^T f. aba.cuh untouched (the
factor is a derived sibling header).
Bug found+fixed: the leaf->root pA propagation must carry the child's own pA[b] forward, not
just U·(u/D); with cb=0 the apply is pa = pA[b] + U·(u/D). Gate-1 identity (src/aba_factor_test.cu):
solveM(tau-bias) == aba.cuh qacc to 2.6e-6, ||M·y-b||/||b|| = 4.4e-9.
Contact-path validation vs oracle: worst qacc relerr 2.239e-5 -- IDENTICAL to dense (2.240e-5);
the M^-1 swap doesn't touch accuracy (residual is PGS numerics). Stack 28352->27440 B (dense
35x35 M removed), compute-sanitizer clean, deterministic, pytest 7/7.
FULL-PHYSICS env-steps/s (RTX 3090, warmup'd): N=65536 2.78e7 (5.0x over dense-M 5.50e6,
~90x MJX 3.1e5), N=16384 2.72e7, N=4096 2.14e7 (8.7x dense). Contacts now cost only ~8% over
the fp32 smooth ABA (30.1M) -> nearly free. Remaining caveat: foot-ground only (not MJX's full
self-collision set) -- honest as "locomotion-relevant contacts." Next: scope self-collision,
maybe fp16 the contact path (smooth fp16 was 47.1M, contact path still fp32), then M5 PPO.

## 2026-06-15 — FUSION THESIS MEASURED = NO-GO; M5 PPO scaffold runs
Measured the question the whole project was built to answer: is policy inference a big
enough fraction of step time that fusing it into the physics kernel beats the bulk-synchronous
(physics-kernel | GEMM-kernel) baseline? SPEC's go/no-go threshold: 20-40%+.
Method (src/bench_nn.cu): physics = the validated full-physics sim_contact step (cudaEvent,
warmup, N sweep); NN = cuBLAS fp16 tensor-op GEMM chain (honest bulk GEMM-kernel baseline, no
torch tax), obs=100, act=29, policy sizes [128,128]/[256,256]/[512,256,128]. Cadence S =
physics substeps per policy action (locomotion S~10 at MuJoCo dt 0.002 / control 0.02).
frac_NN = t_NN / (S*t_phys + t_NN).
Numbers (N=65536, idle GPU): t_phys=2.386 ms (1 substep); t_NN medium=0.505 ms.
  frac_NN: S=1 17.5%, S=5 4.1%, S=10 2.07%. Even S=1 + largest policy only ~31%.
VERDICT: at the realistic locomotion regime (S=10) the NN is ~2% of step time -- an order of
magnitude below the fusion-worthwhile threshold. The physics is so fast (2.78e7 env-steps/s)
that the policy GEMM is a rounding error. Per the project's OWN go/no-go: STOP AT M1.5
(persistent physics + batched cuBLAS policy microkernel). Do NOT build the warp-specialized
ring-buffer fusion (M3/M4) -- it would chase a ~2% slice at enormous complexity. The original
research bet (fused persistent kernel) is answered NO by measurement. The project's WIN is the
validated specialized physics sim (~90x MJX full-physics for locomotion contacts), not fusion.

M5 PPO scaffold: ctypes env (src/g1_env.cu -> libg1env.so, reset/step fully on GPU, reuses the
ABA + foot-contact path unmodified; PD torque applied as M^-1*tau via one extra aba_solveM).
Position (PD) actuators exported (additive g1_model.h: G1_NU, g1_act_{dof,kp,kd,ctrl_lo/hi,qadr},
g1_qpos_stand). Minimal PPO (scripts/ppo.py, clip+GAE) runs end-to-end, reward moves (N=2048,
15 iters: ep-reward/step 0.571->0.775). NOT a finished walker.
GOTCHA: kp=500 position servo + stiff foot contact (solref=0.02s) is UNSTABLE under explicit
semi-implicit Euler at S>=2 (fine at S=1) -- MuJoCo uses implicitfast for exactly this. A real
walker needs implicit/semi-implicit PD integration (next physics task). Mitigated for now with
KP_SCALE=0.2 + velocity clamp. New files: src/bench_nn.cu, src/g1_env.cu, scripts/ppo.py. pytest 7/7.

## 2026-06-15 — Stage A: implicit PD integration (implicitfast-equivalent), stable at kp=500
Fixed the stiff-servo instability that blocked training. The PD servo (tau=kp(target-q)-kd*qd,
kp=500) + stiff foot contact diverged under explicit semi-implicit Euler at S>=2 substeps.
Fix: integrate the velocity-dependent damping IMPLICITLY (MuJoCo implicitfast): solve
(M + dt*B) qacc = -bias + tau with B = diag(kd_i + dof_damping_i), by folding dt*B into the
ABA articulated-inertia diagonal D in a new aba_factorize_damped() (src/aba_factor.cuh; the
plain aba_factorize is preserved for smooth/contact callers). Contacts are solved through the
SAME damped factor (A = J (M+dtB)^-1 J^T + R) for consistency. KP_SCALE=0.2 hack REMOVED;
full kp=500 is the default. aba.cuh untouched.
Validation (clean GPU): vs MuJoCo implicitfast oracle (scripts/gen_implicit_ref.py ->
test_implicit.py) single-step qvel err 1.32e-5, qpos 8.4e-8, final base z matches to 4 dp
(2.5917). Stability (test_stability.py, N=256, kp=500, S=10, 400 steps, hold stand): base
linvel bounded (max 0.042, decaying to 7e-4), zero NaN, zero falls. pytest 7/7.
New: scripts/gen_implicit_ref.py, scripts/test_implicit.py, scripts/test_stability.py,
src/test_implicit.cu. Edited: src/aba_factor.cuh (+aba_factorize_damped), src/g1_env.cu
(implicit integrator), scripts/export_model.py (+ damping/keyframe, additive).

## 2026-06-15 — M5: trained G1 walker (PPO on the fused sim)
With Stage A's stable implicit integrator, PPO learned a walking policy. Run #1 (reward v1)
REWARD-HACKED: forward-velocity reward accrued every step while a fall cost only -2.0 once, so
it lunged forward and faceplanted (vx~0.45, ep_len~16, 100% fall, plateaued). Diagnosis: classic
"suicidal sprint" -- forward reward >> upright/alive penalty.
Reward v2 (src/g1_env.cu, anti-dive): strong per-step ALIVE bonus (+1.0) so long upright episodes
dominate; forward reward GATED by uprightness (r_vel = vx_c * upc) so a pitched lunge earns ~0;
falling TERMINATES at -5.0 with no positive terms. Incentive ordering by design: walk-upright
(~2.1/step) > stand (~1.3) > dive (negative).
Run #2 (N=8192, 1500 iters, ~35 min): stand -> explore-falls -> walk-then-fall (iter ~200,
fwd 1.3m ep_len 78) -> SUSTAINED UPRIGHT WALKING. Converged (iters 1410-1499): ep_len 150.00
(full 3s episode), fall_rate 0.00, vx ~0.85 m/s (CMD_VX 0.8), fwd ~2.6m. Best ckpt iter 1490
fwd 2.697m, zero falls (models/ppo_ckpt/best.pt). The G1 walks forward at command speed without
falling. Reward curves: bench/reward_curve_run1.csv (hacked), models/ppo_ckpt/reward_curve.csv (run2).
M5 essentially done: fast validated sim -> trained walker. Remaining: visual gait check (user),
finish-line task framing, optional self-collision contacts. pytest 7/7.

## 2026-06-15 -- fused on-GPU rollout megakernel + RNE-bias free win
User push: "everything needs to be on gpu and fused, fuck cublas." Built k_rollout (src/g1_env.cu):
ONE launch runs the whole H-step PPO rollout for all worlds -- per-world state stays register/
local-resident across all H control steps, MLP inference inlined (no cuBLAS, weights packed in a
torch buffer, read through L2), actions sampled via per-world curand, trajectory written straight
into torch-owned GPU buffers (O/A/LP/V/R/D/Vlast). Zero host round trips. torch only does the
gradient update, reading those device pointers in place (scripts/ppo_fused.py). Refactored the
physics+reward+done+autoreset core into env_advance() shared by k_step and k_rollout.

HONEST RESULT: fusion alone = 1.09x. Measured (N=8192,H=32,S=10): host round-trip rollout 1080ms,
fused 991ms. The 32 PCIe syncs/iter were only ~8% of the rollout -- the earlier "67% env phase =
round trips" theory was WRONG. The physics KERNEL is the wall (~92% of rollout), and it is local-
memory-bandwidth bound: ~32KB/thread spill (contact J/MinvJT/A + double ABA factor scratch).
Raising occupancy (launch_bounds 1->2->3 blocks/SM) changed nothing -> not latency/occupancy bound.
Standalone sim_contact runs ~10x faster/substep mostly because it has NO launch_bounds (runs ~4
blocks/SM) and does ONE factorization vs env_qacc's two.

RNE-bias free win: env_qacc recovered -bias via aba_qacc (factorize+solve, M^-1(-bias)) then
aba_mulM (M*qacc back) -- two O(N) passes to recover what RNE(q,qv,qacc=0) gives in one. Extracted
the validated RNE-bias recursion from compute_M_bias into src/aba_bias.cuh (compute_bias, fills the
same S basis the damped factor consumes, no dense-M CRBA). src/test_bias.cu: compute_bias ==
compute_M_bias bias bit-for-bit (max relerr 0.0 over 4096 random states) -> inherits the M0 MuJoCo
validation. Result: k_rollout stack frame 32.9KB->25.1KB (-24%), kernel 991ms->902ms (~10%),
2.65e6->2.91e6 substeps/s. Training reward curve unchanged (iter0/25 -3.61/-3.24). pytest 7/7.
Takeaway: fusion+RNE are correct and free, but the order-of-magnitude lives in the REMAINING
factorization + contact solve (per-row MinvJT via solveM, build A, PGS), not orchestration.

## 2026-06-15 -- contact-solve hot-loop optimization (delegated, gated, verified)
Attacked the real bottleneck (local-mem-bandwidth-bound contact solve in env_qacc). env_qacc
pure-moved into src/env_qacc.cuh so it's unit-testable; src/test_envqacc.cu freezes a reference
qacc over 4096 random states (build/envqacc_ref.bin) as the numerical-equivalence gate. Three
levers, each kept only if relerr<=1e-5 AND faster AND pytest 7/7 AND test_bias PASS:
 (1) J-row sparsity: each foot contact's J row is nonzero only on its ~12 ancestor dofs (of 35);
     restrict the O(nefc^2*NV) A-build, b-build, contact_qfrc to a per-contact dof-index list. -6.1%.
 (2) symmetric A-build: A=J M^-1 J^T+diag(R) is symmetric -> compute upper triangle only, mirror
     the lower (A kept dense so PGS access stays contiguous; triangular packing rejected, slows PGS). -5.1%.
 (3) pyramid-basis MinvJT: the 4 pyramidal rows/contact span 3-D {n,mu*t1,mu*t2} -> 3 ABA back-solves
     per contact not 4 (25% fewer solveM), reconstruct the 4 rows by exact linear combo. -6.4%.
contact.cuh J/R/aref/PGS math byte-for-byte untouched (all changes are loop bounds/storage/basis
in env_qacc.cuh), so the sim_contact oracle validation still holds. VERIFIED independently: fused
rollout 902->758 ms (16.5%), 2.91e6->3.46e6 substeps/s; env_qacc vs frozen ref relerr 5.07e-6;
pytest 7/7; test_bias relerr 0; reward-curve shape preserved (iter0/25 -3.69/-3.27). Stack frame
25072->26160 B (GREW slightly -- reconfirms: hot-loop bytes/compute matter, not cold-scratch size).
CUMULATIVE this session: host-loop 1080ms -> fused+RNE+contact 758ms = 1.42x rollout, no host trips,
no cuBLAS. Next lever (not done): build A directly in the 3-basis space per contact (folds the
4-row pyramid analytically), shrinking A-build + MinvJT further; needs PGS f->qfrc re-derivation
+ oracle re-validation. New: src/env_qacc.cuh, src/test_envqacc.cu. Edited: src/g1_env.cu (include).

## 2026-06-15 -- 3-basis contact A reformulation (delegated, gated, verified)
The 4 pyramidal friction rows per foot contact span a 3-D basis {n, mu*t1, mu*t2}. Rewrote the
contact solve in env_qacc.cuh to work entirely in that basis instead of reconstructing 4 dense
M^-1 J^T rows: per contact, 3 ABA back-solves m_a=M^-1 e_a (basis derived from the J rows
build_contact_jac emits, so contact.cuh stays frozen); per contact PAIR a 3x3 Gram G[a][b]=e_a.m_b
(9 sparse dots vs 16) expanded analytically to the 4x4 pyramid block (coeffs d0..3=(1,+-1,0)/(1,0,+-1));
b from 3 basis projections; PGS unchanged on the full 4*ncon system; final correction
dq=sum_c sum_a fb_c[a]*m_{c,a} (linear combo of basis solves) -- REMOVES the contact_qfrc J^T f
build AND the final aba_solveM. VERIFIED: fused 754->687 ms (-8.9%), 3.48e6->3.82e6 substeps/s;
env_qacc vs frozen ref relerr 5.07e-6->4.09e-6 (improved); pytest 7/7; test_bias 0; reward shape
preserved (iter0/25 -3.70/-3.28). contact.cuh + frozen headers untouched -> sim_contact oracle
still holds. Only src/env_qacc.cuh changed. CUMULATIVE rollout: host-loop 1080ms -> 687ms = 1.57x.
Added src/bench_env_phys.py + extern-C g1_env_bench_phys (additive physics-only throughput probe):
env physics in the continuous-foot-contact regime (the realistic TRAINED-walker steady state, feet
always planted) = 2.42e6 substeps/s N=8192, 2.78e6 N=16384. The fused-rollout 3.82e6 is higher only
because an untrained FALLING policy skips contact solves (feet off ground, ncon=0 early-exit); a
converged walker stays in contact so steady-state approaches the ~2.5-2.8e6 contact-heavy rate.

## 2026-06-15 -- finish-line task (M5 complete): trained + demo mp4
Reframed locomotion as a goal task. Finish-line reward in env_advance (g1_env.cu): kept the
proven anti-dive walking reward (alive + uprightness-gated forward progress + stability terms,
fall=-5) and added X_FIN=5.0 m -> crossing x>=X_FIN upright is a WIN (+50 terminal bonus,
episode success). The dense forward term is the "rewarded by how close it gets" signal; the
bonus rewards crossing. Dynamics (env_qacc.cuh) untouched, so all numerical gates still hold.
NOTE: earlier short fused smoke runs had overwritten the good walker's best.pt (saved under the
"fused" tag) -- added --warm and --tag=NAME to ppo_fused.py so task runs save to NAME_best.pt and
never clobber best.pt. Trained the finish task FRESH (best.pt was gone), tag=finish, 1500 iters
N=8192 on the fused on-GPU trainer (~25 min, sps 3.5e5). Learning curve mirrored run-2; mature
gait by ~iter 1400 (eval fwd 2.14m, ep_len 150 full, fall 0.00, vx 0.71); finish_best.pt = best
eval 2.48m (late-PPO wobble at iters 1450-1499 ignored via best-ckpt selection).
PLAYBACK (scripts/play_finish.py, 256 worlds, deterministic, <=480 steps): 256/256 reached the
line (100%); rendered the cleanest winner -- world walks 0.00 -> 4.98 m, crosses x=5 m in 5.5 s.
Added extern-C g1_env_qpos_all (copy all worlds' qpos) to pick a clean winner. Render
(scripts/render_finish.py) draws a green finish gate at X_FIN; bench/finish.mp4 (960x540 50fps),
QuickTime copy bench/finish_qt.mp4 (h264 yuv420p). M5 = full pipeline closed: fast validated sim
-> fused on-GPU PPO -> walk-to-finish-line, demoed. New: scripts/play_finish.py, render_finish.py.

## 2026-06-15 -- contact-solve bottleneck: profiled + 3-lever fan-out (levers 1+3 KEPT)
Profiled the contact solve (ncu as root on gamer + SKIP_CONTACT ablation). Verdict: the foot-
contact solve is ~89% of per-substep time (smooth-only 2.13e7 vs full 2.41e6 substeps/s, N=8192
continuous-contact). LATENCY-bound, NOT bandwidth/compute: ncu DRAM 36%, SM 4.7%, achieved
occupancy 4.1% (register-capped at 255 regs -> 4 blocks/SM), 80% of stalls = long-scoreboard on
local-memory (spill) loads. Within the contact solve, the ~24 aba_solveM M^-1 applies (3 basis x
up to 8 contacts) = 77% of step time (ablation). Fanned out 3 subagents in isolated repo copies:
 L1 (shrink A / working set): eliminating the dense J[G1_MAX_EFC*G1_NV]=1120-float array -- build
    the 3 basis rows directly from point_jac_col (e0=jc.z,e1=mu*jc.y,e2=-mu*jc.x) and inline R/aref,
    so build_contact_jac/build_R_aref/J all vanish. -4128 B frame, +2-4% alone. (On-the-fly A
    reconstruction in PGS was tried and REVERTED: -2x, recompute costs more than the spill it saves.)
 L2 (foot-contact independence): NEGATIVE, nothing kept. Measured off/diag A-block coupling: same-
    foot pairs ratio 0.89, cross-foot 0.04; but dropping even the 4% cross-foot coupling = relerr
    0.53 (fails gate 5 orders). PGS@8 iters is non-converged so any update-order change diverges.
 L3 (multi-RHS solveM): new src/aba_solvem_multi.cuh -- aba_solveM_multi<K> applies M^-1 to all 3
    basis RHS per contact in ONE factor traversal (amortizes the local-mem factor reads). Bit-equiv
    to 3 sequential solves (relerr 1.44e-7, src/test_solvem_multi.cu). +24% alone (N=8192).
COMPOSED L1+L3 (disjoint regions of env_qacc.cuh): L1's J-elim offsets L3's frame growth. Stack
25712(base, was 30208 L3-alone) -> 26000 B. VERIFIED clean (uncontended GPU, reproduced 3x):
bench_env_phys N=8192 2.41e6->3.00e6 (+24.5%), N=16384 2.78e6->3.32e6 (+19.4%); fused rollout
687ms/3.82e6 -> 601ms/4.34e6 (+13.6%). Gates: test_envqacc relerr 5.53e-6, test_bias 0, pytest 7/7,
multi-RHS equiv 1.44e-7. New: src/aba_solvem_multi.cuh, src/test_solvem_multi.cu. Edited env_qacc.cuh.
Remaining: solveM applies still ~70% of step; cross-contact batch (K=3*ncon) would spill harder.
Occupancy still pinned by 255-reg cap -- the next real lever is cutting registers, not arrays.

## 2026-06-15 -- reg-cap occupancy chase: NEGATIVE (kernel already at occupancy ceiling)
Tried trading register pressure for occupancy via __launch_bounds__ minBlocks sweep (unified the
knob ROLLOUT_MINBLK across k_step/k_rollout/k_phys_bench). Result: DEAD END. ptxas already uses 255
regs = exactly 4 blocks/SM (65536/(255*64)=4.0) = the 16.7% theoretical occupancy ceiling, so
MINBLK 1-4 are IDENTICAL (255 regs, 3.01e6 substeps/s N=8192). Forcing MINBLK=6 cuts regs to 168
but DOUBLES spill (1396->3060 B stores) and REGRESSES to 2.68e6 (-11%) -- spill-bound, the extra
occupancy can't pay for the extra local traffic. ncu's achieved 4.1% vs theoretical 16.7% gap is
WORKLOAD IMBALANCE (divergent contact counts across the 32 worlds in a warp), NOT block count, so
more theoretical occupancy wouldn't help anyway. The 1-thread/world kernel is at a genuine local
optimum. Real remaining levers are bigger and different: (a) contact-count binning/sorting (SPEC's
"hard bucket by contact count") to de-divergence warps -- helps the MIXED/falling regime, less the
uniform standing-walker regime; (b) cooperative threads/world to shrink per-thread state -- but the
smooth-kernel coop already LOST (half throughput, narrow-tree lane idling). Kept ROLLOUT_MINBLK=1
(default, best). No code change to the solve. Stopping micro-opt here.

## 2026-06-15 -- corrected diagnosis + K=6 contact-pair batched solve (KEPT, with a tradeoff)
Device-queried the 3090 (cudaGetDeviceProperties) + CUDA Occupancy API on the ACTUAL kernels:
255 regs -> maxActiveBlocks/SM=4 = 16.7% theoretical (authoritative, not hand-math). Reg-cap chase
is a confirmed dead end (forcing 6 blocks cuts regs to 168, doubles spill, -11%). CORRECTED the
earlier wrong diagnosis: ncu on the REAL k_phys_bench in the continuous-contact regime shows DRAM
67% (not the 36% I'd cited from test_envqacc's mixed regime), achieved occ 6.5%, 88% local-scoreboard
stalls; local-mem (spill) traffic ~5e9 sectors DRIVES the 67% DRAM. So it's spill-BANDWIDTH-bound,
not bandwidth-idle. That reopened batching the contact M^-1 solves: full batch (K=3*ncon=24) is dead
(solve scratch O(NB*6*K) = 35KB blows the frame), but K=6 (2 contacts/traversal) halves factor reads
(9->5 traversals for 8 contacts) at +4.3KB scratch. Restructured env_qacc into PASS1 (build Eb/bb/R/
aref) + PASS2 (K=6 paired solve, Mm laid contiguous so one solve fills both contacts, odd contact
zero-pads cols 3-5). Numerically identical (gate relerr unchanged <=1e-5). MEASURED (reproduced 2x):
fused rollout 601->528 ms / 4.34e6->4.96e6 (+13.8%, the TRAINING operating point N=8192); physics
N=8192 3.00->3.19e6 (+6.4%); physics N=16384 3.32->3.22e6 (-3.0% REGRESSION -- the 31.2KB frame
(was 26KB) hurts at larger N where occupancy-demand is higher). Kept because training runs at N=8192
where rollout is +14%; the N=16384 loss is outside the trained regime. pytest 7/7, bias 0.
CUMULATIVE rollout this session: host-loop 1080ms -> 528ms = 2.05x (megakernel + RNE + contact
sparsity/symmetry/3-basis + multi-RHS K=3 + J-elim + K=6 batch). Added diagnostics: src/deviceq.cu,
extern-C g1_occupancy_report. Frame growth (31KB) is the new ceiling concern for the persistent kernel.

## 2026-06-18 -- Blackwell (sm_120, RTX PRO 6000) physics-throughput campaign (+24%, frame-bound)
Goal: push the sim on the PRO 6000 at the nanoG1-matched config (peak physics-steps/s over N=4096..32768).
All measured on anvil, gated vs the MuJoCo oracle (test_envqacc, relerr <= 1e-5).

DIAGNOSIS (ncu on the real k_phys_bench): LOCAL-MEMORY-BANDWIDTH bound, NOT compute. At the N=16384
baseline peak: DRAM 44% SoL, L2 29%, L1 17%, compute 6%, warps_active 5.5%. Local traffic dominates
global ~1000:1 (439M vs 410K sectors). The ~28KB/thread local frame is the kernel. Signature: N=32768
is SLOWER than N=16384 (L2/capacity cliff -- adding worlds thrashes the cache before saturating
bandwidth). KEY MODEL: frame cuts convert ~1:1 (often super-linearly, via the cliff) to throughput and
shift the peak to higher N; occupancy is N-limited at the peak (~1.4 blocks/SM), so the per-thread
FOOTPRINT, not register occupancy, is the lever.

WHAT WORKED (all gate-passing, KEPT, cumulative +24% over the 06-13 K=3 baseline = 1.231e7 -> ~1.53e7
substeps/s stable @ N=20480, ~1.57x nanoG1's 9.75M on the same GPU):
- msolvescr K=3 -> K=1 (one basis column per factor traversal): +12%. Smaller PASS2 scratch (4464->1488B).
  3x factor re-reads are free (compute idle). relerr 5.5e-6.
- CDS 16 -> 12 (true max ancestor-dof count; foot spheres only on bodies 7/13, chain = 6 leg + 6 base
  = exactly 12): +12% more, peak shifted 16384 -> 20480. Shrinks cdof + Eb.
- cdof/cnd int -> uint8 (dof indices <=34, counts <=12): +2%, bit-identical. Same access pattern.

WHAT FAILED (TRIED, MEASURED, REVERTED -- the negative results matter):
- G-block A reconstruction (store 3x3 Gram, rebuild dense A on the fly in PGS): NEUTRAL (+1%, noise).
  A's 4KB array shrank but the heavier PGS inner loop spilled ~5KB MORE (7824->13276 spill stores),
  eating the saving. LESSON: cuts that add hot-loop compute get refunded as spills. (A is NOT the
  binding frame array.)
- Mm elimination (compute M^-1 basis on the fly per contact via symmetry G=Mm_c.e_cp; dq via one
  M^-1 solve of J^T f): frame dropped the MOST (-2480B to 23424B) but throughput -21%. The merged
  loop keeps Mm_c+msolvescr+A churning together and wrecks access locality. LESSON: a clean
  separate-pass structure (PASS2 writes Mm, PASS3 reads it) beats a smaller-frame fused loop;
  locality, not just footprint, decides it.
- fp16 storage of Mm (the M^-1 contact basis): FAILS the gate at relerr 6.3e-3 (budget 1e-5). The
  contact correction dq = M^-1 J^T f is far too precision-sensitive for fp16 (unlike the smooth ABA
  path, where fp16 storage was a win). Mm stays fp32.
- L1 carveout = max-L1: no-op (driver already gives max L1 at 0 shared mem).
- -maxrregcount / occupancy: no help (N limits occupancy at the peak; more resident frames thrash L2).
- BLOCK size 32/64/128: 64 (default) is best; register-bound so all give 256 threads/SM anyway.

STRUCTURAL LEVERS RULED OUT for this per-thread-local-frame-bound SIMT kernel (independent analysis
+ this codebase's prior measurements): warp specialization (no cross-thread data / overlapping roles;
warp-coop already -48..-73%), TMA (no global tensors to stage; traffic is thread-private local),
distributed-shared-memory / clusters (uses ~0 SMEM, no cross-world sharing), tensor cores (per-thread
6x6/3x3 solves, not collective MMA; the NN is 2% of step), cp.async (overlaps global->shared, not
local spill reloads). The megakernel/fusion thesis was already retired (NN ~2% of step).

CEILING: re-profiled at the optimized N=20480 -- now L2 49.7% SoL (binding), DRAM 45.8%, compute 11%,
warps_active 7%. Frame cuts moved us DRAM-bound -> L2-bound. We're latency/concurrency-limited (only
7% warps active) but can't add worlds without the L2 cliff; smaller frame is the only axis and the
safe footprint cuts are exhausted (structural cuts hurt locality, precision cuts fail the gate). The
last Blackwell lever -- L2 persistence (cudaAccessPolicyWindow + cudaLimitPersistingL2CacheSize) over a
world-indexed SoA GLOBAL scratch for Mm (toggle -DMM_GLOBAL) -- was IMPLEMENTED and MEASURED: -49%
(1.49e7 -> 7.48e6), a decisive LOSS, even though it cut the local frame to 22544B. Moving the per-thread
M^-1 basis to global scratch makes every Mm access a global transaction; even L2-persistence-pinned,
that's far slower than local memory's hardware-interleaved L1/L2 caching. DEFINITIVE: explicit L2
residency does NOT beat implicit local caching here. Plumbing reverted (this log keeps the result).
CONCLUSION: kernel is at its numerically-gated ceiling; further gains need relaxing the 1e-5 gate (lower
physics fidelity) or a different algorithm/mapping (warp-coop already proven worse). Net campaign: +24%.

Also CHECKED (adversarial 2nd-opinion proposed shrinking A by capping MAX_CONTACT, claiming "typical
ncon=2-4, A mostly cold"): MEASURED the ncon histogram (NCON_PROBE atomic, instrumentation reverted) over
the stand-pose bench -- ncon is 6/7/8 in 93% of contact-solve calls (8: 28%, 7: 38%, 6: 27%; max 8). The
standing/walking robot plants nearly all 8 foot spheres, so nefc is routinely 24-32 and A[32x32]=4KB (not
16KB; the suggestion mis-sized it 4x) is genuinely near-full. MAX_CONTACT=8 is a HARD physical bound (8
foot spheres), unlike CDS which was a conservative kinematic bound -- capping it would corrupt/drop
contacts in the majority of states. A cannot shrink. Confirms the floor empirically.
