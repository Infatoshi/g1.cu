# SPEC.md — g1.cu

North-star architecture. Journey/fixes in DEVLOG.md; CLAUDE.md holds commands + constraints.

## Goal
Train a humanoid (Unitree G1 reference body) to walk to a finish line by fusing physics
simulation AND policy inference into a SINGLE persistent, warp-specialized CUDA kernel,
eliminating per-step kernel-launch / CPU-GPU sync overhead and the bulk-synchronous
physics-barrier-GEMM-barrier swing that idles either the scalar units or the tensor cores.

Target: NVIDIA RTX 3090 (sm_86, Ampere), CUDA 13.3, on gamer. 24GB VRAM.

## The central thesis
Per-world humanoid state is tiny (~60-100 floats: q ~30, qd ~30). Physics is latency-bound,
branchy, low-arithmetic-intensity scalar work that wants fine-grained per-world SIMT
(lane = world, 32 worlds/warp). The policy is a small MLP that only reaches tensor-core
efficiency when many worlds' observations are batched into a fat GEMM. Opposite granularities.

A shared-memory ring buffer decouples the two clocks:
- Physics warps produce 32-world obs tiles asynchronously.
- NN warps aggregate several tiles into an MMA-shaped GEMM tile, issue mma.sync, write
  actions back through a return ring.
This is the whole bet. If the ring cannot keep the tensor-core consumer fed without stalling
the divergent physics producers, the design loses to bulk-synchronous.

## Kernel structure (intended)
- Persistent grid: ~#SMs CTAs (3090 = 82 SMs), never exit, loop pulling timesteps.
- Per CTA: physics warps + NN warps (software role split; NOT warpgroup-aligned MMA on Ampere).
- STATIC register partitioning (no setmaxnreg on sm_86): physics ~64-96 regs/thread.
- mbarrier + cp.async producer/consumer handshake (NOT lock-step bar.sync) — both exist on sm_80+.
- Per-world solver divergence: hard contact bucketing + per-lane mask + __ballot_sync, capped iters.
- Numerics: fp32 physics (ill-conditioned contact solve), bf16/tf32 policy GEMM.

## Ampere (sm_86) hardware notes — see CLAUDE.md for the hard rules
No setmaxnreg, no wgmma, no tcgen05. Tensor cores via warp-level mma.sync (m16n8k16 bf16/fp16,
m16n8k8 tf32). cp.async + mbarrier available. 3090 bulk GEMM is less likely to be the
bottleneck than big Blackwell, so the fusion win bar is, if anything, harder here.

## Milestones (each gates the next — stop if a gate fails)
- M0: single-world humanoid forward dynamics fp32, validated vs MuJoCo/MJX reference trajectory.
- M1: bulk-synchronous batched baseline (physics kernel | GEMM kernel) — the number to beat.
- M1.5: persistent physics + global obs staging -> CUTLASS Ampere policy -> actions back
  (codex says try THIS before true fusion).
- M2: persistent physics-only kernel, fake action; measure worlds/sec, divergence histogram, occupancy.
- M3: fused + DUMMY NN (delay + writeback) — isolate ring overhead/backpressure.
- M4: one real NN layer (96x128 bf16) vs bulk path; win or within ~10% w/ lower latency, else STOP.
- M5: full MLP + contact bucketing, then PPO + walk-to-finish-line reward.

## Open questions
- Does ring-buffer backpressure keep tensor cores fed under divergent physics? (the dealbreaker)
- Optimal physics:NN warp ratio per CTA — tune empirically.
- Does contact bucketing salvage solver divergence, or does it need the 1-warp=1-world remap?
- On a 3090, is the policy GEMM ever a big enough fraction of step time for fusion to win?
