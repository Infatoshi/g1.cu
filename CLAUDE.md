# CLAUDE.md — g1.cu

Home: gamer-primary — canonical at `gamer:/home/infatoshi/dev/cuda/g1.cu`. Run all
dev/compute there via `ssh gamer`. No Mac/anvil copy. CUDA project for the RTX 3090
(sm_86, Ampere), CUDA 13.3. Edit on gamer directly.

Named for the Unitree G1 humanoid (the intended reference body).

## What this is
Research POC: fuse humanoid physics sim + policy inference into ONE persistent
warp-specialized CUDA kernel, to beat the bulk-synchronous (physics-kernel | GEMM-kernel)
baseline. Full architecture + thesis in SPEC.md. Journey/decisions in DEVLOG.md.

## Build / run / test
- `nvcc` is NOT on fish PATH. Use `/opt/cuda/bin/nvcc` or `bash -lc`. Compile with EXPLICIT
  arch, no fat binaries: `/opt/cuda/bin/nvcc -arch=sm_86 ...`
- Profile: `ncu` (kernel), `nsys` (timeline). Illegal-access: `compute-sanitizer`.
- Benchmarks: `hyperfine` for CLI timing, never ad-hoc loops. Live GPU: `nvtop`.
- (justfile holds real recipes once src exists: `just build`, `just bench`, `just test`.)
- Python glue (PPO, plotting, reference): UV only — `uv run ...`, never bare python/pip.
- Before ANY GPU run: `nvidia-smi`. gamer is the fleet's only free GPU — but RL/sim work
  here may be shared; don't assume exclusive. 24GB VRAM (much tighter than anvil's 96GB).

## Hard constraints (sm_86 Ampere — these differ from Blackwell; do not copy sm_120 advice)
- `setmaxnreg` does NOT exist on Ampere (sm_90+ only). There is NO dynamic register
  donation between warpgroups. STATIC register control is the only option — keep physics
  warps ~64-96 regs/thread and the NN path off the spill cliff. Spilling in a persistent
  kernel is catastrophic (every resident CTA holds resources forever).
- NO `wgmma` (Hopper sm_90a), NO `tcgen05.mma` (Blackwell sm_100). Tensor-core path is
  warp-level `mma.sync` — Ampere shapes: `m16n8k16` bf16/fp16, `m16n8k8` tf32. Stage
  operands to SMEM, issue per-warp `mma.sync`. CUTLASS Ampere GEMM is the reference.
- `cp.async` (sm_80+) and `mbarrier` (sm_80+) DO exist — use them for the ring-buffer
  async handshake and SMEM staging. So the persistent + ring design is viable; only the
  register-donation and warpgroup-MMA pieces are gone.
- Warpgroup-level role split is still the right structure (NN warps vs physics warps), but
  it is a software convention here, not enforced by warpgroup-aligned instructions.

## Design rules baked in from review (codex gpt-5.5, 2026-06-13; written pre-move for sm_120, the kernel-design parts still hold)
- TWO rings (obs: physics->NN, action: NN->physics). Slot = a full NN microbatch tile
  (e.g. 128/256 worlds), NOT a single 32-world warp tile. One aggregator warp packs
  4-8 physics tiles into a slot. mbarrier parity handshake, NOT global counters, NOT
  lock-step bar.sync. Physics must write obs in the NN's consumed layout (no gather/transpose
  tax) or the ring is a cost, not a decoupler.
- Contact solver: HARD bucket by contact count before solving (ballistic / 1-2 / 3-4 /
  pathological), fixed iter budget per bucket, `__ballot_sync` early-exit. Long-tail worlds
  go to a separate queue. If >5-10% hit the pathological queue, the 1-warp=32-worlds solver
  mapping is wrong (consider 1-warp=1-hard-world for the solve phase only).
- Policy cadence: if training uses N physics substeps per action, only fuse the action steps.

## Go / no-go (do not lose sight of this)
Fused persistent kernel must beat bulk-synchronous END-TO-END env-steps/sec by >=15% at
matched numerics with no pathological-queue explosion. A 3-5% win is NOT worth the
complexity. Honest prior (codex): this likely LOSES to "persistent physics + batched policy
microkernel" unless policy inference is 20-40%+ of step time. Build the cheap baseline FIRST
and only escalate if clearly launch/global-traffic bound. (On the 3090 the bulk GEMM is less
likely to be the bottleneck than on big Blackwell, so the bar is, if anything, harder here.)

## Build order (each gates the next — stop if a gate fails)
M0 single-world fwd dynamics fp32 validated vs MuJoCo/MJX reference trajectory (tolerance).
M1 bulk-synchronous batched baseline = the number to beat.
M1.5 persistent physics + global obs staging -> CUTLASS Ampere policy -> actions back
   (codex: try THIS before true fusion).
M2 persistent physics-only kernel, fake action; measure worlds/sec, solver divergence
   histogram, regs, occupancy.
M3 fused + DUMMY NN (delay + writeback) to isolate ring overhead/backpressure.
M4 one real NN layer (96x128 bf16, mma.sync m16n8k16) vs bulk path; must win or come within
   ~10% w/ lower latency, else STOP.
M5 full MLP + contact bucketing, then PPO + walk-to-finish-line reward.
