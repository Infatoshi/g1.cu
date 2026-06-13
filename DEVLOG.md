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

### Next
M0: single-world humanoid forward dynamics in fp32, validate vs a MuJoCo/MJX reference
trajectory. Pick the G1 MJCF + reference engine. Nothing built yet.
