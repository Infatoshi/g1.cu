# g1.cu

A hyperspecialized, fully-fused GPU simulator and on-GPU reinforcement-learning loop for the
Unitree G1 humanoid, running on a single consumer GPU (RTX 3090, sm_86, CUDA 13.x).

The entire RL training rollout — rigid-body physics, foot-ground contact solve, policy-network
inference, action sampling, and reward — runs in **one persistent CUDA kernel**. No per-step
CPU↔GPU round-trips, no cuBLAS. PPO gradient updates run in PyTorch, reading the trajectory
buffers in place. It trains a G1 to walk across a 5 m finish line in **~3–4 minutes**.

## Honest framing (read this first)

This is a research POC, not a breakthrough. Training a legged robot to walk in minutes on one
GPU is the established result behind massively-parallel RL — see Isaac Gym and *"Learning to
Walk in Minutes Using Massively Parallel Deep RL"* (Rudin et al., 2021). General GPU RL-physics
engines (Isaac Lab, MuJoCo MJX, Brax, NVIDIA Warp, Genesis) are mature and, unlike this one,
reusable across arbitrary robots.

What this repo explores is the opposite axis — **hyperspecialization**: a hand-written sim for
ONE robot and ONE contact regime (foot-ground), with the physics *and* the policy fused into a
single megakernel. The closest prior art for hyperspecialized batched GPU simulators is Madrona
(Stanford, 2023), which fuses the *simulation* into a megakernel; here the *policy inference* is
fused in as well. Measured honestly, that fusion contributes little on its own (~1.09×); the
real speed comes from a specialized, spill-minimized contact solve. The value here is the
MuJoCo-validated specialized kernel and the measured, no-hype engineering — not a new capability.

## Results (RTX 3090, measured)

| metric | value |
|---|---|
| Physics throughput, foot-ground contact (N=8192) | ~3.2e6 substeps/s |
| vs MJX full-physics, same GPU (foot-ground)\* | ~10× |
| Fused training rollout (N=8192, H=32, 10 substeps) | 528 ms (4.96e6 substeps/s) |
| Rollout speedup vs host-orchestrated loop | 2.05× |
| Train G1 to cross a 5 m finish line | ~250 PPO iters, ~3–4 min |

\* Apples-to-oranges caveat: this solves only foot-ground contacts (8 foot spheres vs a floor
plane); MJX simulates the full self-collision set. The ~10× is honest only for the
locomotion-relevant contact set, not identical physics.

## Validation

The physics is checked bit-for-bit against MuJoCo:

- forward dynamics (Featherstone ABA) vs MuJoCo `qacc`: relerr ~2e-5
- bias force (RNE, qacc=0) vs MuJoCo `qfrc_bias`: bit-identical
- foot-ground contact solve vs a MuJoCo-matched fp64 PGS oracle: relerr ~2e-5
- every kernel optimization is gated against a frozen reference at relerr ≤ 1e-5

```
just test          # pytest: CUDA fp32 vs MuJoCo reference trajectory (7 tests)
```

## Build & run

Requires CUDA (tested 13.x, sm_86) and [uv](https://github.com/astral-sh/uv). `nvcc` is invoked
with an explicit `-arch=sm_86` and no fat binaries.

```
just build                                       # validated single-world dynamics
just env                                         # batched RL env shared lib (build/libg1env.so)
uv run python scripts/ppo_fused.py 1500 8192     # fully-on-GPU PPO training
uv run python scripts/play_finish.py             # roll the trained policy across the line
uv run python scripts/render_finish.py           # render the demo to bench/finish.mp4
```

A pretrained finish-line walker is included at `models/ppo_ckpt/finish_best.pt`, and a demo
render at `bench/finish_qt.mp4`.

## Layout

- `SPEC.md` — architecture and the original research thesis
- `DEVLOG.md` — the full journey: every optimization, dead end, and measured number
- `src/` — the CUDA core:
  - `aba.cuh` — Featherstone articulated-body forward dynamics (validated)
  - `aba_factor.cuh` — factor-once / O(nbody) `M⁻¹` apply, `aba_solvem_multi.cuh` — multi-RHS
  - `aba_bias.cuh` — RNE bias force; `contact.cuh` — foot-ground pyramidal contact
  - `env_qacc.cuh` — the fused per-substep generalized-acceleration solve
  - `g1_env.cu` — the megakernel (`k_rollout`) and ctypes RL environment
- `scripts/ppo_fused.py` — the on-GPU PPO trainer

## Hardware notes

Targets sm_86 (Ampere, RTX 3090). The kernel is local-memory-bandwidth bound at a 255-register /
16.7%-occupancy ceiling — a genuine local optimum for the one-thread-per-world mapping (full ncu
analysis and the dead ends are in `DEVLOG.md`).

## License & attribution

The code in this repository (`src/`, `scripts/`, `tests/`, docs) is **MIT** — see
[LICENSE](LICENSE).

The Unitree G1 robot model under `models/` (MJCF + meshes) is **not mine**: it is third-party,
redistributed under the **BSD-3-Clause** license from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) /
[MuJoCo Playground](https://playground.mujoco.org/), © Unitree Robotics. The original notice and
attribution are retained in [`models/LICENSE`](models/LICENSE) and
[`models/README.md`](models/README.md). If you use the model, cite MuJoCo Playground
(Zakka et al., 2025).
