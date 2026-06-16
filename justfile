# g1.cu — gamer-primary, sm_86 Ampere (RTX 3090). Run on gamer.
nvcc := "/opt/cuda/bin/nvcc"
arch := "sm_86"

default:
    @just --list

# regenerate the MuJoCo reference trajectory + model header
ref:
    uv run python scripts/export_model.py
    uv run python scripts/gen_reference.py

build:
    @mkdir -p build
    {{nvcc}} -arch={{arch}} -O3 src/dynamics.cu -o build/g1

# run the M0 single-world rollout (default 300 steps)
run nsteps="300": build
    ./build/g1 {{nsteps}}

bench: build
    hyperfine --warmup 3 './build/g1 300'

# M1 bulk-synchronous batched baseline (the number to beat)
sim:
    @mkdir -p build
    {{nvcc}} -arch={{arch}} -O3 -Xptxas -v src/sim.cu -o build/sim

# env-steps/s throughput sweep over world counts
sweep nsteps="300": sim
    ./build/sim {{nsteps}}

# O(N) articulated-body kernel (3x the dense baseline), batched + multi-step
aba:
    @mkdir -p build
    {{nvcc}} -arch={{arch}} -O3 --extended-lambda -Xptxas -v src/sim_aba.cu -o build/sim_aba

# ABA throughput sweep (world counts x on-chip multi-step)
aba-sweep nsteps="300": aba
    ./build/sim_aba {{nsteps}}

# fp64 numpy ABA oracle (matches MuJoCo to ~1e-15)
aba-oracle:
    uv run python scripts/aba_np.py 1 0 0 0 all A

# numerics: CUDA fp32 vs MuJoCo reference
test:
    uv run pytest tests/ -v

# fp64 numpy oracle: my Featherstone vs MuJoCo intermediates (cdof/cvel/M/bias)
oracle:
    uv run python scripts/ref_dynamics_np.py com

sanitize: build
    compute-sanitizer ./build/g1 50

# cost-split: physics-step vs cuBLAS policy-MLP forward (the bulk-synchronous baseline)
bench-nn:
    @mkdir -p build
    {{nvcc}} -arch={{arch}} -O3 src/bench_nn.cu -lcublas -o build/bench_nn
    @for N in 4096 16384 65536; do for P in small medium large; do ./build/bench_nn $N $P 300; done; done

# build the batched RL env shared lib (ctypes target for scripts/ppo.py)
env:
    @mkdir -p build
    {{nvcc}} -arch={{arch}} -O3 -Xcompiler -fPIC --shared src/g1_env.cu -lcurand -o build/libg1env.so

# minimal PPO walk-to-finish-line training on the GPU env
ppo iters="100" nworlds="4096": env
    uv run python scripts/ppo.py {{iters}} {{nworlds}}

# Stage A: implicit (implicitfast) PD integrator -- accuracy vs MuJoCo (single-step ~1e-5)
test-implicit:
    @mkdir -p build
    uv run python scripts/gen_implicit_ref.py
    {{nvcc}} -arch={{arch}} -O3 src/test_implicit.cu -lcurand -o build/test_implicit
    uv run python scripts/test_implicit.py

# Stage A: stability gate -- full kp=500, S=10, hold stand pose (must stay bounded/upright)
test-stable: env
    uv run python scripts/test_stability.py
