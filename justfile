# g1.cu — gamer-primary, sm_86 Ampere (RTX 3090). Run on gamer.
nvcc := "/opt/cuda/bin/nvcc"
arch := "sm_86"

default:
    @just --list

build:
    @echo "TODO: {{nvcc}} -arch={{arch}} -O3 src/*.cu -o build/g1"

bench:
    @echo "TODO: hyperfine ./build/g1 ; ncu/nsys for kernel detail"

test:
    @echo "TODO: uv run pytest tests/  (numerics vs reference)"

sanitize:
    @echo "TODO: compute-sanitizer ./build/g1"
