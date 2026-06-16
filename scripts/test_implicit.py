"""Compare CUDA implicit-PD trajectory vs MuJoCo implicitfast reference (Stage A accuracy).

Pipeline:
  1) scripts/gen_implicit_ref.py -> bench/implicit_ref.npz (MuJoCo implicitfast, contacts off)
  2) pack a .bin the CUDA harness reads, run build/test_implicit -> bench/implicit_cuda.bin
  3) compare qpos/qvel trajectories.
"""
import os
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(HERE, "bench", "implicit_ref.npz")
BIN_IN = os.path.join(HERE, "bench", "implicit_ref.bin")
BIN_OUT = os.path.join(HERE, "bench", "implicit_cuda.bin")
EXE = os.path.join(HERE, "build", "test_implicit")
NQ, NV, ACT = 36, 35, 29


def main():
    if not os.path.exists(NPZ):
        subprocess.run(["uv", "run", "python", os.path.join(HERE, "scripts", "gen_implicit_ref.py")], check=True)
    z = np.load(NPZ)
    dt = float(z["dt"]); nsteps = z["targets"].shape[0]
    # pack: int nsteps, f64 dt, f64 qpos0[NQ], qvel0[NV], Bdiag[NV], targets[nsteps*ACT]
    with open(BIN_IN, "wb") as f:
        np.array([nsteps], dtype=np.int32).tofile(f)
        np.array([dt], dtype=np.float64).tofile(f)
        z["qpos"][0].astype(np.float64).tofile(f)
        z["qvel"][0].astype(np.float64).tofile(f)
        z["Bdiag"].astype(np.float64).tofile(f)
        z["targets"].astype(np.float64).reshape(-1).tofile(f)

    subprocess.run([EXE, BIN_IN], check=True, cwd=HERE)

    with open(BIN_OUT, "rb") as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        qp = np.fromfile(f, dtype=np.float32, count=(n + 1) * NQ).reshape(n + 1, NQ)
        qv = np.fromfile(f, dtype=np.float32, count=(n + 1) * NV).reshape(n + 1, NV)

    ref_qp = z["qpos"]; ref_qv = z["qvel"]
    # per-step max abs error
    qp_err = np.abs(qp - ref_qp)
    qv_err = np.abs(qv - ref_qv)
    print(f"steps={nsteps} dt={dt}")
    print(f"  qpos max abs err: step1={qp_err[1].max():.3e}  step10={qp_err[min(10,nsteps)].max():.3e}  final={qp_err[-1].max():.3e}")
    print(f"  qvel max abs err: step1={qv_err[1].max():.3e}  step10={qv_err[min(10,nsteps)].max():.3e}  final={qv_err[-1].max():.3e}")
    # single-step error (cleanest: no accumulated fp32 drift) -> the integrator-fidelity number
    print(f"  SINGLE-STEP qvel max abs err: {qv_err[1].max():.3e}  qpos: {qp_err[1].max():.3e}")
    print(f"  final base z: cuda={qp[-1,2]:.4f} mujoco={ref_qp[-1,2]:.4f}")
    print(f"  final |qvel|: cuda={np.linalg.norm(qv[-1]):.4f} mujoco={np.linalg.norm(ref_qv[-1]):.4f}")
    # RL-grade tolerance on single step (fp32 vs fp64 oracle); trajectory drift reported separately
    single = qv_err[1].max()
    ok = single < 1e-3 and np.isfinite(qv).all()
    print("RESULT:", "PASS" if ok else "FAIL", f"(single-step qvel err {single:.3e} < 1e-3)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
