"""ABA acceptance test: O(N) articulated-body dynamics matches MuJoCo (fp32) and
the batched multi-step kernel stays deterministic + oracle-matched.
"""
import os, subprocess, numpy as np, pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NVCC = "/opt/cuda/bin/nvcc"
NSTEPS = 300


def sh(cmd): return subprocess.run(cmd, cwd=HERE, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def built():
    sh(["uv", "run", "python", "scripts/export_model.py"])
    sh(["uv", "run", "python", "scripts/gen_reference.py"])
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    sh([NVCC, "-arch=sm_86", "-O3", "--extended-lambda", "src/aba_test.cu", "-o", "build/aba_test"])
    sh([NVCC, "-arch=sm_86", "-O3", "--extended-lambda", "src/sim_aba.cu", "-o", "build/sim_aba"])
    sh(["./build/aba_test", str(NSTEPS)])
    return np.load(os.path.join(HERE, "bench", "ref_traj.npz"))


def test_aba_qacc(built):
    ref = built
    qacc = np.fromfile(os.path.join(HERE, "bench", "aba_qacc.bin"), dtype=np.float32)
    assert np.linalg.norm(qacc - ref["qacc"][0]) / np.linalg.norm(ref["qacc"][0]) < 1e-4


def test_aba_trajectory(built):
    ref = built
    nq = ref["qpos"].shape[1]
    traj = np.fromfile(os.path.join(HERE, "bench", "aba_traj.bin"), dtype=np.float32).reshape(-1, nq)
    rq = ref["qpos"]
    assert np.abs(traj[:, :3] - rq[:, :3]).max() < 1e-4          # base pos
    assert np.abs(traj[:, 7:] - rq[:, 7:]).max() < 1e-4          # joints
    dq = np.minimum(np.abs(traj[:, 3:7] - rq[:, 3:7]).sum(1),
                    np.abs(traj[:, 3:7] + rq[:, 3:7]).sum(1)).max()
    assert dq < 1e-4                                             # base quat


def test_aba_batched_matches_and_deterministic(built):
    ref = built
    # multi-step batched run; world 0 vs MuJoCo + all-worlds determinism
    out = subprocess.run([os.path.join(HERE, "build", "sim_aba"), str(NSTEPS), "4096", "16"],
                         cwd=HERE, check=True, capture_output=True, text=True).stdout
    assert "determinism=0.0e+00" in out
    nq = ref["qpos"].shape[1]
    fin = np.fromfile(os.path.join(HERE, "bench", "sim_aba_final.bin"), dtype=np.float32)
    rq = ref["qpos"][NSTEPS]
    assert np.abs(fin[:3] - rq[:3]).max() < 1e-4
    assert np.abs(fin[7:] - rq[7:]).max() < 1e-4
