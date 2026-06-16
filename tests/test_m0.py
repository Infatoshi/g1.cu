"""M0 acceptance test: CUDA fp32 single-world forward dynamics vs MuJoCo.

End-to-end: (re)generate the MuJoCo reference, build the CUDA engine, roll out,
and assert FK/CRBA/RNE/qacc/trajectory match within fp32 tolerance.

Run: uv run pytest tests/ -v
"""
import os
import subprocess
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NVCC = "/opt/cuda/bin/nvcc"
NSTEPS = 300


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=HERE, check=True, capture_output=True, text=True, **kw)


@pytest.fixture(scope="module")
def outputs():
    sh(["uv", "run", "python", "scripts/export_model.py"])
    sh(["uv", "run", "python", "scripts/gen_reference.py"])
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    sh([NVCC, "-arch=sm_86", "-O3", "src/dynamics.cu", "-o", "build/g1"])
    sh(["./build/g1", str(NSTEPS)])
    ref = np.load(os.path.join(HERE, "bench", "ref_traj.npz"))
    nv, nq = ref["M0"].shape[0], ref["qpos"].shape[1]
    diag = np.fromfile(os.path.join(HERE, "bench", "cuda_fk.bin"), dtype=np.float32)
    M = diag[:nv*nv].reshape(nv, nv)
    bias = diag[nv*nv:nv*nv+nv]
    qacc = diag[nv*nv+nv:nv*nv+2*nv]
    traj = np.fromfile(os.path.join(HERE, "bench", "cuda_traj.bin"), dtype=np.float32).reshape(-1, nq)
    return ref, M, bias, qacc, traj


def relerr(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_mass_matrix(outputs):
    ref, M, *_ = outputs
    assert relerr(M, ref["M0"]) < 1e-5
    assert np.abs(M - M.T).max() == 0.0  # exactly symmetric by construction


def test_bias(outputs):
    ref, _, bias, *_ = outputs
    assert relerr(bias, ref["qfrc_bias"][0]) < 1e-5


def test_qacc(outputs):
    ref, _, _, qacc, _ = outputs
    assert relerr(qacc, ref["qacc"][0]) < 1e-4


def test_trajectory(outputs):
    ref, _, _, _, traj = outputs
    rq = ref["qpos"]
    assert traj.shape == rq.shape
    # base position (world), joint angles: convention-free, compare directly
    base_pos_err = np.abs(traj[:, :3] - rq[:, :3]).max()
    joint_err = np.abs(traj[:, 7:] - rq[:, 7:]).max()
    # base quaternion: account for double-cover sign ambiguity
    dq = np.minimum(np.abs(traj[:, 3:7] - rq[:, 3:7]).sum(1),
                    np.abs(traj[:, 3:7] + rq[:, 3:7]).sum(1)).max()
    assert base_pos_err < 1e-4, base_pos_err
    assert joint_err < 1e-4, joint_err
    assert dq < 1e-4, dq
