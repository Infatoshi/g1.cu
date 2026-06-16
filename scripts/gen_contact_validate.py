"""Emit bench/contact_validate.bin: pure-foot-ground states + MuJoCo full qacc, for the
CUDA full-physics path (src/sim_contact.cu validate). fp32, little-endian.

Layout: int32 nstates; then qpos[nstates*NQ], qvel[nstates*NV], qacc[nstates*NV] (float32);
then int32 label_code[nstates] (0=ballistic,2=mid,3=hi). Only pure foot-ground states are
included (ballistic + mid + hi buckets); lo/tail in the oracle carry arm/leg self-collisions
that the foot-ground solver does not model.

The CUDA path recomputes qacc_smooth from its own ABA and the contact correction from scratch;
we compare its full qacc to MuJoCo's stored qacc (the real target).

Run: uv run python scripts/gen_contact_validate.py
"""
import os, struct
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "bench", "contact_solve_ref.npz")
OUT = os.path.join(HERE, "bench", "contact_validate.bin")
NQ, NV = 36, 35
FOOT_GEOMS = {15, 16, 17, 18, 30, 31, 32, 33}
LABEL_CODE = {"ballistic": 0, "lo": 1, "mid": 2, "hi": 3, "tail": 4}


def is_pure_foot_ground(ref, k):
    g1, g2 = ref["cgeom1"][k], ref["cgeom2"][k]
    if len(g1) == 0:
        return True
    return all((a == 0 and b in FOOT_GEOMS) or (b == 0 and a in FOOT_GEOMS)
               for a, b in zip(g1.tolist(), g2.tolist()))


def main():
    ref = np.load(REF, allow_pickle=True)
    qpos, qvel, qacc, labels = [], [], [], []
    for k in range(len(ref["label"])):
        if not is_pure_foot_ground(ref, k):
            continue
        qpos.append(ref["qpos"][k].astype(np.float32))
        qvel.append(ref["qvel"][k].astype(np.float32))
        qacc.append(ref["qacc"][k].astype(np.float32))
        labels.append(LABEL_CODE[str(ref["label"][k])])
    n = len(qpos)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<i", n))
        np.array(qpos, dtype="<f4").tofile(f)
        np.array(qvel, dtype="<f4").tofile(f)
        np.array(qacc, dtype="<f4").tofile(f)
        np.array(labels, dtype="<i4").tofile(f)
    print(f"wrote {OUT}  ({n} pure-foot-ground states; codes {labels})")


if __name__ == "__main__":
    main()
