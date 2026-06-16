"""Probe: can a Warp kernel hold a per-thread 35x35 matrix + do a Cholesky solve?"""
import warp as wp
wp.init()
NV = 35
vecN = wp.types.vector(length=NV, dtype=wp.float32)
matN = wp.types.matrix(shape=(NV, NV), dtype=wp.float32)


@wp.kernel
def probe(out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    M = matN()
    b = vecN()
    for i in range(NV):
        b[i] = float(i) + 1.0
        for j in range(NV):
            M[i, j] = wp.where(i == j, 2.0, 0.1)
    for i in range(NV):
        for j in range(i + 1):
            s = M[i, j]
            for k in range(j):
                s = s - M[i, k] * M[j, k]
            if i == j:
                M[i, i] = wp.sqrt(s)
            else:
                M[i, j] = s / M[j, j]
    x = vecN()
    for i in range(NV):
        s = b[i]
        for k in range(i):
            s = s - M[i, k] * x[k]
        x[i] = s / M[i, i]
    for i in range(NV - 1, -1, -1):
        s = x[i]
        for k in range(i + 1, NV):
            s = s - M[k, i] * x[k]
        x[i] = s / M[i, i]
    out[tid] = x[0] + x[NV - 1]


o = wp.zeros(4, dtype=wp.float32)
wp.launch(probe, dim=4, inputs=[o])
wp.synchronize()
print("PROBE OK, sample out:", o.numpy())
