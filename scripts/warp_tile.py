"""Warp tile API: block-level cooperative matmul -> tensor cores. This is how the
policy MLP (and a block-per-world dense solve) maps to mma on the GPU. One block
computes one output tile cooperatively (block_dim threads), not one thread/world.
"""
import numpy as np
import warp as wp
wp.init()
dev = "cuda:0"

# batched MLP layer: Y[N,H] = X[N,F] @ W[F,H], tiled over rows.
F, H = 64, 128
TILE_N = 32
BLOCK = 64


@wp.kernel
def mlp_layer(X: wp.array2d(dtype=wp.float16),
              W: wp.array2d(dtype=wp.float16),
              Y: wp.array2d(dtype=wp.float32)):
    g = wp.tid()                                   # one block per row-tile
    x = wp.tile_load(X, shape=(TILE_N, F), offset=(g * TILE_N, 0))
    w = wp.tile_load(W, shape=(F, H))
    y = wp.tile_zeros(shape=(TILE_N, H), dtype=wp.float32)
    wp.tile_matmul(x, w, y)                        # mma; uses tensor cores for fp16
    wp.tile_store(Y, y, offset=(g * TILE_N, 0))


def main():
    N = 256
    Xh = np.random.randn(N, F).astype(np.float16)
    Wh = np.random.randn(F, H).astype(np.float16)
    X = wp.array(Xh, dtype=wp.float16, device=dev)
    W = wp.array(Wh, dtype=wp.float16, device=dev)
    Y = wp.zeros((N, H), dtype=wp.float32, device=dev)
    wp.launch_tiled(mlp_layer, dim=[N // TILE_N], inputs=[X, W, Y], block_dim=BLOCK)
    wp.synchronize()
    ref = Xh.astype(np.float32) @ Wh.astype(np.float32)
    err = np.abs(Y.numpy() - ref).max() / np.abs(ref).max()
    print(f"tile matmul {N}x{F} @ {F}x{H}: rel err vs numpy = {err:.2e} (fp16 tensor-core)")


if __name__ == "__main__":
    main()
