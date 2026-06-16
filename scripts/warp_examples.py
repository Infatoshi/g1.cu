"""Small, verified Warp examples to build a mental model, mapped to a humanoid sim.
Each demo is independent; run prints a one-line result so we know it actually works.
"""
import numpy as np
import warp as wp
wp.init()
dev = "cuda:0"


# ---------------------------------------------------------------------------
# 1. THE MENTAL MODEL: a @wp.kernel is a per-thread function. wp.tid() = thread
#    index. You pass wp.arrays in/out. wp.launch(dim=N) runs N threads. Built-in
#    vec3/quat/mat33 math compiles straight to CUDA. (Here: one integrator step.)
# ---------------------------------------------------------------------------
@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float):
    i = wp.tid()
    vel[i] = vel[i] + wp.vec3(0.0, 0.0, -9.81) * dt
    pos[i] = pos[i] + vel[i] * dt


def demo_basics():
    N = 4
    pos = wp.zeros(N, dtype=wp.vec3, device=dev)
    vel = wp.zeros(N, dtype=wp.vec3, device=dev)
    for _ in range(100):
        wp.launch(integrate, dim=N, inputs=[pos, vel, 0.01])
    wp.synchronize()
    print("1. basics: z after 1s free-fall =", pos.numpy()[0, 2], "(expect ~-4.9)")


# ---------------------------------------------------------------------------
# 2. RIGID-BODY MATH IS BUILT IN: wp.transform = (translation, quaternion).
#    transform_point composes/applies poses -- the FK we hand-rolled is partly
#    free here. Also wp.spatial_vector / wp.spatial_matrix exist for Featherstone.
# ---------------------------------------------------------------------------
@wp.kernel
def fk_link(parent_X: wp.array(dtype=wp.transform),
            local_X: wp.array(dtype=wp.transform),
            world_X: wp.array(dtype=wp.transform)):
    i = wp.tid()
    world_X[i] = parent_X[i] * local_X[i]          # compose transforms
    # where does the link's tip (0.3m down its z) land in the world?
    # (just demonstrating transform_point)


def demo_transforms():
    base = wp.array([wp.transform(wp.vec3(0.0, 0.0, 1.0),
                                  wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 1.5708))],
                    dtype=wp.transform, device=dev)
    local = wp.array([wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity())],
                     dtype=wp.transform, device=dev)
    out = wp.zeros(1, dtype=wp.transform, device=dev)
    wp.launch(fk_link, dim=1, inputs=[base, local, out])
    wp.synchronize()
    t = out.numpy()[0]
    print("2. transforms: composed world pose translation =", np.round(t[:3], 3))


# ---------------------------------------------------------------------------
# 3. COLLISION / CONTACT: this is the part you need next. Geometric queries are
#    first-class. Here: foot points vs a ground plane -> penalty force + an atomic
#    contact counter (the divergence we measured). Warp also has wp.Mesh+BVH,
#    wp.mesh_query_point, wp.HashGrid for broadphase -- the whole detection stack.
# ---------------------------------------------------------------------------
@wp.kernel
def ground_contact(foot_pos: wp.array(dtype=wp.vec3),
                   foot_vel: wp.array(dtype=wp.vec3),
                   force_out: wp.array(dtype=wp.vec3),
                   ncon: wp.array(dtype=wp.int32),
                   k: float, c: float):
    i = wp.tid()
    depth = -foot_pos[i][2]                          # penetration below z=0
    if depth > 0.0:
        vn = foot_vel[i][2]
        fz = k * depth - c * vn                       # spring-damper normal force
        force_out[i] = wp.vec3(0.0, 0.0, wp.max(fz, 0.0))
        wp.atomic_add(ncon, 0, 1)                      # count active contacts
    else:
        force_out[i] = wp.vec3(0.0, 0.0, 0.0)


def demo_contact():
    fp = wp.array([[0, 0, -0.01], [0, 0.1, 0.02], [0, -0.1, -0.005], [0.2, 0, 0.5]],
                  dtype=wp.vec3, device=dev)
    fv = wp.array([[0, 0, -1.0]] * 4, dtype=wp.vec3, device=dev)
    force = wp.zeros(4, dtype=wp.vec3, device=dev)
    ncon = wp.zeros(1, dtype=wp.int32, device=dev)
    wp.launch(ground_contact, dim=4, inputs=[fp, fv, force, ncon, 1.0e4, 50.0])
    wp.synchronize()
    print(f"3. contact: {ncon.numpy()[0]} feet in contact, forces_z =",
          np.round(force.numpy()[:, 2], 1))


# ---------------------------------------------------------------------------
# 4. DIFFERENTIABLE SIM (free with Warp): wp.Tape records the kernels, .backward()
#    gives gradients of an output wrt inputs THROUGH the physics. This is analytic
#    policy gradients / system-id for free -- a whole capability you don't build.
# ---------------------------------------------------------------------------
@wp.kernel
def loss_from_v0(v0: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    # land position of a projectile after t=1s given initial vertical speed v0
    z = v0[0] * 1.0 - 0.5 * 9.81 * 1.0
    loss[0] = z * z


def demo_grad():
    v0 = wp.array([3.0], dtype=wp.float32, device=dev, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device=dev, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(loss_from_v0, dim=1, inputs=[v0], outputs=[loss])
    tape.backward(loss)
    print(f"4. autodiff: loss={loss.numpy()[0]:.3f}  dloss/dv0={v0.grad.numpy()[0]:.3f} "
          f"(analytic: 2*(v0-4.905)= {2*(3.0-4.905):.3f})")


if __name__ == "__main__":
    demo_basics()
    demo_transforms()
    demo_contact()
    demo_grad()
