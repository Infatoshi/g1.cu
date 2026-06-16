"""Offscreen-render the finish-line trajectory (bench/finish_traj.npy) to an MP4, with a
bright finish-line gate drawn at x=X_FIN. Headless EGL (no display). Tracking camera follows
the base so the robot stays framed as it walks toward the line.

  uv run python scripts/render_finish.py [out.mp4] [traj.npy]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bench", "finish.mp4")
TRAJ = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "bench", "finish_traj.npy")

W, H, FPS = 960, 540, 50
X_FIN = 5.0


def add_finish_line(scn, x):
    """Append a bright vertical gate + a ground stripe at x as user geoms to the scene."""
    for spec in (
        # thin tall green slab spanning the path (the 'gate')
        dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 1.2, 1.0], pos=[x, 0.0, 1.0],
             rgba=[0.1, 0.9, 0.2, 0.55]),
        # ground stripe on the floor at the line
        dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 1.2, 0.002], pos=[x, 0.0, 0.001],
             rgba=[0.1, 0.9, 0.2, 0.9]),
    ):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, spec["type"], np.array(spec["size"]),
                            np.array(spec["pos"]), np.eye(3).flatten(),
                            np.array(spec["rgba"], np.float32))
        scn.ngeom += 1


def main():
    traj = np.load(TRAJ)
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "models", "scene.xml"))
    m.vis.global_.offwidth = W
    m.vis.global_.offheight = H
    d = mujoco.MjData(m)
    assert traj.shape[1] == m.nq, f"traj nq {traj.shape[1]} != model nq {m.nq}"

    renderer = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance = 4.0
    cam.azimuth = 90.0
    cam.elevation = -12.0

    frames = []
    for t in range(traj.shape[0]):
        d.qpos[:] = traj[t]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        cam.lookat[:] = d.qpos[0:3]
        renderer.update_scene(d, cam)
        add_finish_line(renderer.scene, X_FIN)
        frames.append(renderer.render())

    imageio.mimsave(OUT, frames, fps=FPS, quality=8, macro_block_size=None)
    dx = float(traj[-1, 0] - traj[0, 0])
    print(f"wrote {OUT}  ({len(frames)} frames @ {FPS}fps = {len(frames)/FPS:.1f}s, {W}x{H}); "
          f"base advanced {dx:.2f} m toward finish at {X_FIN} m")


if __name__ == "__main__":
    main()
