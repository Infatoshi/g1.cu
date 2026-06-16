"""Offscreen-render the recorded walk trajectory (bench/walk_traj.npy) to an MP4.

Headless EGL render on gamer (no display needed). A tracking camera follows the base
so the robot stays in frame as it walks ~5m forward. Kinematic playback: set qpos per
frame, mj_forward, render.

  uv run python scripts/render_mp4.py [out.mp4] [traj.npy]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU render, no X display

import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bench", "walk.mp4")
TRAJ = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "bench", "walk_traj.npy")

W, H, FPS = 960, 540, 50  # control dt = 0.02s -> 50 fps real-time


def main():
    traj = np.load(TRAJ)
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "models", "scene.xml"))
    m.vis.global_.offwidth = W   # enlarge offscreen framebuffer (default 640x480)
    m.vis.global_.offheight = H
    d = mujoco.MjData(m)
    assert traj.shape[1] == m.nq, f"traj nq {traj.shape[1]} != model nq {m.nq}"

    renderer = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance = 4.0
    cam.azimuth = 90.0     # look along +y so forward (+x) walks left-to-right across frame
    cam.elevation = -15.0

    frames = []
    for t in range(traj.shape[0]):
        d.qpos[:] = traj[t]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        # track the base so the walker stays centered
        cam.lookat[:] = d.qpos[0:3]
        renderer.update_scene(d, cam)
        frames.append(renderer.render())

    imageio.mimsave(OUT, frames, fps=FPS, quality=8, macro_block_size=None)
    dx = float(traj[-1, 0] - traj[0, 0])
    print(f"wrote {OUT}  ({len(frames)} frames @ {FPS}fps = {len(frames)/FPS:.1f}s, "
          f"{W}x{H}); base advanced {dx:.2f} m")


if __name__ == "__main__":
    main()
