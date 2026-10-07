"""Shared visual-only world grid, target axes, and pose text for FK/IK viewers."""

import mujoco
import numpy as np


GRID_SPACING = 0.1       # metres; minor lines every 10 cm
GRID_HALF_EXTENT = 1.5  # metres around the world origin
GRID_MAJOR_EVERY = 5    # major lines every 50 cm
AXIS_COLORS = ([1, 0.2, 0.2, 1], [0.2, 1, 0.2, 1], [0.2, 0.4, 1, 1])


def _connector(scene, start, end, color, *, arrow=False, width=1.0, label=""):
    if scene.ngeom >= scene.maxgeom:
        return
    kind = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_LINE
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                       np.array(color, dtype=np.float32))
    mujoco.mjv_connector(geom, kind, width, np.asarray(start, dtype=float), np.asarray(end, dtype=float))
    geom.label = label
    scene.ngeom += 1


def draw_world_grid(scene):
    """Append an XY grid just above z=0, plus coloured world axes. No collisions."""
    height = 0.003  # Avoid depth fighting with the model's floor plane.
    count = int(round(GRID_HALF_EXTENT / GRID_SPACING))
    for index in range(-count, count + 1):
        if index == 0:
            continue  # The world x/y axes occupy these lines.
        value = index * GRID_SPACING
        major = index % GRID_MAJOR_EVERY == 0
        color = [0.85, 0.88, 0.92, 1] if major else [0.45, 0.55, 0.65, 1]
        width = 1.5 if major else 1.0
        _connector(scene, [-GRID_HALF_EXTENT, value, height], [GRID_HALF_EXTENT, value, height], color, width=width)
        _connector(scene, [value, -GRID_HALF_EXTENT, height], [value, GRID_HALF_EXTENT, height], color, width=width)
    _connector(scene, [-GRID_HALF_EXTENT, 0, height], [GRID_HALF_EXTENT, 0, height], AXIS_COLORS[0], width=2)
    _connector(scene, [0, -GRID_HALF_EXTENT, height], [0, GRID_HALF_EXTENT, height], AXIS_COLORS[1], width=2)
    _connector(scene, [0, 0, height], [0, 0, 0.5], AXIS_COLORS[2], arrow=True, width=0.004, label="World Z")


def draw_target_axes(scene, target):
    """Append the desired body's frame; preserve the grid and any other geometry."""
    origin = target[:3, 3]
    for index, color in enumerate(AXIS_COLORS):
        _connector(scene, origin, origin + 0.12 * target[:3, index], color,
                   arrow=True, width=0.006, label="Desired" if index == 0 else "")


def rotation_to_rpy_deg(rotation):
    """Canonical RPY for Rz(yaw) Ry(pitch) Rx(roll); roll=0 at gimbal lock."""
    cosine_pitch = np.hypot(rotation[0, 0], rotation[1, 0])
    pitch = np.arctan2(-rotation[2, 0], cosine_pitch)
    if cosine_pitch > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return np.rad2deg([roll, pitch, yaw])


def desired_pose_text(target, rpy_deg=None):
    """Top-right desired world pose. Preserve explicitly supplied input angles."""
    rpy = rotation_to_rpy_deg(target[:3, :3]) if rpy_deg is None else np.asarray(rpy_deg)
    labels = ["Desired pose", "x (m)", "y (m)", "z (m)", "roll (deg)",
              "pitch (deg)", "yaw (deg)", "Rotation order", "World XY grid", "Major lines"]
    values = ["World frame", *(f"{value:.4f}" for value in target[:3, 3]),
              *(f"{value:.2f}" for value in rpy), "Rz Ry Rx",
              f"{GRID_SPACING:.2f} m", f"{GRID_SPACING * GRID_MAJOR_EVERY:.2f} m"]
    return (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "\n".join(labels), "\n".join(values))
