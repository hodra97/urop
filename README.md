# Robot Kinematics with MuJoCo

Python implementations of product-of-exponentials (PoE) forward kinematics,
space/body Jacobians, and numerical full-pose inverse kinematics, verified against
MuJoCo. Supports Franka Panda (7 arm joints) and UR5e (6).
All `franka_*.py` entry points work with either robot.

## Installation

Use Python 3.12 or newer for the pinned dependencies in `requirements.txt`.
The commands below use Bash; the project has been tested on Linux with Python 3.12.
GUI programs require a graphical desktop and OpenGL support.

This repository contains the Python project only. Download the robot models
separately from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
After cloning this repository, run these commands from its root directory:

```bash
# Skip cloning if this sibling checkout already exists.
git clone https://github.com/google-deepmind/mujoco_menagerie.git ../mujoco_menagerie
git -C ../mujoco_menagerie checkout "$(cat MENAGERIE_COMMIT.txt)"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The model profiles expect this layout; keep the model XML files and their assets together:

```text
workspace/
├── franka_mujoco/                 # This repository
│   ├── README.md
│   ├── requirements.txt
│   └── ...
└── mujoco_menagerie/              # Separate model repository
    ├── franka_emika_panda/
    └── universal_robots_ur5e/
```

`MENAGERIE_COMMIT.txt` pins the reference model revision. If models are stored
elsewhere, update `XML_PATH` in the corresponding robot profile.

## Run

Activate `.venv` and run from this repository's root:

```bash
export ROBOT_MODULE=ur5e_model  # Use panda_model for Panda; default when unset
python franka_fk.py            # Compare FK against MuJoCo
python franka_jacobian.py      # Print and verify space/body Jacobians
python franka_fk_GUI.py        # Verify FK, then animate home <-> target
python franka_ik_GUI.py        # Solve a target pose, print joints, animate and hold
```

For one command only: `ROBOT_MODULE=panda_model python franka_fk_GUI.py`.
Use no spaces around `=`. A separate assignment without `export` may not reach
Python. `unset ROBOT_MODULE` restores the default. Robot selection is independent
of virtual-environment activation; `GUI` in the filename is uppercase.

## FK joint inputs and viewer

Edit `JOINT_OFFSETS` in the selected robot's input file. The profile loads it
automatically for FK, Jacobian verification, and the GUI.

| ROBOT_MODULE | Input file | End-effector body |
|---|---|---|
| `panda_model` | `panda_joint_inputs.py` — 7 offsets | `hand` |
| `ur5e_model` | `ur5e_joint_inputs.py` — 6 offsets | `wrist_3_link` |

**Target joint coordinates = home coordinates + offsets.** Zero preserves home.
Units are radians for revolute joints and metres for prismatic joints.
Invalid names, dimensions, nonfinite values, and out-of-limit targets are rejected.
Save edits and restart the program to reload inputs; no terminal restart is needed.

The GUI prints joint start/target/offset values and repeats a two-second trip in
each direction. It updates `qpos` with `mj_forward()` rather than simulating actuator
tracking. To animate only one joint's configured offset:

```bash
ROBOT_MODULE=ur5e_model python franka_fk_GUI.py --joint elbow_joint
```

Other joint coordinates stay at home, although descendant links move with their
parent. Omit `--joint` to animate all joints together. Both FK and IK viewers show
a world XY grid (0.10 m spacing, major lines every 0.50 m, extent +/-1.5 m), with
red x, green y, and blue z axes. The top-right panel displays the desired world
position (m) and roll/pitch/yaw (degrees). In FK, this pose is calculated from the
selected goal joint coordinates. These overlays do not change the physical model.

`python franka_demo.py` runs a separate `mj_step()` physics demonstration. It does
not command the arm to the input offsets; a configured gripper opens/closes every
two simulation seconds.

## IK pose inputs and viewer

Edit the shared `end_effector_inputs.py` for either robot:

```python
TARGET_POSITION = [0.45, 0.10, 0.45]  # world x, y, z in metres
TARGET_RPY_DEG = [180.0, 0.0, 0.0]   # roll, pitch, yaw in degrees
INITIAL_JOINTS = None                # home; or n absolute joint coordinates
```

These are absolute world-frame coordinates for the profile's end-effector **body
origin**, not home offsets or a gripper/tool tip. Orientation is
`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`, mapping body coordinates into world.
The same target can be reachable for one robot and unreachable for another.
IK does not use the FK joint-offset files. Save input edits and restart the program.

```bash
ROBOT_MODULE=ur5e_model python franka_ik_GUI.py
ROBOT_MODULE=panda_model python franka_ik_GUI.py --no-viewer
# Optional one-run input overrides and a clearly unreachable example:
python franka_ik_GUI.py --position 0.45 0.10 0.45 --rpy-deg 180 0 0
python franka_ik_GUI.py --no-viewer --position 100 0 0
```

`IK_algorithm.py` uses the body Jacobian, SE(3) logarithmic error, weighted damped
least squares, joint limits, backtracking, and deterministic restarts. Successful
solutions meet 0.1 mm position and 0.001 rad orientation tolerances and are checked
against MuJoCo. The terminal prints absolute final joint values (rad and degrees
for hinges, metres for slides), residuals, and iteration counts.

| Status | Meaning |
|---|---|
| `SUCCESS` | A joint-limited solution meets both pose tolerances |
| `IMPOSSIBLE: out of task-space` | Target position exceeds a conservative geometric reach bound |
| `NO_CONVERGENCE` | No solution found within the search budget; does not prove unreachability |
| `INVALID_INPUT` | Invalid pose, initial joints, or solver settings |
| `VERIFICATION_FAILED` | Candidate failed the independent MuJoCo pose check |

The reach bound is not an exact workspace test; it cannot rule out every
impossible orientation or joint-limited pose. `--max-iterations` (default 250 per
attempt) and `--restarts` (default 8 additional guesses) control the search budget.
Exit codes are 0 for success, 1 for solver/verification failure, and 2 for invalid input.

The GUI draws target axes (x red, y green, z blue) and overlays status, live joint
values, and pose errors. The desired pose panel preserves the RPY values supplied
by the input file or CLI, including during a failed solve. On success, it interpolates from the initial joints to the
solution in `--duration` seconds (default 3), then holds until the window closes.
On solver failure it holds the initial pose and displays the reason. Invalid inputs
are rejected before opening a window. This is joint-space visualization, not a
straight Cartesian path or collision-aware motion plan.

## Modules and conventions

| File | Responsibility |
|---|---|
| `robot_config.py` | Select a profile and call its `load_robot()` |
| `panda_model.py`, `ur5e_model.py` | Configure model paths, endpoint, home, optional gripper, and input module |
| `robot_model.py` | Discover the arm chain; expose limits, position/velocity indices, and home geometry |
| `joint_inputs.py` | Apply and validate offsets; generate verification configurations |
| `FK_algorithm.py` | MuJoCo-independent n-joint PoE FK returning a 4x4 endpoint transform |
| `jacobian_algorithm.py` | Screw axes, Adjoint transforms, and 6xn space/body Jacobians |
| `end_effector_inputs.py` | Shared world position, RPY orientation, and optional initial joints for IK |
| `IK_algorithm.py` | NumPy-only joint-limited full-pose IK and SE(3) error calculations |
| `franka_ik_GUI.py` | Solve and verify IK, print final joints, and visualize status/motion |
| `viewer_overlays.py` | Shared world grid, desired pose axes, and position/orientation text |

Twists use `[omega; v]` order. Space coordinates use the world frame; body
coordinates use the configured endpoint frame. The lower half of `J_s @ qdot`
is not the endpoint position rate `p_dot`; verification converts the MuJoCo result
to the same convention before comparison.

## Tests

```bash
python -m unittest discover -v
```

- `test_jacobian_algorithm.py`: central-difference checks against FK across joint counts/types, transform identities, and input validation.
- `test_robot_model.py`: Panda/UR5e/synthetic-model FK and Jacobians, chain discovery, indices, input isolation, and viewer loops/options.
- `test_IK_algorithm.py`: pose conventions, zero/pi rotation logs, reachable/unreachable targets, limits, both robots, and IK viewer behavior.

Tests need both model folders and exercise both profiles regardless of
`ROBOT_MODULE`. Viewer tests use mocks and do not open windows. FK/Jacobian entry
points compare home, target, and 10 random configurations; sampling settings live
in `joint_inputs.py`.

## Adding a robot

Create a joint-input module and a profile exposing `load_robot()`. Return
`RobotModel.from_xml_path(xml_path, end_effector_body, ...)`, providing
`home_keyframe` and the input module's `JOINT_OFFSETS` as `default_offsets`.
Select the profile through `ROBOT_MODULE`. Setting `home_keyframe=None` uses model `qpos0`.

Supported chains have a fixed base and hinge/slide joints. Downstream fingers and
unrelated scene objects are excluded. Ball/free joints in the arm chain, mocap
ancestors, and closed-chain constraint solving are unsupported.
