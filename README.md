# Single-arm FK, Jacobians, and MuJoCo Viewer

Robot-independent product-of-exponentials (PoE) forward kinematics and space/body
Jacobians, checked against MuJoCo. Supports Panda (7 arm joints) and UR5e (6).
All `franka_*.py` entry points work with either profile. IK is not implemented yet.

## Setup and execution

Run from `franka_mujoco/`. Model assets must be available in the sibling folders
`../mujoco_menagerie/franka_emika_panda/` and
`../mujoco_menagerie/universal_robots_ur5e/`.
`MENAGERIE_COMMIT.txt` records the reference model revision.

```bash
# First-time setup
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# On later runs, activate the environment and select a robot
export ROBOT_MODULE=ur5e_model  # Use panda_model for Panda; default when unset
python franka_fk.py            # Compare FK against MuJoCo
python franka_jacobian.py      # Print and verify space/body Jacobians
python franka_fk_GUI.py        # Verify FK, then animate home <-> target
```

`export` applies to subsequent programs launched from that terminal. For one run:
`ROBOT_MODULE=panda_model python franka_fk_GUI.py`.
Do not put spaces around `=`. A standalone assignment without `export` may not
reach Python. `unset ROBOT_MODULE` restores the default selection.
Robot selection is independent of virtual-environment activation; `GUI` is uppercase.

## Joint inputs and visualization

Edit `JOINT_OFFSETS` in the selected robot's input file. The profile loads it
automatically, so FK, Jacobian verification, and the GUI share the same target.

| ROBOT_MODULE | Input file | End-effector body |
|---|---|---|
| `panda_model` | `panda_joint_inputs.py` — 7 offsets | `hand` |
| `ur5e_model` | `ur5e_joint_inputs.py` — 6 offsets | `wrist_3_link` |

- **Target joint coordinates = home coordinates + offsets.** Zero preserves home; units are radians for revolute joints and metres for prismatic joints.
- Save edits and restart the program to reload inputs. Restarting the terminal is unnecessary.
- `joint_inputs.py` applies offsets and validates names, dimensions, finite values, and joint limits; keep robot-specific values in the input files above.

The GUI prints each joint's start, target, and offset, then repeats a two-second
trip in each direction. It assigns `qpos` and calls `mj_forward()`; it does not
simulate actuator tracking or solve IK. To inspect one joint:

```bash
ROBOT_MODULE=ur5e_model python franka_fk_GUI.py --joint elbow_joint
```

Only that joint receives its configured offset; other joint coordinates stay at
home. Descendant links still move with their parent. Omit `--joint` to move all joints.

`python franka_demo.py` runs a separate `mj_step()` physics demonstration.
It does not command the arm to the input offsets. If configured, the gripper
alternates open/closed every two simulation seconds.

## Code structure

| File | Responsibility |
|---|---|
| `robot_config.py` | Select a profile through the environment and call `load_robot()` |
| `panda_model.py`, `ur5e_model.py` | Configure XML, endpoint, home, optional gripper, and input module |
| `robot_model.py` | Discover the arm chain; expose limits, position/velocity indices, and home geometry |
| `joint_inputs.py` | Apply and validate offsets; generate verification configurations |
| `fk_algorithm.py` | MuJoCo-independent n-joint PoE FK returning a 4x4 endpoint transform |
| `jacobian_algorithm.py` | Screw axes, Adjoint transforms, and 6xn space/body Jacobians |

Twists use `[omega; v]` order. The space frame is world; the body frame is the
configured endpoint. The lower half of `J_s @ qdot` is not the endpoint position
rate `p_dot`; the MuJoCo comparison explicitly converts between these conventions.

## Tests and extension

```bash
python -m unittest discover -v
```

- `test_jacobian_algorithm.py`: checks Jacobians against central differences of FK, across joint counts/types; validates transforms and invalid inputs.
- `test_robot_model.py`: checks Panda, UR5e, and synthetic models; covers FK/Jacobians, chain discovery, indices, input isolation, and GUI loops/options.

Integration tests exercise both profiles regardless of `ROBOT_MODULE`. Viewer tests
use mocks and do not open windows. FK/Jacobian entry points compare home, target,
and 10 random configurations; sampling count and seed live in `joint_inputs.py`.

To add a robot, create its input module and a profile with `load_robot()` returning
`RobotModel.from_xml_path(xml_path, end_effector_body, ...)`. Supply a
`home_keyframe` and the input module's `JOINT_OFFSETS` as `default_offsets`, then
select the profile via `ROBOT_MODULE`. With `home_keyframe=None`, home uses model `qpos0`.

Supported chains have a fixed base and hinge/slide joints. Downstream fingers and
unrelated scene objects are excluded. Ball/free joints in the arm chain, mocap
ancestors, and closed-chain constraint solving are unsupported.

Study materials live in `../study_guides/`. Regenerate the PDF with
`build_jacobian_guide.py`; its additional dependencies are `reportlab`,
`matplotlib`, `numpy`, and `pillow`.
