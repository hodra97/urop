# Robot Kinematics with MuJoCo

Python implementations of product-of-exponentials (PoE) forward kinematics,
space/body Jacobians, and numerical full-pose inverse kinematics, verified against
MuJoCo. Supports Franka Panda (7 arm joints), UR5e (6), and TidyBot
(3 planar-base joints + 7 arm joints). The FK/Jacobian/IK entry points work
with every profile below.

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
    ├── universal_robots_ur5e/
    └── stanford_tidybot/
```

`MENAGERIE_COMMIT.txt` pins the reference model revision. If models are stored
elsewhere, update `XML_PATH` in the corresponding robot profile.

## Run

Activate `.venv` and run from this repository's root:

```bash
export ROBOT_MODULE=tidybot_model  # Or panda_model (default), ur5e_model
python fk.py                     # Compare FK against MuJoCo
python jacobian.py               # Print and verify space/body Jacobians
python fk_GUI.py                 # Verify FK, then animate home <-> target
python ik_GUI.py                 # Solve a target pose, print joints, animate and hold
```

For one command only: `ROBOT_MODULE=panda_model python fk_GUI.py`.
Use no spaces around `=`. A separate assignment without `export` may not reach
Python. `unset ROBOT_MODULE` restores the default. Robot selection is independent
of virtual-environment activation; `GUI` in the filename is uppercase.

Use the entry-point names above. The previous compatibility entry points and
physics demonstration have been removed; FK and single-pose IK CLI options remain available.

## FK joint inputs and viewer

Edit `JOINT_OFFSETS` in the selected robot's input file. The profile names it with
`FK_INPUT_MODULE`; FK, Jacobian verification, and the FK GUI load it when run.
Model construction and IK do not read these FK inputs.

| ROBOT_MODULE | Input file | End-effector body |
|---|---|---|
| `panda_model` | `panda_joint_inputs.py` — 7 offsets | `hand` |
| `ur5e_model` | `ur5e_joint_inputs.py` — 6 offsets | `wrist_3_link` |
| `tidybot_model` | `tidybot_joint_inputs.py` — 10 offsets | `bracelet_link` |

**Target joint coordinates = home coordinates + offsets.** Zero preserves home.
Units are radians for revolute joints and metres for prismatic joints.
Invalid names, dimensions, nonfinite values, and out-of-limit targets are rejected.
Save edits and restart the program to reload inputs; no terminal restart is needed.

TidyBot's first three coordinates are `joint_x` and `joint_y` (metres), then
`joint_th` (yaw in radians). `joint_1` through `joint_7` are the arm's revolute
joints. Its input file includes example offsets for both base and arm.
Setting the three base offsets to zero keeps the base at home during FK;
IK independently uses all ten joints. Gripper finger joints are excluded.

The GUI prints joint start/target/offset values and repeats a two-second trip in
each direction. It updates `qpos` with `mj_forward()` rather than simulating actuator
tracking. To animate only one joint's configured offset:

```bash
ROBOT_MODULE=ur5e_model python fk_GUI.py --joint elbow_joint
ROBOT_MODULE=tidybot_model python fk_GUI.py --joint joint_x
```

Other joint coordinates stay at home, although descendant links move with their
parent. Omit `--joint` to animate all joints together. Both FK and IK viewers show
a world XY grid (0.10 m spacing, major lines every 0.50 m, extent +/-1.5 m), with
red x, green y, and blue z axes. The top-right panel displays the desired world
position (m) and roll/pitch/yaw (degrees). In FK, this pose is calculated from the
selected goal joint coordinates. These overlays do not change the physical model.

## IK pose inputs and viewer

Edit the shared `end_effector_inputs.py` for any selected robot:

```python
TARGET_POSITION = [0.50, 0.40, 0.44]  # world x, y, z in metres
TARGET_RPY_DEG = [145.0, 45.0, 70.0]  # roll, pitch, yaw in degrees
INITIAL_JOINTS = None                # home; or n absolute joint coordinates
```

These are absolute world-frame coordinates for the profile's end-effector **body
origin**, not home offsets or a gripper/tool tip. Orientation is
`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`, mapping body coordinates into world.
The same target can be reachable for one robot and unreachable for another.
IK does not use the FK joint-offset files. Save input edits and restart the program.

```bash
ROBOT_MODULE=tidybot_model python ik_GUI.py
ROBOT_MODULE=panda_model python ik_GUI.py --no-viewer
# Optional one-run input overrides and an out-of-reach UR5e example:
python ik_GUI.py --position 0.45 0.10 0.45 --rpy-deg 180 0 0
ROBOT_MODULE=ur5e_model python ik_GUI.py --no-viewer --position 100 0 0
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

TidyBot's planar base has unlimited joint ranges in this model, so there is no
finite positional reach bound. A distant target can exhaust the IK search budget
and return `NO_CONVERGENCE` rather than `IMPOSSIBLE`. IK may move the base as well
as the arm; there is no base-lock CLI option.

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
| `robot_config.py` | `load_profile()` selects a profile; `load_robot(profile)` explicitly creates a fresh robot |
| `panda_model.py`, `ur5e_model.py`, `tidybot_model.py` | Configure XML, endpoint, home, optional gripper, and the FK input module name |
| `panda_joint_inputs.py`, `ur5e_joint_inputs.py`, `tidybot_joint_inputs.py` | Store named FK offsets relative to home |
| `robot_model.py` | Discover the arm chain; own MuJoCo model/data, limits, indices, and home geometry |
| `joint_inputs.py` | Load FK offsets on request, validate them against a robot, and generate verification configurations |
| `FK_algorithm.py` | MuJoCo-independent n-joint PoE FK returning a 4x4 endpoint transform |
| `jacobian_algorithm.py` | Screw axes, Adjoint transforms, and 6xn space/body Jacobians |
| `end_effector_inputs.py` | Shared world position, RPY orientation, and optional initial joints for IK |
| `IK_algorithm.py` | NumPy-only joint-limited full-pose IK and SE(3) error calculations |
| `fk.py`, `jacobian.py` | `run_checks(robot, offsets)` compares analytical results against MuJoCo |
| `fk_GUI.py` | Verify FK and animate the supplied robot, including `--joint` selection |
| `ik_GUI.py` | Solve and verify IK, print final joints, and visualize the supplied robot |
| `viewer_overlays.py` | Shared world grid and desired pose axes/text |

## Execution flow

Imports define functions and constants without loading XML or opening a viewer.
Each FK/Jacobian/IK entry point explicitly creates its robot in `main()`:

1. `robot_config.load_profile()` reads `ROBOT_MODULE` at call time, defaulting to
   `panda_model`. An explicit module name can also be supplied.
2. `robot_config.load_robot(profile)` calls the profile's factory and returns a
   new `RobotModel`, with its own MuJoCo model/data and captured home geometry.
3. FK and Jacobian entry points call `joint_inputs.load_offsets(profile)` and pass
   the offsets alongside the robot. IK reads `end_effector_inputs.py` separately.
4. Verification, output, and viewer helpers receive the robot explicitly. The
   numeric algorithms continue to receive arrays and joint types.

For example, to verify a robot from Python:

```python
import joint_inputs
import robot_config
from fk import run_checks

profile = robot_config.load_profile("ur5e_model")
robot = robot_config.load_robot(profile)
offsets = joint_inputs.load_offsets(profile)
run_checks(robot, offsets)
```

`selected_joint_angles(robot, offsets)` and `test_configurations(robot, offsets)`
take task inputs explicitly; omitting `offsets` means zero offsets (home).
`RobotModel` stores geometry and simulation state, not FK targets. Programmatic
calls such as `solve_target(robot, target, initial)` and
`visualize_fk_motion(robot, start, goal)` use the supplied instance, allowing
multiple robots in one process. The renamed entry points keep the existing CLI options.

## Kinematics algorithms

All three algorithms use home geometry from `robot_model.py`:
$q_{\mathrm{home}}$, world-frame unit axes $a_i$, points $r_i$ on those axes,
and endpoint pose $M=T(q_{\mathrm{home}})$. MuJoCo supplies reference geometry
and verification; the calculations use NumPy. Twists are ordered $[\omega;v]$.
The IK settings below use the solver defaults.

### FK: product of exponentials (`FK_algorithm.py`)

With $\theta_i=q_i-q_{\mathrm{home},i}$, `fk_manual()` evaluates

$$
T(q)=e^{[S_1]\theta_1}\cdots e^{[S_n]\theta_n}M,\qquad
S_i=
\begin{cases}
[a_i;\,-a_i\times r_i] & \text{revolute},\\
[0;\,a_i] & \text{prismatic}.
\end{cases}
$$

`joint_transform()` uses Rodrigues' rotation $R_i$ and translation
$(I-R_i)r_i$ for hinges, or translation $a_i\theta_i$ for slides.
Transforms multiply in chain order; home need not be all-zero.

### Jacobians: screw axes and Adjoint (`jacobian_algorithm.py`)

The space Jacobian transforms each home screw by the preceding joints:

$$
J_{s,i}=\operatorname{Ad}_{P_{i-1}}S_i,\quad
P_{i-1}=\prod_{j=1}^{i-1}e^{[S_j]\theta_j},\quad P_0=I,
\qquad
\operatorname{Ad}_{[R,p]}=
\begin{bmatrix}R&0\\{[p]_\times R}&R\end{bmatrix}.
$$

The body Jacobian is $J_b=\operatorname{Ad}_{T(q)^{-1}}J_s$.
Both are $6\times n$, with $V_s=J_s\dot q$ and $V_b=J_b\dot q$.
Their linear components are $v_s=\dot p-\omega_s\times p$ and
$v_b=R^\top\dot p$. `jacobian.py` converts MuJoCo's endpoint
Jacobian to these conventions; tests independently check them by finite differences.

### IK: weighted DLS with constrained search (`IK_algorithm.py`)

1. Validate inputs and joint limits; reject positions beyond a conservative
   reach bound plus position tolerance. Passing this check does not establish reachability.
2. Evaluate PoE FK, the body-frame logarithmic error, and weights:

$$
e_b=\operatorname{Log}\!\left(T(q)^{-1}T_d\right)^\vee,\qquad
W=\operatorname{diag}(c,c,c,1,1,1),\quad c=0.3\ \mathrm{m/rad}.
$$

3. Freeze fixed joints and joints at a limit where
   $g=(WJ_b)^\top We_b$ points outward. For free columns $F$,
   set $A=(WJ_b)_{[:,F]}$ and solve

$$
\Delta q_F=A^\top(AA^\top+\lambda^2I_6)^{-1}We_b,\qquad
\Delta q_{\bar F}=0,\qquad
q_{\mathrm{trial}}=\operatorname{clip}(q+\alpha\Delta q,q_{\min},q_{\max}).
$$

4. Scale the whole step to at most 0.3 rad per hinge / 0.03 m per slide.
   Backtrack $\alpha=1,1/2,\ldots,1/32$, requiring a decrease greater than
   $10^{-12}$ in $\|We_b\|$. Retry with
   $\lambda_0,10\lambda_0,100\lambda_0$ as needed
   ($\lambda_0=10^{-3}$). On a stalled step or exhausted budget,
   restart with four local and four wider guesses by default (RNG seed 7).
5. Require both $\|p_d-p\|\le10^{-4}$ m and
   $\|e_{b,\omega}\|\le10^{-3}$ rad. Check position separately because the
   log's linear part generally differs from $p_d-p$. On `NO_CONVERGENCE`, return
   the best weighted-residual candidate with `success=False`.

The GUI verifies successful solutions with MuJoCo, then interpolates
$q(t)=q_0+(3u^2-2u^3)(q_f-q_0)$, $u=\operatorname{clip}(t/T,0,1)$.
This is a single-pose solve followed by joint-space animation, not CLIK or
Cartesian path control.

## Tests

```bash
python -m unittest discover -v
```

- `test_jacobian_algorithm.py`: central-difference checks against FK across joint counts/types, transform identities, and input validation.
- `test_robot_model.py`: Panda/UR5e/synthetic-model FK and Jacobians, chain discovery, indices, input isolation, and viewer loops/options.
- `test_IK_algorithm.py`: pose conventions, zero/pi rotation logs, reachable/unreachable targets, limits, both robots, and IK viewer behavior.

Tests need the Panda and UR5e model folders and exercise these profiles regardless of
`ROBOT_MODULE`. Viewer tests use mocks and do not open windows. FK/Jacobian entry
points compare home, target, and 10 random configurations; sampling settings live
in `joint_inputs.py`.

The test files are development checks and are not imported by normal execution.
Keep them to catch regressions; the FK GUI's built-in FK verification is a
separate check. Adding a profile does not automatically add model-specific test cases.

## Adding a robot

Add the external XML/assets, a profile, and an FK input module. No robot registry
or changes to common algorithms, entry points, or viewers are needed.

For example, create `my_robot_model.py`:

```python
from pathlib import Path
from robot_model import RobotModel

XML_PATH = Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "my_robot" / "scene.xml"
END_EFFECTOR_BODY_NAME = "tool"  # A body name in the XML
HOME_KEYFRAME = None            # Uses model qpos0; or an XML keyframe name
FK_INPUT_MODULE = "my_robot_joint_inputs"  # An import name, not an import statement

def load_robot():
    return RobotModel.from_xml_path(
        XML_PATH, END_EFFECTOR_BODY_NAME,
        home_keyframe=HOME_KEYFRAME,
        gripper_actuator=None,  # Optional actuator metadata
    )
```

Create `my_robot_joint_inputs.py`, using your XML's joint names:

```python
JOINT_OFFSETS = {"joint_a": 0.0, "joint_b": 0.1}
```

Missing joints receive zero offsets. The FK input module is imported only for
FK tasks; keep the profile's `load_robot()` independent of task inputs.
Select the new robot with `export ROBOT_MODULE=my_robot_model`, then use the
same commands above. IK uses `end_effector_inputs.py`.

Supported chains have a fixed base and hinge/slide joints. Downstream fingers and
unrelated scene objects are excluded. Ball/free joints in the arm chain, mocap
ancestors, and closed-chain constraint solving are unsupported.
