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
python trajectory_GUI_v2.py      # CLIK with adaptive reference time; nominal 5 s, total budget 10 s
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
Model construction, IK, and Cartesian trajectory execution do not read these FK inputs.

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

## V2: CLIK with adaptive reference time

V2 uses one CLIK path. It does not call static `solve_ik()` at each waypoint.
V1 remains available as `trajectory_solver_v1.py` and `trajectory_GUI_v1.py`;
its algorithms and CLI options are preserved. The unversioned trajectory solver
and GUI names have been removed. Single-pose `ik_GUI.py` remains independent.

```bash
# Defaults: Panda + straight position / SLERP orientation
python trajectory_GUI_v2.py

# Choose robot and reference independently
ROBOT_MODULE=ur5e_model TRAJECTORY_MODULE=straight_line_trajectory \
  python trajectory_GUI_v2.py --no-viewer

# Optional short motion; identical flags work with Panda, UR5e, and TidyBot
ROBOT_MODULE=tidybot_model python trajectory_GUI_v2.py \
  --offset 0.02 0 0 --rotation-offset-deg 0 0 10
```

The initial joints and final pose come from `end_effector_inputs.py` unless
`--position`, `--rpy-deg`, `--offset`, or `--rotation-offset-deg` overrides a
component. Relative position is in world coordinates; relative orientation is
postmultiplied onto the starting rotation. Model loading never reads FK offsets.

The nominal reference takes `--duration 5` seconds. `--max-time 10` is the total
motion-time budget, including any slow progress or final correction, **not** an
additional ten seconds after nominal arrival. A successful solve completes the
reference and meets both final pose tolerances. Exceeding the nominal duration
alone is not failure. Failure reasons include `TIMEOUT`, `STALLED` (no meaningful
progress/error reduction for `--stall-time 1` second), `NUMERICAL_FAILURE`, and
`VERIFICATION_FAILED`. Failure retains the last accepted state. Exit codes are
0 for success, 1 for solver/verification failure, and 2 for invalid inputs/I/O.

The two clocks are motion/playback time $t$ and nominal reference time $\tau$:

$$
t_{k+1}=t_k+h,\qquad \tau_{k+1}=\min(T,\tau_k+\alpha_k h),\qquad 0\le\alpha_k\le1.
$$

The reference returns `sample(tau) -> (T_d, V_d)` where $V_d$ is differentiated
with respect to $\tau$. Its feedforward speed is scaled by $\alpha$; pose feedback
remains active even when reference progress pauses. `tau/duration` is a nominal
time fraction, not necessarily geometric path progress. For the default straight
reference, $u=\tau/T$ and $s=3u^2-2u^3$ determine position and SLERP progress.
Virtual clock scaling follows the principle described in
[ROS 2 Control's speed scaling documentation](https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/speed_scaling.html);
this implementation chooses the rate from its own kinematic constraints/checks.

At each state, v2 separates translational and rotational feedback. With current
pose $(R,p)$ and nominal desired-body twist $V_d=[\omega_d;v_d]$, it commands

$$
V_c=\begin{bmatrix}
\alpha R^\top R_d\omega_d + K\operatorname{Log}(R^\top R_d)^\vee\\
R^\top\big(\alpha R_d v_d+K(p_d-p)\big)
\end{bmatrix}.
$$

A NumPy active-set solver computes the bounded DLS velocity. For step length
$h$ and per-joint speed caps $c_i$, the optimization and elementwise bounds are

$$
\dot q^*=\underset{\ell\le v\le u}{\arg\min}
\left(\|WJ_bv-WV_c\|^2+\lambda^2\|v\|^2\right),\qquad
\ell_i=\max\left(-c_i,\frac{q_{\min,i}-q_i}{h}\right),\quad
u_i=\min\left(c_i,\frac{q_{\max,i}-q_i}{h}\right).
$$

These bounds enforce joint speeds and the next joint coordinates. Bound joints
are handled by resolving the remaining variables, rather than clipping all
coordinates after an unconstrained solve. The default $W$ and damping remain
`--orientation-scale 0.3` and `--damping 0.001`; `--gain` defaults to 5.
The experimental speed caps are `--revolute-speed 1` rad/s and
`--prismatic-speed 0.1` m/s; these are not manufacturer actuator ratings.

Candidate steps use explicit midpoint integration. Let $f(q,\tau,\alpha)$ be the
bounded velocity above, using the same bounds computed at the interval's start:

$$
v_0=f(q_k,\tau_k,\alpha),\qquad
q_{1/2}=q_k+\frac{h}{2}v_0,\qquad
q_{k+1}=q_k+h\,f\left(q_{1/2},\tau_k+\frac{\alpha h}{2},\alpha\right).
$$

The governor first tries
full reference speed, then searches smaller alpha values if the next segment
fails its pose checks. This is a checked, bounded phase search, not a globally
time-optimal planner. Defaults are `--dt 0.01` and ten phase-search refinements.
Every candidate is checked at its endpoint and at 1/4, 1/2, and 3/4 of the joint
segment, against the corresponding reference times. Rejected candidates are
never added to the playable path. Accepted segments are independently checked
again using MuJoCo before playback. For each candidate it checks
$q_\beta=(1-\beta)q_k+\beta q_{k+1}$ against
$T_d((1-\beta)\tau_k+\beta\tau_{k+1})$ for
$\beta\in\{1/4,1/2,3/4,1\}$. When full progress fails, it tests a zero-progress
correction and bisects the alpha interval ten times, retaining a tested valid
candidate. If no movement passes, it holds the last checked state. The final
interval caps alpha at $(T-\tau_k)/h$ so the reference does not overshoot.

`--position-tolerance 0.0001` m and `--orientation-tolerance 0.001` rad bound
errors against **the pose at tau** at those probes. Proximity to that reference
point also bounds distance to the geometric position path. It is a stronger
condition than merely lying somewhere on the same curve. Finite probes do not
prove an exactly zero error at every continuous instant; physical actuator
tracking, collision avoidance, and acceleration/jerk limits are outside this
kinematic implementation. Reported acceleration is a diagnostic only.

The v2 viewer plays the recorded states once and holds the last pose. It does
not alternate methods or automatically reset, and adds no hidden one-second
pause. It shows motion time, reference time, alpha, pose errors, the sampled
desired curve in yellow, and the verified actual path in green. It uses `qpos`
and `mj_forward()`, not `mj_step()` actuator control. A verification failure
holds the initial pose instead of replaying a rejected result.

Logs go to a new `runs/trajectory_v2_TIMESTAMP/` directory, or `--output NEW_PATH`:

- `reference.csv`: nominal reference poses/twists versus tau.
- `clik.csv`: t, tau, alpha, reference-time fraction, desired/native poses,
  pose and final-goal errors, checked segment errors, joints, derivatives,
  bound activity, rejected candidate counts, and calculation time.
- `summary.json`: model and trajectory module, settings, clock endpoints,
  final-goal errors, maximum checked errors, limits and MuJoCo verification.

Alpha describes the interval ending at each row; row zero uses alpha=0.
`checked_path_within_tolerance` concerns the accepted prefix even on timeout;
only `SUCCESS` also means that the entire reference was completed. Motion time
is distinct from the time Python spends calculating the offline trajectory.

### Selecting and adding a reference module

`trajectory_config.load_profile()` reads `TRAJECTORY_MODULE` at call time,
using `straight_line_trajectory` by default. `--trajectory-module MODULE` overrides
it for one v2 run. `unset TRAJECTORY_MODULE` restores the default. Importing the
loader does not select a module or load a robot.

A new module needs this factory; its internal class name is unrestricted:

```python
# my_trajectory.py
# Implement MyTrajectory in this module or import it from your reference code.
def create_trajectory(start, goal, duration):
    return MyTrajectory(start, goal, duration)
```

The returned object must expose:

| Member | Contract |
|---|---|
| `start`, `goal` | Requested world-frame SE(3) transforms, shape `(4, 4)` |
| `duration` | Requested positive nominal duration in seconds |
| `sample(tau)` | Return `(pose, twist)` for `0 <= tau <= duration` |
| `pose` | World-frame SE(3) pose of the selected end-effector body origin |
| `twist` | Finite six-vector `[omega; v]` in the desired body's frame, differentiated with respect to tau |

`sample(0)` and `sample(duration)` must agree with the endpoints. Position and
orientation should form a smooth reference with a consistent analytic twist;
returning world linear velocity as the body-twist component is incorrect.
Additional path parameters, such as arc centers, belong in that module/factory.
V2 does not depend on a concrete class, a straight-line formula, or `sample_times()`.
Use `TRAJECTORY_MODULE=my_trajectory` without changing solver/viewer code or
adding a registry entry.

## V1: archived DLS IK / CLIK comparison

`trajectory_GUI_v1.py` always computes **both** independent joint trajectories:
pointwise DLS IK and CLIK, with the same initial joints, desired poses, times,
joint limits, speed caps, and error tolerances. One method failing does not prevent
the other from running. Neither consumes the other's joint solution.

The default initial joints and final world pose come from `end_effector_inputs.py`.
Use `--position` and `--rpy-deg` to override the final pose, as in the single-pose
IK program. A reachable final pose does not guarantee a feasible straight path.
For a small example tested on Panda, UR5e, and TidyBot:

```bash
ROBOT_MODULE=tidybot_model python trajectory_GUI_v1.py \
  --offset 0.02 0 0 --rotation-offset-deg 0 0 10 --duration 3 --dt 0.02

# Calculate and save without opening a window:
ROBOT_MODULE=panda_model python trajectory_GUI_v1.py --no-viewer \
  --offset 0.02 0 0 --rotation-offset-deg 0 0 10 --duration 3 --dt 0.02
```

`--offset` adds a world-frame translation to the initial endpoint position.
`--rotation-offset-deg` postmultiplies the initial rotation by `Rz Ry Rx`, so its
angles describe a **local-frame** increment. These override position and orientation
independently: omitting either override uses that component from the shared input
file. They are mutually exclusive with the respective absolute CLI options.

The reference generator in `straight_line_trajectory.py` uses

$$
u=\operatorname{clip}(t/T,0,1),\quad s=3u^2-2u^3,\qquad
p_d=p_0+s(p_f-p_0),\qquad
R_d=R_0\exp(s[\phi]_\times),\quad \phi=\operatorname{Log}(R_0^\top R_f)^\vee.
$$

The rotation expression is equivalent to shortest-arc quaternion SLERP. At a
180-degree difference, either direction is shortest; the rotation-log convention
selects one. RPY angles themselves are not interpolated. The shared time scaling
gives zero translational and angular velocities at both ends. The desired-body
twist is $V_d=[\dot s\phi; R_d^\top\dot p_d]$, with
$\dot s=6u(1-u)/T$ and $\dot p_d=\dot s(p_f-p_0)$.
V1 samples this reference at motion time $t$; v2 uses virtual time $\tau$ and
multiplies the nominal twist by alpha in its controller.

`trajectory_solver_v1.py` receives numerical geometry, without loading a robot:

- **DLS IK:** call the existing `solve_ik()` at each reference sample, starting
  from the previous accepted joints. Restarts are disabled to avoid random branch
  jumps. Stop at the last accepted sample if IK fails or if an interval would
  exceed a joint speed cap (`SPEED_LIMIT`); never substitute a false solution.
- **CLIK:** with $E=T(q)^{-1}T_d$, form
  $V_c=\operatorname{Ad}_E V_d+K\operatorname{Log}(E)^\vee$. Apply a weighted,
  damped body-Jacobian inverse to obtain joint velocity, scale to speed caps,
  project into joint limits, and integrate $q_{k+1}=q_k+\Delta t\dot q_k$.
  This is one feedback update per time interval. CLIK also uses DLS for the
  velocity inverse; its procedure differs from converging static IK per waypoint.
  The frame transport follows the [Modern Robotics task-space velocity controller](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-3-motion-control-with-velocity-inputs-part-3-of-3/), using proportional feedback without an integral term.

For the pointwise method, $q_k=\operatorname{IK}(T_d(t_k);q_{k-1})$ uses the
weighted DLS iteration described under **IK** below, with `restarts=0`.
The candidate is accepted only if every
$|q_{k,i}-q_{k-1,i}|/(t_k-t_{k-1})\le c_i$ (up to numerical slack).
It does not retime the reference when this check fails.

For CLIK, let $F$ contain free joints after freezing fixed joints and joints at
a limit whose weighted gradient points outward. With
$W=\operatorname{diag}(0.3,0.3,0.3,1,1,1)$ and $A=(WJ_b)_{[:,F]}$,

$$
\dot q_F=A^\top(AA^\top+\lambda^2I_6)^{-1}WV_c,\qquad
\dot q_{\bar F}=0,\qquad
\gamma=\max\left(1,\max_i\frac{|\dot q_i|}{c_i}\right),\qquad
q_{k+1}=\operatorname{clip}\left(q_k+\Delta t\frac{\dot q}{\gamma},q_{\min},q_{\max}\right).
$$

V1 advances reference time on schedule even when $\gamma>1$, so the robot can
lag and leave the desired straight path. Its velocity scaling differs from
v2's constrained velocity solve and adaptive reference clock.

Defaults: `--duration 5`, `--dt 0.01`, `--settle-time 1`, `--gain 5`,
`--damping 0.001`, `--orientation-scale 0.3`, `--max-iterations 250`,
`--position-tolerance 0.0001`, `--orientation-tolerance 0.001`,
`--revolute-speed 1` rad/s, `--prismatic-speed 0.1` m/s. The speed caps are
experiment settings, not actuator specifications. The reference holds the final
pose during settle time; both methods receive this same interval. The exact
arrival time is included even when duration is not a multiple of dt.

Every accepted joint sample is checked against MuJoCo. Output goes to a unique
`runs/trajectory_TIMESTAMP/` directory, or a new path supplied with `--output`:

- `reference.csv`: the complete desired poses and desired-body twists.
- `dls_ik.csv`, `clik.csv`: time, desired/native poses, pose errors, joints,
  finite-difference joint velocities/accelerations, solve times, and limit flags.
- `summary.json`: initial/goal poses, model/endpoint, settings, joint metadata,
  maximum and sample RMS errors, errors at time T, goal settling time, limits,
  computation times, and FK/MuJoCo agreement. Unlimited bounds use JSON `null`.

Existing output directories are rejected instead of overwritten. DLS failure
logs contain only the accepted prefix, with failure time in the summary; missing
time-T errors are `null`. `SUCCESS` means the **final** pose, including any settle
time, meets tolerance. `path_within_tolerance` requires a complete recorded
trajectory and checks every recorded sample. CLIK may finish successfully despite
larger transient tracking errors.
Exit codes: 0 if both succeed, 1 if either solver/verification fails, 2 for invalid
input or an output/model-loading error. `NO_CONVERGENCE` is not proof of impossibility.

The viewer alternates between both recorded joint paths, restarting each at the
same initial posture. Each method gets `duration + settle_time + 1` seconds:
the final extra second is a viewer-only hold, so the default cycle is 7 seconds
per method even though the recorded trajectory ends at 6 seconds.
It displays the desired line in yellow, DLS IK in cyan,
CLIK in green, a world grid, moving desired axes, and live errors. Failed paths
hold their last accepted sample. Playback uses sample-and-hold `qpos` updates and
`mj_forward()`. It does not claim an exactly straight continuous path between
samples, actuator tracking, acceleration limits, or collision avoidance.

The straight reference concerns the configured **body origin**. For TidyBot this
is `bracelet_link`, not an offset gripper tip; rotating the wrist can make that
offset tip follow a different path. Both trajectory solvers may move TidyBot's
planar base as well as its arm.

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
| `end_effector_inputs.py` | Shared world position, RPY orientation, and optional initial joints for IK and Cartesian trajectories |
| `IK_algorithm.py` | NumPy-only joint-limited full-pose IK and SE(3) error calculations |
| `fk.py`, `jacobian.py` | `run_checks(robot, offsets)` compares analytical results against MuJoCo |
| `fk_GUI.py` | Verify FK and animate the supplied robot, including `--joint` selection |
| `ik_GUI.py` | Solve and verify IK, print final joints, and visualize the supplied robot |
| `straight_line_trajectory.py` | Shared straight-position / SLERP mathematics plus `create_trajectory()` factory |
| `trajectory_solver_v1.py` | Preserved v1 DLS IK / CLIK comparison |
| `trajectory_config.py` | Select a reference module using `TRAJECTORY_MODULE`; invoke its factory explicitly |
| `trajectory_solver_v2.py` | Bounded CLIK, virtual reference clock, and accepted-segment pose checks |
| `trajectory_GUI_v1.py` | Preserved v1 verification, exports, and alternating playback |
| `trajectory_GUI_v2.py` | Load the selected reference/model, verify v2 results with MuJoCo, export and replay once |
| `viewer_overlays.py` | Shared world grid, desired pose axes/text, and sampled path drawing |

## Execution flow

Imports define functions and constants without loading XML or opening a viewer.
Each FK/Jacobian/IK/trajectory entry point explicitly creates its robot in `main()`:

1. `robot_config.load_profile()` reads `ROBOT_MODULE` at call time, defaulting to
   `panda_model`. An explicit module name can also be supplied.
2. `robot_config.load_robot(profile)` calls the profile's factory and returns a
   new `RobotModel`, with its own MuJoCo model/data and captured home geometry.
3. FK and Jacobian entry points call `joint_inputs.load_offsets(profile)` and pass
   the offsets alongside the robot. IK and trajectories read `end_effector_inputs.py`
   separately. V1 compares two methods. V2 loads the selected trajectory profile and runs only CLIK.
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
same commands above. IK and Cartesian trajectories use `end_effector_inputs.py`;
the new trajectory modules require no robot-specific branches.

Supported chains have a fixed base and hinge/slide joints. Downstream fingers and
unrelated scene objects are excluded. Ball/free joints in the arm chain, mocap
ancestors, and closed-chain constraint solving are unsupported.
