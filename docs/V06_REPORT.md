# Humanoid-1 v0.6: measured single support and two-step advance

The new skill lifts the right foot, advances and lands it, transfers support,
lifts and advances the left foot, and returns to standing. All six final 30-second
trials passed on MuJoCo 3.17.0. The nominal trial also passed on MuJoCo 3.2.7.
This is an engineered slow flat-floor sequence, not continuous walking, a learned
locomotion policy, a physical-robot result, or completion of the book scope.

## Measured motion

The 16-actuator, 32.8 kg MJCF, core safety code and existing standing/weight-shift
controllers are unchanged from v0.5. Only joint motor commands move the robot.
There are no welds, root forces, state teleports, kinematic foot attachments or
hidden assistance. The two disturbed cases include explicitly logged external
pushes; these are disturbances, not stabilizing control.

| Case | Foot travel left/right (mm) | Minimum verified single-support dwell (s) | Peak torso tilt (rad) | Passed |
|---|---:|---:|---:|---|
| Nominal | 39.95 / 39.83 | 1.762 | .08574 | yes |
| Random commands, seed 101 | 44.38 / 44.25 | 1.852 | .08449 | yes |
| Random commands, seed 102 | 36.56 / 36.46 | 1.856 | .08649 | yes |
| Random commands, seed 103 | 38.07 / 37.96 | 1.754 | .08333 | yes |
| Random commands + pushes, seed 201 | 44.86 / 44.74 | 1.796 | .08605 | yes |
| Random commands + pushes, seed 202 | 38.43 / 38.34 | 1.858 | .08524 | yes |

Nominal maximum sole clearances were 25.06 mm left and 24.76 mm right, measured
at the lowest transformed sole corner. Verified single-support dwell requires
the swing foot to be at least 15 mm above the floor with less than 1 N load,
the stance foot to carry at least 90% of body weight, and measured geometric
support margin to remain at least 20 mm. This cannot be satisfied merely by
commanding a foot to rise or by showing a COM shift.

Across the six cases: minimum achieved maximum clearance per foot was 24.22 mm;
minimum support margin was 48.40 mm; maximum stance-foot drift was .743 mm;
maximum final COM error was .750 mm; maximum requested/applied torque was
43.35 Nm; and there were no saturated actuators, nominal joint-limit events,
or safety stops. Full precision and per-case values are in
[`evidence/v0.6/engineered/summary.json`](../evidence/v0.6/engineered/summary.json).

## Controller and safety

`contact_control.py` adds COM/torso Jacobian-derivative terms, swing-foot motion
tasks and a bounded contact-wrench allocation QP. The QP balances the free-root
dynamics while imposing nonnegative normal loads and linear friction, CoP and
torsion bounds. CoP bounds account for the sole's offset from the ankle origin.
The active-set solver uses NumPy only and checks feasibility and residuals;
an infeasible or failed solve latches a stop rather than applying an unbounded
least-squares fallback.

These are approximate contact-wrench bounds, not the exact frictional wrench
cone or a complete acceleration/torque-constrained whole-body optimization.
Actual contacts, clearances and motion are independently evaluated in physics.
Torque clipping remains at the original motor limits and any saturation blocks
task acceptance. The inverse-dynamics references are not applied as world forces.

Existing core joint ranges, torque limits, 0.07 rad emergency tolerance, .60 m
pelvis minimum, velocity-norm limit and stop latch are retained unchanged. The
new task also requires no nominal joint-range event, torso tilt <= .10 rad,
pelvis height .95–1.05 m, COM speed <= .10 m/s, COM tracking error <= .03 m,
foot tracking error <= .015 m, and stance drift <= .003 m.

Standing and weight-shift criteria are not changed. Their bilateral contact
requirement applies to those skills. Stepping deliberately uses a separate,
explicit support schedule: unexpected loss of a planned support foot remains
a stop condition, while the designated swing foot may leave the ground. The
stepping COM reference follows a fixed bounded program, rather than requiring
the COM to stay within 30 mm of its original standing location.

The runtime's persisted operator stop is checked at every physics substep.
Tests include a stop written by a second process and a core latch that prevents
any further simulation stepping until reset. Physical emergency-stop behavior
is a separate future implementation; freezing simulation time is not one.

## Regression and integration

The original controller comparison was rerun. v0.1 at 20 ms stops at 1.630 s;
v0.2 at 20 ms stops at 1.782 s; v0.1 at 2 ms stops at 1.814 s. v0.2 at 2 ms and
all seven v0.3 cases complete 30 s. Timing remains an important contributor;
v0.6 does not claim a new standing improvement. The baseline comparison is
committed in `evidence/v0.6/baseline_regression.json`.

The local runtime accepts `take two steps`, executes the real program, saves
measured foot displacements/clearance/support dwell as spatial memory, and stores
task/episodic evidence with hashes. A real second process successfully recalled
the completed task and checked its evidence. Each request still starts a fresh
simulation episode; this is not continuous navigation or general language
planning. `walk`, manipulation and other unsupported requests still fail before
motion starts. No learned policy is promoted or activated.

The suite contains contact feasibility/infeasibility tests, offset-foot bounds,
seeded/continuous references, core safety faults, short-run false-success checks,
runtime cancellation, and a real 30-second stepping regression. Local tests pass
on MuJoCo 3.17.0 and 3.2.7. CI runs the suite on Linux/macOS with both versions,
plus the prior PPO checks and real runtime task/recall checks. See the PR checks
for the actual CI status; Mac compatibility does not imply an M3-specific
performance measurement.

## Evidence and reproduction

```bash
python -m pip install -r requirements-dev.txt
OPENBLAS_NUM_THREADS=1 python -m pytest -q
OPENBLAS_NUM_THREADS=1 python -m humanoid1.stepping --suite --output results/v06
python -m humanoid1.runtime --state-dir results/my_humanoid execute "take two steps"
python -m humanoid1.runtime --state-dir results/my_humanoid recall TASK_ID
# macOS passive viewer:
mjpython -m humanoid1.stepping --view
```

`trace_10hz.csv` files are complete review samples from finished physics-step
CSVs. Acceptance and extrema use all 15,000 physics samples per 30-second case.
The exporter verifies final time, sample count and per-trial review rows, and
records full CSV hashes. Full position/velocity/torque traces regenerate under
`results/`; they are not all duplicated into Git.

[`two_steps.mp4`](../evidence/v0.6/two_steps.mp4) is an actual 750-frame, 25 fps
recording with a simultaneous foot close-up, actual loads, measured clearance
and forward foot travel. It is not generated animation. The recording has its
own passed trial report. Optional recording dependencies are Pillow and ffmpeg:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 python scripts/record_stepping.py
```

EGL is a headless Linux rendering option, not a requirement for the CPU physics
baseline or macOS viewer. Model/source/artifact hashes accompany the evidence.

## Full TOC status and next work

The latest `Humanoids_Master_TOC_Research_Linked_45_Chapters(1).docx` was read and
audited: all 45 chapter titles and 587 numbered sections/appendix entries match
the register. Chapter 9 gains partial evidence for contact scheduling,
single-support balance, kinematics, dynamics and coordinated motion. No chapter
is marked complete, and the continuous walking/running entry remains planned.

Next motion work is repeated gait, longer forward travel, command tracking,
stopping, recovery and disturbance tests. Subsequent milestones still include
actuator/power/thermal modeling; articulated hands and touch; vision and sensory
fusion; advanced local reasoning, voice and world models; measured useful
learning; social behavior; connectivity/fault handling; and integrated acceptance.
Clinical/biological replacement, longevity and subjective-experience claims need
separate research and evidence. A reference implementation cannot establish
those claims by filling in software modules.
