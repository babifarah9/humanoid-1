# Humanoid-1 v0.3 engineering and verification report

Date: 29 October 2026. Starting repository commit: `4893c1eba564069baaf8950661409d8848ea1e02`.

## Outcome

v0.3 completed seven 30-second standing trials on MuJoCo 3.17.0 and a nominal 30-second compatibility trial on 3.2.7. Every v0.3 trial passed the declared envelope, with zero nominal joint-limit events, zero torque saturations and no safety stop. The nominal cases use only joint motors and floor contact: there is no weld, tether, root actuator, stabilizing external force or model-parameter change. The push cases add only the documented disturbance.

## Baseline diagnosis

The original controllers were reproduced using their original 20 ms command interval and the unchanged 2 ms physics timestep. v0.1 stops at **1.630 s** and v0.2 at **1.782 s**, both because pelvis height drops below **0.60 m**. Both exhibit nominal knee and elbow range crossings before falling. None crosses the existing range-plus-0.07-rad emergency threshold before the height stop. A nominal range excursion and an emergency joint-limit stop are different observations.

The baseline holds explicit damping/PD torques for ten physics steps. The timing ablation removes nominal joint crossings in v0.1 at 2 ms but does not prevent its fall; v0.2 at 2 ms completes 30 seconds. Thus sample-and-hold timing is a major measured contributor to the original instability. Contact-aware dynamics, reference posture and feedback also change in v0.3; these experiments do not isolate a unique contribution from every term or prove that inverse dynamics is necessary for quiet standing.

### First nominal range crossing for every affected joint

All positions are radians, velocities rad/s and torques generalized actuator Nm. Values below are sampled at actual integration time (2 ms resolution). Full precision crossing events and worst excursions are retained in `evidence/v0.3/main/summary.json`. Torque is what the previous physics step applied, captured before a stop zeroes commands.

| Controller | Joint | Time (s) | Position | Velocity | Torque | Nominal range |
|---|---|---:|---:|---:|---:|---|
| v01 | left_knee | 0.282 | -0.106390219067 | -13.296108741405 | 100.000000000 | [-0.1, 2.3] |
| v01 | right_knee | 0.282 | -0.106390219067 | -13.296108741405 | 100.000000000 | [-0.1, 2.3] |
| v01 | left_elbow | 1.328 | 0.101153462854 | 0.656199708653 | -30.000000000 | [-2.5, 0.1] |
| v01 | right_elbow | 1.328 | 0.101153487840 | 0.656199765600 | -30.000000000 | [-2.5, 0.1] |
| v02 | left_knee | 0.084 | -0.108721779592 | -6.488991513984 | 100.000000000 | [-0.1, 2.3] |
| v02 | right_knee | 0.084 | -0.108721779592 | -6.488991513984 | 100.000000000 | [-0.1, 2.3] |
| v02 | left_elbow | 1.286 | 0.100331067694 | 2.178111289645 | -30.000000000 | [-2.5, 0.1] |
| v02 | right_elbow | 1.326 | 0.101386694186 | 2.197575617651 | -30.000000000 | [-2.5, 0.1] |

### Worst nominal range excursions

| Controller | Joint | Time (s) | Position | Velocity | Torque | Excess beyond range (rad) |
|---|---|---:|---:|---:|---:|---:|
| v01 | left_knee | 0.366 | -0.165024659077 | -2.136040435962 | 100.000000000 | 0.065024659077 |
| v01 | right_knee | 0.366 | -0.165024659077 | -2.136040435962 | 100.000000000 | 0.065024659077 |
| v01 | left_elbow | 1.608 | 0.137380669951 | 0.645130350372 | -30.000000000 | 0.037380669951 |
| v01 | right_elbow | 1.608 | 0.137380841899 | 0.645126146589 | -30.000000000 | 0.037380841899 |
| v02 | left_knee | 0.286 | -0.166317989891 | -1.682230569563 | 100.000000000 | 0.066317989891 |
| v02 | right_knee | 0.286 | -0.166317989891 | -1.682230569564 | 100.000000000 | 0.066317989891 |
| v02 | left_elbow | 1.488 | 0.134895405204 | 0.700402231301 | -30.000000000 | 0.034895405204 |
| v02 | right_elbow | 1.608 | 0.138384412533 | 0.757466967362 | -30.000000000 | 0.038384412533 |

## Controller design

1. Recompute kinematics, gravity/Coriolis bias, passive forces, inertia and measured floor/foot loads. Initial posture is the existing valid reset pose; v0.3 does not request the baselines' immediate knee bend.
2. Form posture, whole-body COM and torso orientation acceleration tasks. Foot position/orientation constraints hold the nominal double-support stance. Foot Jacobian time derivatives account for contact-frame acceleration; COM/torso Jacobian derivatives are neglected near rest.
3. Solve the small constrained least-squares acceleration system using rank-revealing SVD. Both feet impose a redundant yaw condition, so directly inverting that system is singular.
4. Allocate predicted supporting wrenches to satisfy free-root force/moment dynamics. Compute only actuated torques from `M*a + bias - passive - J_contact.T*wrench`. No predicted wrench is applied to the world or floating root.
5. Add joint-range repulsion inside the original bounds, clip all commands to unchanged motor limits, and run the independent emergency latch at each physics step. Contact-loss, non-foot contact, excessive tilt, invalid residual and numerical-solve failures additionally stop the whole-body controller.

Gravity compensation must account for the load path through the feet. Simply adding actuated entries of `qfrc_bias` does not balance the unactuated root. The new allocation explicitly models this load path. However, it is weighted least squares, **not** a friction-cone/CoP constrained QP; predicted wrench feasibility is not guaranteed under large disturbances.

## Declared acceptance envelope

The test includes 15,000 physics steps (30 seconds at 2 ms). After 0.10 s contact settling, **every physics sample** must satisfy: torso tilt <=0.10 rad; COM speed <=0.10 m/s; COM displacement from the nominal target <=0.03 m; pelvis height 0.95–1.05 m; each foot normal load >=1 N; no non-foot floor contact; geometric COM projection margin >=0.02 m inside the convex hull of loaded sole footprints. No nominal joint-range event is allowed anywhere, including startup. Existing emergency checks are never exempt. The support margin is geometric, not a measured ZMP or proof of dynamic balance.

## Reproducible comparisons

| Controller | Command interval | Scenario / seed | Elapsed (s) | Standing envelope | Joint crossing events | Stop |
|---|---:|---|---:|---|---:|---|
| v01 | 20 ms | nominal / 0 | 1.630 | FAIL | 46 | pelvis below floor-clearance threshold |
| v02 | 20 ms | nominal / 0 | 1.782 | FAIL | 33 | pelvis below floor-clearance threshold |
| v01 | 2 ms | nominal / 0 | 1.814 | FAIL | 0 | pelvis below floor-clearance threshold |
| v02 | 2 ms | nominal / 0 | 30.000 | PASS | 0 | none |
| v03 | 2 ms | nominal / 0 | 30.000 | PASS | 0 | none |
| v03 | 20 ms | nominal / 0 | 30.000 | PASS | 0 | none |
| v03 | 2 ms | initial_velocity / 1 | 30.000 | PASS | 0 | none |
| v03 | 2 ms | initial_velocity / 2 | 30.000 | PASS | 0 | none |
| v03 | 2 ms | initial_velocity / 3 | 30.000 | PASS | 0 | none |
| v03 | 2 ms | push_x / 0 | 30.000 | PASS | 0 | none |
| v03 | 2 ms | push_y / 0 | 30.000 | PASS | 0 | none |

Initial-velocity cases draw root XY velocity uniformly in +/-0.015 m/s and root roll/pitch rate in +/-0.01 rad/s, using seeds 1, 2 and 3. Push cases apply 5 N to the torso in +X or +Y for 0.10 s beginning at 5.00 s (0.5 Ns impulse). These are small, explicitly scoped tests, not a broad robustness certificate.

### Nominal v0.3 measurements (2 ms commands)

| Metric | Measured value |
|---|---:|
| Maximum torso tilt after settling | 9.85300340162e-06 rad |
| Maximum COM speed after settling | 0.000501977765799 m/s |
| Maximum COM target displacement after settling | 0.00072078336787 m |
| Minimum geometric support margin after settling | 0.0989432099426 m |
| Minimum margin inside any joint range, all samples | 0.099980889391 rad |
| Maximum applied motor torque, all samples | 0.66708 Nm |
| Final pelvis height | 0.999281711535 m |
| Final total normal load | 321.768000046 N |
| External body/generalized stabilizing force | 0.0 / 0.0 |

Expected weight is 32.8 * 9.81 = 321.768 N, consistent with the settled measured load. Tiny floating-point time error (30 s recorded as 29.99999999999425) is handled by a timestep-scaled numerical tolerance, not by accepting fewer steps.

## Safety and integration validation

- 14 automated tests passed on both MuJoCo 3.17.0 and 3.2.7: existing model/runtime tests, compensation/model checks, original stop thresholds, exact final-substep snapshots, latch persistence/reset, contact loss, external stop, invalid residual/action, short standing envelope, Gymnasium contract/seeding, and promotion rejection. The environment checker emits only expected warnings for unbounded state observation ranges.
- `models/humanoid1.xml` is unchanged. The original v0.1/v0.2 controller implementations and raw-torque RL interface remain available.
- The Gymnasium checker, SB3 checker and one actual 256-policy-step CPU PPO update completed. The exploratory policies triggered envelope termination; no learned standing skill is validated.
- The zero-residual Gymnasium environment separately completed six held-out 30-second episodes (nominal; initial velocities seeds 101–103; X/Y pushes). Its episode traces and evaluation report were generated locally. A compact report is committed in `evidence/v0.3/rl_integration.json`.
- A CI matrix is supplied for Linux/macOS and both tested MuJoCo versions. CI results must be checked on the PR; local tests are Linux measurements.

## Evidence and reproduction

Environment: `Linux-6.18.55-x86_64-with-glibc2.39`, Python 3.12.14, NumPy 2.3.5, MuJoCo 3.17.0. The lower-version compatibility result uses MuJoCo 3.2.7. No MacBook Air M3 hardware was available for direct runtime validation. The implementation uses NumPy and MuJoCo on CPU and retains the macOS viewer path; optional PPO uses CPU Torch.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 python -m humanoid1.verification --suite --output results/v03_final
python scripts/export_evidence.py results/v03_final evidence/v0.3/main
```

Full 500 Hz CSVs contain all 16 joint positions, velocities and applied torques, plus diagnostics. They are regenerated under `results/` rather than committing roughly 160 MB of raw logs. Committed JSON retains every crossing event, worst excursion, acceptance result, source/model SHA256 and raw-log SHA256. Compact 10 Hz traces are for review only; acceptance uses the full-rate samples. `base_commit` identifies the starting commit; module hashes identify the tested working-tree source. Non-physics learning/report files may change without rerunning unchanged physics.

## Scope and next work

The requested unsupported standing milestone is met for this unchanged flat-floor model and the stated tests. This does not establish physical-robot stability, walking, large-disturbance recovery, uneven terrain, generalization to other morphologies or a decade-scale research breakthrough. Next engineering steps are constrained feasible-wrench optimization, meaningful model/terrain randomization, contact-mode planning, and skill-specific validation. The optional learning architecture prepares these experiments without relaxing limits or automatically activating a trained candidate.
