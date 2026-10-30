# Humanoid-1 v0.5 weight shifting and first system integration

## Result

The engineered controller passed six 30-second double-support weight-shifting cases: a fixed left/right sequence, three randomized command sequences, and two randomized sequences with seeded horizontal pushes. It also passed the nominal 30-second task on MuJoCo 3.2.7. The original robot model, motor/joint ranges and core safety thresholds are unchanged.

The first integrated runtime now plans a bounded motion request, executes actual MuJoCo control, records ground-truth observations and outcomes, persists five memory types in SQLite, and recalls a task in a new process with evidence-hash verification. This is an early integrated software slice toward the full TOC, not a complete humanoid or an advanced generative local brain.

## Engineered task and safety

Four minimum-jerk excursions shift the lateral COM target between center and +/-16–20 mm, then return home. Position, velocity and acceleration reference feedforward is added through a narrow COM-task hook; the standing default keeps its identical acceleration formula. Both feet remain loaded. No root actuator, weld, stabilizing force, modified friction or relaxed torque/joint limits is introduced.

The original standing envelope remains anchored to the initial home COM: <=30 mm displacement, <=0.10 m/s COM speed, <=0.10 rad torso tilt, pelvis height 0.95–1.05 m, >=1 N per foot, no non-foot support, no nominal joint-limit event, and >=20 mm geometric support margin after 0.10 s initial contact settling. A moving command does not move this safety boundary.

Task acceptance additionally requires >=30 s completion, <=3 mm RMS/8 mm peak XY tracking error, >=28 mm achieved lateral COM range, >=53% left loading on positive holds and <=47% on negative holds, <=2 mm final center error, and <=3 mm sole-center drift. Disturbance recovery requires <=1 mm XY tracking and <=3 mm/s velocity error continuously for 0.25 s, starting within 1.5 s after each pulse. Recovery values report the start of the qualifying dwell window, not its end.

## Matched baseline and PPO results

| Case | Engineered pass | PPO pass | Engineered RMS (mm) | PPO RMS (mm) | Engineered left/right hold loading |
|---|---|---|---:|---:|---|
| seed 0 / random=False / push=False | True | True | 0.090528 | 1.661249 | 54.840% / 45.160% |
| seed 101 / random=True / push=False | True | True | 0.087140 | 1.630926 | 54.786% / 45.572% |
| seed 102 / random=True / push=False | True | False | 0.079970 | 1.423862 | 54.028% / 45.678% |
| seed 103 / random=True / push=False | True | True | 0.076031 | 1.475744 | 54.339% / 46.026% |
| seed 201 / random=True / push=True | True | True | 0.136281 | 1.570610 | 54.831% / 45.874% |
| seed 202 / random=True / push=True | True | True | 0.153547 | 1.493206 | 54.208% / 45.701% |

The PPO candidate completed 16,384 policy steps (50 Hz, 500 Hz underlying control), using random command timing/amplitudes and randomized pushes, conservative initial exploration and a KL update guard. The mean RMS tracking error was **1.542600 mm**, versus **0.103916 mm** for the engineered baseline. It completed all six 30-second episodes without a core safety stop, but failed the load-transfer criterion in seed 102 and regressed in tracking, mechanical work and contact-slip metrics. **Promotion was rejected.** No learned policy is active in the runtime.

Promotion requires all six matched final cases, identical models/source/criteria/commands/disturbances, >=5% mean RMS improvement, <=2% per-case tracking regression, <=10% mechanical-work increase, and bounded slip regression. The old standing promotion rule is unchanged; this is a separate task-specific value gate.

## Measurements and interpretation

Mechanical work integrates the absolute product of applied joint torque and velocity. It is not electrical energy, battery life, actuator heating or motor-loss modeling. Contact slip integrates normal-load-weighted tangential contact-point speed on each foot; sole drift separately measures horizontal foot-body displacement from its initial location. A geometric support margin is not a ZMP/CoP or feasible-wrench certificate.

Commands genuinely differ by seed in the randomized cases. Push directions, amplitudes (4–8 N) and durations (0.10–0.15 s) also differ by seed. The nominal case is deterministic. These remain small, flat-ground, synthetic-robot tests, with no claim of physical walking or broad real-world robustness.

## Actual simulation video

`evidence/v0.5/weight_shift.mp4` renders the real MuJoCo motion and measured COM/load values at 25 fps (750 frames, 30 seconds). It is engineered control, not a learned-policy demonstration. The small 2 cm motion is made visible by measured overlays. The video run independently passed acceptance. Recording needs optional Pillow/FFmpeg; the baseline control needs neither.

## Reproduction

```bash
python -m humanoid1.weight_shift --suite --output results/v05/engineered
mjpython -m humanoid1.weight_shift --view
python -m humanoid1.runtime --state-dir results/my_humanoid demo
```

Optional learning experiment:

```bash
python -m pip install -r requirements-experiment.txt
python -m humanoid1.weight_shift_learning train --steps 16384 --seed 7
python -m humanoid1.weight_shift --suite --policy results/v05/ppo_candidate.zip --output results/v05/ppo
python -m humanoid1.weight_shift_learning compare --candidate results/v05/ppo/summary.json --baseline results/v05/engineered/summary.json
```

Use `evidence/v0.5/ppo_candidate.zip` in the evaluation command to run the exact rejected checkpoint. Training metadata records source/checkpoint hashes and actual actor parameter changes. Full 500 Hz CSV logs regenerate locally; committed traces are 10 Hz for review.

## Full system scope

[`FULL_SYSTEM_SCOPE.md`](FULL_SYSTEM_SCOPE.md) and `requirements/book_features.json` cover all 45 chapters, 8 parts, 6 appendices and 587 numbered topics in the canonical research-linked TOC. Each feature is linked to a capability, acceptance goal, status and evidence boundary. No chapter is marked complete. The first local runtime is documented in [`INTEGRATED_RUNTIME.md`](INTEGRATED_RUNTIME.md).

Walking/single support, articulated hands, visual/chemical perception, natural voice, advanced local reasoning, predictive world models, social interaction, full power/thermal design, live communications and optional NVIDIA backends remain incomplete. Consciousness, organ replacement, neural restoration and life extension require separate scientific evidence and cannot be established by this simulator.
