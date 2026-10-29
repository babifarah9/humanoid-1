# Humanoid-1 local reference system

The target is the **complete integrated humanoid described by the book TOC**.
The current implementation is partial. The source-aligned [full scope](docs/FULL_SYSTEM_SCOPE.md)
and [machine-readable register](requirements/book_features.json) track all 45 chapters
and 587 numbered topics, with no chapter falsely marked complete.

**v0.6 adds a measured two-step forward advance:** each foot lifts, moves about
40 mm forward, lands, and the robot returns to standing in a 30-second episode.
The controller explicitly changes support feet and allocates bounded contact
wrenches. This is a slow flat-floor skill, not continuous walking. See the
[v0.6 report](docs/V06_REPORT.md), [evidence](evidence/v0.6/), and
[actual simulation video with a foot close-up](evidence/v0.6/two_steps.mp4).

```bash
python -m humanoid1.stepping --suite --output results/v06
python -m humanoid1.runtime --state-dir results/my_humanoid execute "take two steps"
# macOS passive viewer:
mjpython -m humanoid1.stepping --view
```

The new [integrated local runtime](docs/INTEGRATED_RUNTIME.md) connects bounded
planning, verified motion, observed state and persistent SQLite memory. Try:

```bash
python -m humanoid1.runtime --state-dir results/my_humanoid demo
```

This executes a real 30-second weight-shifting task and verifies recall from a
second process. It is a small integration slice; advanced local reasoning,
walking, hands, multimodal perception, natural voice and other TOC capabilities
remain under development. The [v0.5 measured report](docs/V05_REPORT.md) includes
six passed engineered weight-shift cases and an actual simulation video. The PPO
weight-shift candidate was rejected because it worsened the measured task.

Humanoid-1 is a free, local MuJoCo research project for the *Humanoids* book.
The unchanged model has **16 actuated joints, 32.8 kg total mass, and a free-floating root**.
v0.3 adds contact-aware whole-body inverse dynamics, precise joint-limit diagnostics,
a physics-step safety latch, and optional bounded-residual PPO interfaces.

**Measured result:** v0.3 completed seven 30-second standing trials in MuJoCo 3.17.0,
including nominal operation at 500 Hz and 50 Hz, three seeded initial-velocity
perturbations, and separate 5 N / 0.1 s horizontal pushes. No joint-limit events
or safety stops occurred. Nominal standing also passed on MuJoCo 3.2.7.
See [the engineering report](docs/V03_REPORT.md) and [committed evidence](evidence/v0.3/).
These are simulations on Linux x86_64, not physical-robot or MacBook performance measurements.

## Run locally (MacBook Air M3 or other CPU)

Use Python 3.11 or 3.12 in the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m humanoid1.verification --mode v03 --seconds 30
```

The default controller recomputes torques every **2 ms**, with no GPU or cloud service.
The output includes `stable`, `achieved_30s`, actual simulation duration, all joint-limit
crossings and worst excursions, safety-stop snapshots, and declared standing criteria.
CSV records are sampled at every physics step, with all joint positions, velocities
and applied torques at the same simulation timestamp. Generated logs live under `results/`.

To compare all versions and timing/disturbance cases:

```bash
python -m humanoid1.verification --suite --output results/v03
```

To see the Python controller on macOS:

```bash
mjpython -m humanoid1.verification --mode v03 --seconds 30 --view
```

If your `mjpython` launcher does not support `-m`, use
`mjpython -c 'from humanoid1.verification import main; main()' --mode v03 --seconds 30 --view`.
The passive viewer syncs at 50 Hz but is not paced to wall time; simulation may run
faster or slower than playback time. The standalone MuJoCo model viewer does not run
this feedback controller. For machine-readable verification, prefer headless runs.

## Safety and acceptance

Original torque ranges, joint ranges, 0.07 rad joint-stop tolerance, pelvis minimum
0.60 m, and velocity-norm limit 90 are preserved. The latch now also checks the
**last substep** of a frame batch, saves the first failure, and blocks all subsequent
stepping until explicit reset. The whole-body controller adds bilateral-contact,
non-foot-support, torso-tilt, invalid-residual and numerical-solve stops.

A successful benchmark must complete the requested duration without a safety stop,
without **any nominal joint-range crossing**, and satisfy the standing envelope at
every 2 ms sample after a declared 0.10 s contact-settling window. The envelope covers
bilateral loaded feet, no other body on the floor, torso tilt, pelvis height, COM
speed/error, and geometric support margin. The startup window is included in the
30 s elapsed simulation and is exempt only from the standing-envelope evaluation,
never from the existing emergency checks or joint-event recording.

This is a local flat-ground standing controller. Wrench allocation is a weighted
least-squares model, not a friction-cone/center-of-pressure constrained QP. Simulation
checks are not a formal stability proof or a physical emergency-stop system.
The separate v0.6 stepping controller adds a small contact-wrench QP with linear
friction, CoP and torsion bounds. It is not a full torque/acceleration-constrained
whole-body QP or a hardware contact-feasibility guarantee. Existing standing
checks remain unchanged; stepping uses an explicit contact schedule and its own
declared envelope, with the same original core safety limits.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
# Optional Gymnasium environment tests:
python -m pip install 'gymnasium>=1.0,<2'
python -m pytest -q
```

The original v0.1 and v0.2 controller formulas and MJCF are retained. Legacy commands
remain available (`humanoid1.cli`, `humanoid1.benchmark`); the stricter success criterion
is implemented in `humanoid1.verification`, not the legacy survival-only benchmark.

## PPO and autonomous skill preparation

The standing result is an engineered controller, not a trained policy. Optional
CPU PPO training, seeded evaluation, compressed experience traces, and a
hash-bound standing-skill promotion gate are available separately:

```bash
python -m pip install -r requirements-rl.txt
python -m humanoid1.learning train --timesteps 10000 --output results/learning/candidate
python -m humanoid1.learning evaluate --output results/learning/baseline
python -m humanoid1.learning evaluate --policy results/learning/candidate.zip --output results/learning/candidate_eval
python -m humanoid1.learning promote --policy results/learning/candidate.zip --candidate-report results/learning/candidate_eval/evaluation.json --baseline-report results/learning/baseline/evaluation.json
```

Promotion rejects candidates that fail held-out 30-second episodes, exceed any
standing envelope, regress in reward, or mismatch model/controller/configuration
or policy hashes. Training does **not** automatically activate or promote a policy.
A 256-timestep CPU smoke run verified the training integration; its exploratory
policies failed the standing envelope and establish no learned skill.
See [the learning architecture](docs/LEARNING_ARCHITECTURE.md).

## Files and references

- `humanoid1/core.py`: model/reset, torque clipping, joint events, emergency latch.
- `humanoid1/whole_body.py`: contact measurements, COM/torso/posture tasks and inverse dynamics.
- `humanoid1/verification.py`: baselines, timing ablations, acceptance criteria and evidence.
- `humanoid1/ppo_env.py`: optional Gymnasium residual control and substep checks.
- `humanoid1/learning.py`: local PPO training/evaluation and explicit promotion.
- `humanoid1/rl_env.py`: original raw-torque interface retained for compatibility.
- [MuJoCo dynamics and conventions](https://mujoco.readthedocs.io/en/stable/computation/index.html).
- [MuJoCo Python/viewer](https://mujoco.readthedocs.io/en/stable/python.html).
- [Gymnasium environment contract](https://gymnasium.farama.org/api/env/).
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

## Recorded PPO experiment

The [PPO follow-up report](docs/PPO_EXPERIMENT.md) records two completed 16,384-step
CPU training runs. The selected residual policy passed all six final 30-second
standing evaluations, but its return was lower than the engineered baseline in
every case. **Promotion was rejected; v0.3 remains the default.** This demonstrates
a working training/evaluation/rejection loop, not autonomous acquisition of a
useful new skill or standing learned from scratch.

Both checkpoints, progress records and experience archives are in
[`evidence/v0.4/`](evidence/v0.4/). A fresh experiment is reproducible with:

```bash
python -m pip install -r requirements-experiment.txt
python scripts/run_ppo_experiment.py --output results/ppo_reproduction --promote
```

Training now starts with conservative exploration (`--log-std-init=-3` by default);
normalized action and motor torque bounds are unchanged. The existing exact
checkpoint can be evaluated with:

```bash
python -m humanoid1.learning evaluate --policy evidence/v0.4/selected_policy.zip --output results/selected_policy
```
