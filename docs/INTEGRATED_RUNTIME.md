# Humanoid-1 integrated local runtime

This is the first working integration slice toward the full book scope. It connects
bounded task planning, real MuJoCo motor control, observed robot state, persistent
memory and recall across process restart. It does not complete the humanoid.

## Run locally

The baseline needs only the existing NumPy/MuJoCo requirements and Python's SQLite.
No cloud account, model download or GPU service is required.

```bash
python -m humanoid1.runtime --state-dir results/my_humanoid status
python -m humanoid1.runtime --state-dir results/my_humanoid demo
```

The demo executes a real 30-second weight-shifting simulation, checks acceptance,
saves the task/evidence/observed state and launches a second Python process to
recall the same task and verify the evidence hash. It writes JSON/CSV evidence
and `memory.sqlite` under the selected state directory. Reuse that directory to
retain memory across subsequent runs; deleting it deletes that local memory.

For individual commands:

```bash
python -m humanoid1.runtime --state-dir results/my_humanoid execute "stand" --seconds 30
python -m humanoid1.runtime --state-dir results/my_humanoid execute "demonstrate weight shifting"
python -m humanoid1.runtime --state-dir results/my_humanoid execute "take two steps"
python -m humanoid1.runtime --state-dir results/my_humanoid execute "walk forward"
python -m humanoid1.runtime --state-dir results/my_humanoid stop-walking
python -m humanoid1.runtime --state-dir results/my_humanoid remember project Humanoid-1
python -m humanoid1.runtime --state-dir results/my_humanoid recall-memory project
python -m humanoid1.runtime --state-dir results/my_humanoid recall TASK_ID
python -m humanoid1.runtime --state-dir results/my_humanoid stop
python -m humanoid1.runtime --state-dir results/my_humanoid reset-stop
```

A stop persists across processes and blocks new motion. Executing loops also poll
that persisted operator stop at physics substeps. Reset is explicit and applies
to fresh simulation episodes; the core stop latch is never bypassed. Unsupported
requests such as unrestricted walking, grasping or smelling fail before motion begins. No
unknown request is converted into a fabricated successful response.

The v0.6 `take two steps` skill runs the bounded 30-second forward-advance
program: right foot, left foot, then standing recovery. It requires at least
30 seconds and uses measured sole clearance, load, placement and support dwell.
It is separate from continuous walking; `walk` remains unsupported. Every motion
request starts a fresh simulation episode, so multiple requests do not constitute
continuous physical navigation. Its actual foot observations are saved as spatial
memory and its evidence is hash-checked on recall.

v0.7 adds the exact requests `walk forward`, `walk forward then stop`, and
`demonstrate walking`. They select six slow straight flat-floor steps followed
by standing, with a default 60-second horizon. Turning, backward motion and
arbitrary navigation requests are rejected. `stop-walking` requests a balanced
stop of currently running walking episodes; `--task-id` targets one episode.
The request persists across CLI processes and cannot affect a later episode.
The current swing is completed before recentering; this deliberate stop can
take several simulated seconds and is not an emergency stop.

Emergency `stop` retains priority and never gets reset by `stop-walking`.
Controlled stopping has its own measured latency bound and standing dwell. The
task outcome is `controlled_stop` with `goal_completed=false`, distinct from
successful completion of all six steps. Progress is published at phase changes
and about 5 Hz as labeled MuJoCo ground truth, allowing another process to
observe an actually airborne foot before issuing the request.

## Implemented contracts

| Layer | Current implementation | Remaining scope |
|---|---|---|
| Local AI Brain | Deterministic bounded grammar and typed skill requests | Advanced local generative/multimodal reasoning and uncertainty-aware planning |
| Autonomous Learning | Actual PPO training, matched evaluation and rejection gates | Measured useful new skills, transfer, continual learning and retention |
| Persistent Memory | SQLite episodic, semantic, procedural, spatial and task records; restart recall; evidence hash verification | Semantic retrieval, consolidation, forgetting/privacy controls and long-term tests |
| Embodied Intelligence and World Model | Explicitly labeled physics-state observations and saved task outcomes | Visual/object understanding, learned dynamics and counterfactual predictions |
| Motion Control and Safety | Standing, weight shifting, two-step advance, slow six-step straight gait and balanced stopping; independent emergency checks | General/dynamic walking, turning, terrain, manipulation, broad fault recovery and hardware assurance |
| Physical Body | Existing 16-actuator, 32.8 kg MuJoCo baseline | Hands, facial/eye actuation, calibrated sensors, power/thermal systems and physical construction |

The current planner selects one supported motion skill per request. Recording,
observation and recall are managed as part of the runtime lifecycle. This is
not general natural-language task decomposition. A state snapshot is ground
truth from the simulator, not camera perception or a learned world model.

## Verification

`tests/test_full_runtime.py` verifies source coverage, typed persistence across a
real process restart, stop persistence, unsupported-request rejection and tamper
detection. The recorded full demo is separate from these short software tests.
`tests/test_weight_shift.py` checks reference smoothness/bounds and operator
cancellation. The full benchmark measures motor behavior and ground forces.
`tests/test_stepping.py` adds a real full-horizon motion regression, unilateral
contact feasibility checks, reference boundaries and failure tests. The standing
and stepping skills are both checked for cancellation from another process.

See [the full TOC scope](FULL_SYSTEM_SCOPE.md) and
[`requirements/book_features.json`](../requirements/book_features.json). Every
chapter and all numbered topics remain visible, with no chapter declared complete.
