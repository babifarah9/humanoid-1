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
python -m humanoid1.runtime --state-dir results/my_humanoid remember project Humanoid-1
python -m humanoid1.runtime --state-dir results/my_humanoid recall-memory project
python -m humanoid1.runtime --state-dir results/my_humanoid recall TASK_ID
python -m humanoid1.runtime --state-dir results/my_humanoid stop
python -m humanoid1.runtime --state-dir results/my_humanoid reset-stop
```

A stop persists across processes and blocks new motion. Executing loops also poll
that persisted operator stop at physics substeps. Reset is explicit and applies
to fresh simulation episodes; the core stop latch is never bypassed. Unsupported
requests such as walking, grasping or smelling fail before motion begins. No
unknown request is converted into a fabricated successful response.

## Implemented contracts

| Layer | Current implementation | Remaining scope |
|---|---|---|
| Local AI Brain | Deterministic bounded grammar and typed skill requests | Advanced local generative/multimodal reasoning and uncertainty-aware planning |
| Autonomous Learning | Actual PPO training, matched evaluation and rejection gates | Measured useful new skills, transfer, continual learning and retention |
| Persistent Memory | SQLite episodic, semantic, procedural, spatial and task records; restart recall; evidence hash verification | Semantic retrieval, consolidation, forgetting/privacy controls and long-term tests |
| Embodied Intelligence and World Model | Explicitly labeled physics-state observations and saved task outcomes | Visual/object understanding, learned dynamics and counterfactual predictions |
| Motion Control and Safety | Verified standing; bounded bilateral weight shift; core/substep/operator stop checks | Single support, walking, manipulation, broad fault recovery and hardware assurance |
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

See [the full TOC scope](FULL_SYSTEM_SCOPE.md) and
[`requirements/book_features.json`](../requirements/book_features.json). Every
chapter and all numbered topics remain visible, with no chapter declared complete.
