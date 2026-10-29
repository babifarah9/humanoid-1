# Humanoid-1 v0.1 — Local simulation baseline

An **experimental**, fully local, minimal humanoid simulation and guarded joint-control foundation for *Humanoids*. **Not** an autonomous robot, validated standing policy, safety-certified controller, or trained RL system yet.

## Current content

- Original MJCF humanoid: free-floating pelvis, legs, feet, torso, head, arms; **16 actuated joints**; no fingers yet.
- Simple joint PD posture controller and experimental ankle-pitch feedback; intended as a *baseline to test*, not an established balance controller.
- Saturated torques, nominal position checks, pelvis height and extreme-velocity stop; when triggered, **simulation stepping halts**. This is not physical emergency-stop validation.
- CSV logs and modular observation/action environment for future Gymnasium/PPO training.
- Exploratory random gain search (**not** reinforcement learning).

## Setup (MacBook Air M3, macOS)

Python **3.11 or 3.12** recommended:

```bash
cd Humanoid1_v0_1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tests/test_structure.py
python tests/test_runtime.py
python -m humanoid1.cli --seconds 5
```

Interactive visualizer on macOS (MuJoCo viewer may require the `mjpython` launcher):

```bash
mjpython -m humanoid1.cli --seconds 15 --view
```

If `mjpython -m` is not supported by your installation, use `mjpython -c 'from humanoid1.cli import run; run(seconds=15, visualize=True)'` from the project root.

The run creates `results/stand.csv` and prints a summary including the actual simulated duration and whether safety stopped the episode. **Do not assume it stands successfully**; falling or immediate safety stops are meaningful results for v0.1.

## Project files

- `models/humanoid1.xml`: physics model, joints, mass assumptions, motors.
- `humanoid1/core.py`: MuJoCo model load, observation, PD controller, safety-stop latch.
- `humanoid1/cli.py`: baseline standing rollout, optional viewer, CSV logging.
- `humanoid1/rl_env.py`: lightweight reset/step interface (not a full Gymnasium wrapper).
- `humanoid1/optimize.py`: exploratory PD-gain trial runner; **not RL**.
- `tests/test_structure.py`: structural XML integrity, runnable without MuJoCo.
- `tests/test_runtime.py`: real MuJoCo import and stepping check; requires installation.

## Research rigor and development roadmap

**v0.1 validation status:** XML and Python structure checked in the authoring environment; MuJoCo itself could not be installed here due to blocked package network access. Runtime behavior has **not yet been verified**. No walking, learning, semantic memory, local LLM, or hands are implemented at this version.

1. Run macOS tests and inspect logs; fix dynamics or balance instability.
2. Add balance benchmarks and perturbation tests; validate baseline across multiple seeds.
3. Add Gymnasium-compatible wrapper and small local PPO model; compare against PD baselines.
4. Add experience storage and a policy evaluation/promotion gate.
5. Add locally hosted small language model, task planning, and persistent memories.
6. Expand to manipulation, touch, navigation, and integrated end-to-end tasks.

All mandatory components are free software; no remote inference or cloud billing. A physical robot will require hardware.

## Important safety boundary

This is a physics simulation research demo. Numerical checks are not formally verified safety guarantees. Learned policies must be tested in simulation and supervised on hardware; physical emergency-stop hardware and independent safety systems are necessary before physical experiments.

## References

- MuJoCo Python installation and viewer: https://mujoco.readthedocs.io/en/stable/python.html
- Gymnasium MuJoCo Humanoid baseline: https://gymnasium.farama.org/environments/mujoco/humanoid/
