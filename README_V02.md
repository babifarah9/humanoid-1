# Humanoid-1 v0.2 — balance engineering package

This release retains the original v0.1 model, tests and CLI, while adding **experimental** gravity/bias compensation, torso-tilt and center-of-mass feedback, ground-force diagnostics, and reproducible baseline comparisons. It is not a trained RL policy and is not a safety-certified robot controller.

## Install

In Terminal from this extracted directory (Mac Apple Silicon):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tests/test_structure.py
PYTHONPATH=. python tests/test_runtime.py
python tests/test_v02_structure.py
```

## Run baseline then balance controller

```bash
python -m humanoid1.benchmark --mode baseline --seconds 30 --log results/v02_baseline.csv
python -m humanoid1.benchmark --mode balance --seconds 30 --log results/v02_balance.csv
```

The CLI prints `elapsed_s`, `stopped`, `reason`, `achieved_30s` and other metrics. The diagnostics CSV records COM position, center-of-mass relative to foot midpoint, torso orientation, contact count, estimated contact normal forces and torque saturation. A candidate that merely survives without maintaining attitude/contact is **not** proven stable.

If the v0.2 controller still falls, test 16 controller combinations plus baseline:

```bash
python -m humanoid1.benchmark --mode sweep --seconds 10
```

Full results are saved to `results/v02_sweep.json`. This evaluates sign conventions / feedback scaling; do not interpret the best result as physically validated or universally optimal. Do not disable the safety latch. The previous v0.1 baseline is preserved.

## Opening the MuJoCo model viewer on macOS

```bash
python -m mujoco.viewer --mjcf="$PWD/models/humanoid1.xml"
```

The stand-alone viewer **does not run the Python feedback controller**. Thus a model falling there does not evaluate either standing controller. On macOS `mjpython` is generally needed for passive viewer integration, not this blocking viewer command.

## Known limitations

- No MuJoCo package was available in the build environment; **physics performance and 30-second standing are not verified**. User should run and share benchmark output.
- This stage does not implement reinforcement learning. Train PPO only after reliable balance diagnostics and safety tests.
- Sagittal/lateral contact dynamics, feedforward limitations and force distribution need further tuning. COM error is measured against foot midpoint, not support polygon or ZMP.
- Contact forces sum positive contact normal components; they are approximate diagnostics, not calibrated ground-reaction sensors.
