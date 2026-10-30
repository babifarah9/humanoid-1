# PPO residual-standing experiment after Humanoid-1 v0.3

Date: 29 October 2026. Parent standing baseline: commit `812ac2f312e61de5a63c3c5949437ace8a716c4f`. This is an experimental learning follow-up, not a promoted v0.4 controller release.

## Result

Two actual CPU PPO training runs completed **16,384 policy steps each** (32,768 total), and actor parameters changed. Both candidates passed five 10-second validation episodes. The selected candidate (seed 1) then completed all six 30-second final evaluation episodes with no joint-limit events or safety stops.

**Promotion was rejected.** The learned residual produced a lower matched episodic return than zero residual in every final case, by approximately 0.000082–0.000086. The existing 0.000001 return tolerance was not relaxed. No skill registry was created, no trained policy was activated, and the engineered v0.3 controller remains the default. Completing these tests does not show that PPO acquired standing from scratch: the engineered whole-body controller remains active underneath the residual policy.

## What was trained

- A 69-input, 64/64-hidden-unit tanh actor with 16 normalized residual actions. SB3 PPO, five update epochs per 256-step rollout, batch size 64, default 0.0003 learning rate, CPU only, one Torch thread per run.
- Policy decisions every 20 ms; whole-body control, torque clipping and emergency checks every 2 ms. The maximum residual bound remains +/-2 Nm per actuator.
- Training scenarios are small random initial root XY velocities (+/-0.015 m/s) and roll/pitch rates (+/-0.01 rad/s). Initial exploration standard deviation is exp(-3) or exp(-4), rather than the earlier default of 1.0. The torque bound, physical model, control equations, criteria and stops are unchanged.
- Seeds 0 and 1 generated ten complete stable 30-second training episodes each, plus a final partial episode. These are training observations, not promotion evidence.

| Candidate | Initial log std | Policy steps | Training wall time (s) | Maximum actor output-weight change | Mean validation return |
|---|---:|---:|---:|---:|---:|
| seed 0 | -3 | 16384 | 226.009 | 0.055701520294 | 9.999385178260 |
| seed 1 | -4 | 16384 | 223.845 | 0.044310193509 | 9.999429317319 |

Training wall times are Linux measurements; they are not MacBook Air M3 training-time estimates.

## Selection and final evaluation

An experiment plan was written before training. Candidate selection used five 10-second cases: nominal seed 11, initial-velocity seeds 12 and 13, and fixed X/Y push cases with labels 14 and 15. Both candidates passed; seed 1 had the higher mean validation return. Selection was frozen before evaluating that checkpoint on the final suite.

Final initial-velocity seeds 101–103 were not used for training or selection. Nominal standing and the fixed 5 N / 0.1 s push patterns are deterministic; changing their seed labels does not create new disturbance realizations. Their final 30-second tests extend the validation duration, and must not be described as statistically independent unseen environments.

| Final scenario | Seed | Duration (s) | Engineered baseline return | PPO residual return | Difference | Outcome |
|---|---:|---:|---:|---:|---:|---|
| nominal | 101 | 30.000 | 29.998438498726 | 29.998353324250 | -0.000085174475 | Standing passes; promotion fails |
| initial_velocity | 101 | 30.000 | 29.998427594603 | 29.998343721901 | -0.000083872702 | Standing passes; promotion fails |
| initial_velocity | 102 | 30.000 | 29.998425124300 | 29.998339378088 | -0.000085746212 | Standing passes; promotion fails |
| initial_velocity | 103 | 30.000 | 29.998423271207 | 29.998338093394 | -0.000085177813 | Standing passes; promotion fails |
| push_x | 101 | 30.000 | 29.998420007581 | 29.998337762861 | -0.000082244720 | Standing passes; promotion fails |
| push_y | 101 | 30.000 | 29.998419455154 | 29.998333586634 | -0.000085868521 | Standing passes; promotion fails |

Across all final PPO episodes, maximum post-settling torso tilt was 0.00101441491341 rad; minimum geometric support margin was 0.0985144948814 m; maximum normalized residual action was 0.0443261787295, equivalent to 0.0886523574591 Nm. All six zero-residual reference episodes also passed. The original declared standing envelope and 0.10 s settling convention remain in force.

## Engineering changes

- Record checkpoint/model/source hashes, dependency versions, actual step counts, actor parameter changes, episode outcomes, progress CSVs and periodic checkpoints.
- Accumulate episode extrema at every physics substep, rather than reporting only final states. Keep the standing envelope shared with the original verifier.
- Provide a reproducible two-candidate runner with a declared selection rule, separate validation split, final evaluation and conditional promotion.
- Reject duplicate baseline records, non-finite durations/returns, validation-only evidence, baseline stop/joint events, criterion mismatches and reward regressions. These tighten evidence validation; they do not change robot limits or relax promotion requirements.
- Extend CI with CPU PPO training smoke checks and full final-checkpoint evaluations on Linux and macOS. Check the PR for actual CI status.

## Reproduction and evidence

Use Python 3.12 and `requirements-experiment.txt` for the recorded direct dependency versions. On Linux, install CPU Torch from its CPU wheel index before the remaining requirements; on macOS the regular Torch wheel supports the CPU path. No GPU or paid service is required.

```bash
python -m pip install -r requirements-experiment.txt
python scripts/run_ppo_experiment.py --output results/ppo_reproduction --promote
```

The runner registers a standing skill only if the original return/safety gate passes. It never activates a trained policy. Results can differ numerically across versions, processors and operating systems; compare measured outcomes rather than assuming identical hashes from retraining.

Run the exact selected checkpoint without retraining:

```bash
python -m humanoid1.learning evaluate --policy evidence/v0.4/selected_policy.zip --output results/selected_policy
```

Committed evidence in `evidence/v0.4/` includes both final checkpoints, training metadata/progress, validation reports, baseline/final reports, the selection and rejection decision, and checksums. `selected_policy.zip` duplicates the selected seed-1 checkpoint for a stable command path. It is a rejected promotion candidate, not the default controller.

`experience.zip` retains all final-evaluation actions, rewards and timestamps at 50 Hz, with observation checkpoints at 10 Hz. Its manifest includes original full-observation archive hashes, checkpoint indices and archive entry hashes. Full 50 Hz observations and further training checkpoints regenerate under `results/`; they are not all committed. The 500 Hz safety extrema are in the evaluation reports. PPO uses its on-policy rollout buffer; the archived episodes are not replayed into PPO.

## Interpretation

The original statement that current controllers fail after 1.6–2.4 seconds is historical. v0.3 already passed 30-second unsupported standing, including Linux/macOS ARM64 CI; the newly trained residual also completed the stated final standing tests. Evidence for autonomous acquisition of a useful new skill is still insufficient: this policy did not improve the declared objective over the engineered baseline. Further work should define a task with measurable additional value and an appropriate curriculum, then validate it on broader unseen conditions. More training alone is not evidence of improvement.
