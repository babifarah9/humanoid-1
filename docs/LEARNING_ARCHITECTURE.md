# Local learning architecture after standing stabilization

## Implemented boundary

The engineered controller provides the standing baseline. A Gymnasium policy
observes the generalized state, COM error, measured foot loads, contact ages,
previous residual action, and elapsed episode fraction. It requests 16 normalized
actions at 50 Hz; these are clipped to [-1, 1] and converted to at most +/-2 Nm per
actuator. The whole-body controller recomputes at 500 Hz beneath that policy.
The unchanged motor limits clip the combined torque. Every physics substep passes
through the same core emergency latch; policy evaluation also enforces the stricter
standing envelope. A terminal safety stop freezes simulation rather than resetting
silently or continuing with a fallen robot.

`ResidualStandEnv` supports seeded reset, finite float32 observations, bounded Box
actions, separate safety termination/time-limit truncation, and strict episode reset.
The Gymnasium checker and SB3 checker were run. The legacy raw-torque `StandEnv`
remains available, but the residual wrapper is the intended PPO path.

## Local candidate loop

1. **Collect:** PPO receives on-policy transitions from the residual environment.
   Initial-velocity perturbations are the first simple curriculum; the nominal
   reference is captured before perturbing the state. No LLM or remote inference
   participates in motor control.
2. **Optimize:** SB3 PPO uses a small 64/64 MLP on `device='cpu'`, one Torch thread,
   256-step rollouts, and explicit seeds. The default script requests only 10,000
   policy steps. This is a starting experiment, not a claim that this budget is
   sufficient to acquire a skill or fit a particular MacBook wall-time budget.
3. **Checkpoint:** Save a candidate PPO ZIP plus seed, actual step count and
   `promoted: false`. The policy is never installed into the baseline controller.
4. **Evaluate:** Deterministic held-out seeds 101–103, quiet standing, initial
   velocity, and separate small X/Y pushes. Save every policy-step observation,
   action and reward to compressed NPZ, plus per-episode outcome metadata.
5. **Gate:** All six episodes must finish >=30 s with no safety termination,
   nominal joint event or standing-envelope violation. Each matched return must
   be at least the zero-residual baseline return, allowing 1e-6 numerical tolerance.
   Model, controller source, criteria, configuration and policy hashes bind the
   evidence to its candidate. This is a conservative standing gate, not a proof
   of improved capability or a security boundary against forged reports.
6. **Register:** An explicit `promote` invocation appends a standing-skill record
   to a local JSON registry. Runtime activation is still explicit. The registry
   has no authority to remove safety limits, alter the MJCF or deploy on hardware.

PPO is on-policy: its rollout buffer is not a general experience-replay learner.
Archived NPZ episodes are for reproducibility and later imitation/analysis; the
script does not feed old rollouts back into PPO.

## Evidence and remaining work

A CPU smoke run completed one 256-timestep PPO update using SB3 2.9.2, Gymnasium
1.4.0 and Torch 2.15.0+cpu. Exploratory residuals triggered the standing-envelope
termination, which is expected to reject unsafe candidates. This demonstrates
integration only. No trained standing, walking, manipulation or autonomous skill
acquisition is claimed. The zero-residual environment is evaluated independently.

Before claiming autonomous skill acquisition, implement and validate a task
curriculum, explicit task success metrics, larger held-out perturbations, bounded
model/domain randomization, retention/regression suites, and rollback to a known
controller. Walking will need a contact-mode planner, swing-foot tasks, feasible
contact-wrench constraints and new validation; reusing a double-support standing
reward does not establish walking. Memory, world models and local high-level task
planning can choose among approved skills later; they must not issue unrestricted
torques or bypass the safety supervisor.

This preserves Humanoid-1's six-layer direction: Local AI Brain; Autonomous Learning;
Persistent Memory; Embodied Intelligence/World Model; Motion Control/Safety Supervisor;
and Physical Body. v0.3 implements a narrow simulated part of the last two, plus
interfaces for future learning and persistent evaluation records.
