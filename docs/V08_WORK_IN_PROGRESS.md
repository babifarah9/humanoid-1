# v0.8 commanded walking — work in progress

This branch contains the initial distance-command implementation built on the
published v0.7 gait. It is not a verified v0.8 release.

`humanoid1.commanded_walking` constructs 6–12 alternating steps for distances
from 0.20 to 0.45 m. Its candidate directions are forward, backward, left and
right with a fixed body-facing orientation. This adds translation targets;
it does not implement yaw turning, obstacle navigation or hardware deployment.
The existing body, contact controller and safety limits are unchanged.

The implementation retains v0.7's measured per-step landing, clearance,
support-load, standing and controlled-stop checks. New acceptance checks cover
the final command endpoint, tracking perpendicular to the intentional balance
reference and torso yaw. Two extension hooks in `walking.py` select a gait plan
and measure progress in the requested direction.

Initial local checks pass for all four reference endpoints, position/velocity/
acceleration continuity, interrupted-plan standing endpoints, rejection of
short-episode success, emergency-stop priority and whitespace errors.

The 31 reference, validation and fail-closed tests pass. Eight initial physics
trials all stopped at the new 0.02 rad torso-yaw check, including the forward
case. They do not establish any successful new walking direction or distance.
The rejected trials and test log are preserved in `evidence/v0.8/development/`.
These are development diagnostics, not accepted release evidence.

Still pending: diagnosis against v0.7, successful full-horizon physics trials,
full regression tests, persistent runtime integration, accepted source-matched
evidence export and CI. No new direction is
advertised as a verified runtime skill until those checks pass.

Run a candidate simulation with:

```bash
OPENBLAS_NUM_THREADS=1 python -m humanoid1.commanded_walking --direction backward --distance .22
```

The full book feature scope remains incomplete; v0.7 is the latest published
walking evidence.
