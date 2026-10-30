# Humanoid-1 v0.7: repeated forward gait and controlled stopping

The 16-actuator, 32.8 kg reference body now executes six alternating steps in
one continuous 60-second MuJoCo episode, advances 0.219833 m nominally, and
returns to standing. This extends v0.6's two-step demonstration. It remains a
very slow straight flat-floor gait, with no turning or terrain adaptation.

## Measured outcomes

The recorded suite contains six full gait trials: nominal, three randomized
command programs (seeds 101–103), and two randomized programs with small
single-support pushes (seeds 201–202). All six pass; body progress ranges from
0.201146 to 0.246814 m. Separate matching-source seven- and eight-step program summaries also report passing results; their full-rate traces are not included in this publication.

Each accepted step must have at least 0.5 s of measured single support with
15 mm sole clearance, unloaded swing foot, loaded stance foot and adequate
support margin. Landing requires measured bilateral load, placement error at
most 5 mm, and at least 0.1 s of landing dwell. Final standing requires at least
2 s with low COM/foot velocity and torso tilt. Acceptance uses every 2 ms
physics sample; complete 10 Hz review traces accompany the reports.

Across the six full gait cases, minimum measured support margin is 48.48 mm
and maximum stance drift is 0.673 mm. No nominal joint-limit event, torque
saturation or safety stop occurs. Existing model, core safety, whole-body
controller and contact allocator are unchanged from v0.6. The stepping loop
gains task hooks so both motion programs share its physics-step checks.

## Controlled stop and runtime

`stop-walking` requests a balanced stop of a running walking episode. The plan
finishes a current swing/landing or the current smooth support transfer,
recenters, stands upright, and measures standing dwell. Scheduled requests at
5.0, 8.8 and 24.4 s pass with zero, one and three completed steps respectively.
Measured stop completion takes 10.85–11.50 s after these requests. This slow
planned response is separate from the existing emergency latch.

A completed controlled stop records `status=controlled_stop` and
`goal_completed=false`; it does not count as finishing the forward program.
Emergency stop retains priority at every physics substep. Persistent stop
requests are scoped to the task ID; a second process can observe an airborne
foot, request the stop, and later recall the same hash-checked task evidence.
Runtime commands still start fresh simulation episodes and use simulator
ground truth plus a deterministic planner.

## Reproduction

```bash
python -m pip install -r requirements-dev.txt
OPENBLAS_NUM_THREADS=1 python -m pytest -q
OPENBLAS_NUM_THREADS=1 python -m humanoid1.walking --suite --output results/v07
OPENBLAS_NUM_THREADS=1 python -m humanoid1.walking --steps 8 --output results/eight_steps
python scripts/verify_walking_runtime.py --output results/runtime_walk
python scripts/verify_walking_runtime.py --controlled-stop --output results/runtime_stop
# macOS viewer:
mjpython -m humanoid1.walking --view
# Optional actual recording, using Pillow and ffmpeg:
MUJOCO_GL=egl python scripts/record_walking.py
```

The nine-case suite was regenerated during this continuation because some
older full-rate CSVs were incomplete. All nine fresh simulation results pass.
The fresh nominal raw file was also truncated; its complete prior trace was
retained only after checking matching source/model fingerprints and exact
equality of every summary field except file path and wall time. The other eight
traces are from the fresh run. All nine exported traces pass sample-count and
terminal-time checks; provenance is in `publication_checks.json`. The recovered seven/eight-step summaries
and recording were retained after checking their complete controller/model
fingerprints against the source byte for byte. Original 75-test logs for MuJoCo
3.17.0/3.2.7 are labeled `recovered_`; fresh test and runtime results from the
continuation are separately recorded. The video is an actual 1,500-frame,
25 fps simulation with simultaneous foot close-up, loads, clearance and travel.

Evidence: [`evidence/v0.7`](../evidence/v0.7/). CI status is reported by the PR;
local checks do not establish a new macOS or MacBook Air M3 result.
Current publication verification passes 75 tests on MuJoCo 3.2.7 in 297.51 s.
The preceding continuation also recorded 71 tests with two
optional Gymnasium module skips on MuJoCo 3.17.0. Both real runtime runs
(full gait and cross-process airborne stop) pass with restart recall.

## Book scope and next milestone

Chapter 9's walking/running entry now has partial evidence for repeated gait.
All 45 chapters and 587 numbered entries remain tracked; no chapter is marked
complete. General navigation, human-speed walking, running, uneven terrain,
hands/touch, vision, voice and advanced local intelligence remain unfinished.
No learned controller is promoted and physical deployment is unverified.

The next locomotion milestone is command-driven longer gait, direction control
and recovery tests. It must retain measured per-step contact/clearance evidence
and final-standing criteria before expanding the executable runtime grammar.
