"""Measured two-step forward advance on the unchanged 16-actuator model.

CPU/NumPy/MuJoCo only. This is a bounded, slow, flat-floor skill, not continuous
walking. Support changes are explicit and checked against actual foot contacts.
"""
from dataclasses import dataclass, asdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import mujoco
from .core import Humanoid, MODEL
from .contact_control import ContactController
from .verification import CRITERIA, support_margin


STEP_CRITERIA = dict(min_duration_s=30., max_tracking_com_m=.03,
    max_tracking_foot_m=.015, max_torso_tilt_rad=.10, max_com_speed_m_s=.10,
    min_pelvis_z_m=.95, max_pelvis_z_m=1.05, min_support_margin_m=.02,
    max_stance_drift_m=.003, max_final_placement_error_m=.005,
    min_clearance_m=.015, max_swing_load_n=1., min_single_support_s=.5,
    min_stance_load_fraction=.9, max_final_com_error_m=.003,
    max_landing_vertical_speed_m_s=.08)


def interpolate(t, start, end, a, b):
    """Position, velocity and acceleration, with zero endpoint derivatives."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if t <= start:
        return a.copy(), np.zeros_like(a), np.zeros_like(a)
    if t >= end:
        return b.copy(), np.zeros_like(b), np.zeros_like(b)
    u = (t - start) / (end - start); duration = end - start
    delta = b - a
    return (a + delta * (10*u**3 - 15*u**4 + 6*u**5),
            delta * (30*u**2 - 60*u**3 + 30*u**4) / duration,
            delta * (60*u - 180*u**2 + 120*u**3) / duration**2)


@dataclass(frozen=True)
class StepProgram:
    seed: int = 0
    random_commands: bool = False
    disturbed: bool = False

    def specification(self):
        rng = np.random.default_rng(self.seed)
        distance = float(rng.uniform(.035, .045)) if self.random_commands else .04
        shift = float(rng.uniform(.095, .105)) if self.random_commands else .10
        clearance = float(rng.uniform(.028, .035)) if self.random_commands else .03
        pulses = []
        if self.disturbed:
            for start in (7.3, 16.3):
                angle = rng.uniform(-np.pi, np.pi); magnitude = rng.uniform(2., 4.)
                pulses.append(dict(start_s=start, end_s=start+.10,
                    force_n=[float(magnitude*np.cos(angle)), float(magnitude*np.sin(angle)), 0.]))
        return dict(step_distance_m=distance, lateral_shift_m=shift,
                    commanded_clearance_m=clearance, pulses=pulses)


def reference(t, specification):
    distance, shift, height = (specification[k] for k in
        ('step_distance_m', 'lateral_shift_m', 'commanded_clearance_m'))
    phases = [(1., 3., [0., 0., 0.], [0., 0., -.03]),
        (3., 6., [0., 0., -.03], [.5*distance, shift, -.03]),
        (9., 12., [.5*distance, shift, -.03], [.5*distance, 0., -.03]),
        (12., 15., [.5*distance, 0., -.03], [distance, -shift, -.03]),
        (18., 21., [distance, -shift, -.03], [distance, 0., -.03]),
        (22., 25., [distance, 0., -.03], [distance, 0., 0.])]
    com = np.zeros(3); velocity = np.zeros(3); acceleration = np.zeros(3)
    for start, end, a, b in phases:
        if t >= start:
            com, velocity, acceleration = interpolate(t, start, end, a, b)
    support = (0,) if 6. <= t < 9. else ((1,) if 15. <= t < 18. else (0, 1))
    feet = np.zeros((2, 3)); foot_velocity = np.zeros_like(feet); foot_acceleration = np.zeros_like(feet)
    for i, start in ((1, 6.), (0, 15.)):
        if t >= start:
            feet[i], foot_velocity[i], foot_acceleration[i] = interpolate(
                t, start, start+1., [0., 0., 0.], [distance, 0., height])
        if t >= start+2.:
            feet[i], foot_velocity[i], foot_acceleration[i] = interpolate(
                t, start+2., start+3., [distance, 0., height], [distance, 0., 0.])
    return dict(com=com, com_velocity=velocity, com_acceleration=acceleration,
                feet=feet, foot_velocity=foot_velocity, foot_acceleration=foot_acceleration,
                support=support)


def sole_clearance(robot, geom):
    """Lowest transformed sole corner, not just ankle/body height."""
    m, d = robot.model, robot.data
    rotation = d.geom_xmat[geom].reshape(3, 3)
    return float(min((d.geom_xpos[geom] + rotation @ (m.geom_size[geom] * [x, y, -1.]))[2]
                     for x, y in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.))))


class StepTask:
    max_seconds = 60.

    def __init__(self, program=None, seconds=30., log=None, stop_check=None):
        if not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or not .1 < seconds <= self.max_seconds:
            raise ValueError(f'motion duration must be in (0.1, {self.max_seconds:g}] seconds')
        self.program = program or StepProgram()
        if not isinstance(self.program.seed, int) or isinstance(self.program.seed, bool):
            raise ValueError('seed must be an integer')
        self.specification = self.program.specification()
        self.seconds = float(seconds); self.stop_check = stop_check
        self.log = Path(log) if log else None
        self.r = Humanoid(); self.c = ContactController(self.r)
        self.failed = set(); self.rows = []; self.done = False; self.last = {}
        self.steps = 0; self.max_requested = 0.; self.max_applied = 0.; self.max_saturated = 0
        self.external_max = 0.; self.generalized_external_max = 0.; self.work = 0.
        self.single_support_dwell = np.zeros(2); self.single_support_run = np.zeros(2)
        self.max_clearance = np.zeros(2); self.max_landing_speed = np.zeros(2)
        self.stance_anchor = self.c.home_feet.copy(); self.previous_support = (0, 1)
        self.wall_start = time.perf_counter()

    def motion_reference(self, t):
        return reference(t, self.specification)

    def before_substep(self, t):
        """Optional task command handling; default stepping behavior is unchanged."""

    def landing_phase(self, t, foot):
        return t >= (8. if foot == 1 else 17.)

    def after_sample(self, t, reference, diagnostics, feet, loads, clearance, margin, row):
        """Optional task-specific measured outcomes; no default side effects."""

    def step(self, frames=10):
        if self.done:
            raise RuntimeError('task has finished; create a new task to reset')
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
            raise ValueError('frames must be a positive integer')
        r, c = self.r, self.c; dt = float(r.model.opt.timestep)
        for _ in range(frames):
            if self.stop_check is not None and self.stop_check():
                r.emergency_stop('operator runtime stop')
            if r.stopped or r.data.time >= self.seconds - 1e-10:
                break
            t = self.steps * dt
            self.before_substep(t)
            ref = self.motion_reference(t)
            for i in ref['support']:
                if i not in self.previous_support:
                    self.stance_anchor[i] = r.data.xpos[c.feet[i]].copy()
            self.previous_support = ref['support']
            c.target_com = c.home_com + ref['com']
            c.com_velocity = ref['com_velocity']; c.com_acceleration = ref['com_acceleration']
            c.foot_positions = c.home_feet + ref['feet']
            c.foot_velocity = ref['foot_velocity']; c.foot_acceleration = ref['foot_acceleration']
            c.planned_support = ref['support']
            r.data.xfrc_applied[:] = 0.
            for pulse in self.specification['pulses']:
                if pulse['start_s'] <= t < pulse['end_s']:
                    r.data.xfrc_applied[c.torso, :3] = pulse['force_n']
            self.external_max = max(self.external_max, float(np.max(np.abs(r.data.xfrc_applied))))
            self.generalized_external_max = max(self.generalized_external_max, float(np.max(np.abs(r.data.qfrc_applied))))
            torque = c.act()
            self.max_requested = max(self.max_requested, c.last.get('max_requested_torque_nm', 0.))
            self.max_saturated = max(self.max_saturated, c.last.get('n_saturated', 0))
            if r.stopped:
                break
            # This is the sole state integrator: no qpos teleport, root wrench,
            # equality attachment or kinematic foot movement is used.
            r.step(torque, frames=1); self.steps += 1
            applied = r.data.qfrc_actuator[r.vadr].copy()
            self.max_applied = max(self.max_applied, float(np.max(abs(applied))))
            self.work += dt * float(np.sum(np.abs(applied * r.data.qvel[r.vadr])))
            di = c.diagnostics(); now = self.motion_reference(self.steps * dt)
            com = r.data.subtree_com[c.pelvis]
            error = float(np.linalg.norm(com - c.home_com - now['com']))
            feet = np.asarray([r.data.xpos[b].copy() for b in c.feet])
            clearance = np.array([sole_clearance(r, g) for g in c.foot_geoms])
            foot_error = float(np.max(np.linalg.norm(feet - c.home_feet - now['feet'], axis=1)))
            loads = np.array([di['left_normal_n'], di['right_normal_n']])
            margin = support_margin(r, com[:2], loads >= 1.)
            q = r.data.qpos[r.qadr]
            joint_margin = float(np.min(np.minimum(q-r.limits[:, 0], r.limits[:, 1]-q)))
            drift = float(max(np.linalg.norm((feet[i]-self.stance_anchor[i])[:2]) for i in ref['support']))
            for i in range(2):
                self.max_clearance[i] = max(self.max_clearance[i], clearance[i])
                eligible = (i not in ref['support'] and clearance[i] >= STEP_CRITERIA['min_clearance_m']
                    and loads[i] < STEP_CRITERIA['max_swing_load_n']
                    and loads[1-i] >= STEP_CRITERIA['min_stance_load_fraction']*32.8*9.81
                    and margin >= STEP_CRITERIA['min_support_margin_m'])
                self.single_support_run[i] = self.single_support_run[i]+dt if eligible else 0.
                self.single_support_dwell[i] = max(self.single_support_dwell[i], self.single_support_run[i])
                if i not in ref['support'] and self.landing_phase(t, i) and loads[i] >= 1.:
                    jac = np.zeros((3, r.model.nv))
                    mujoco.mj_jacBody(r.model, r.data, jac, None, c.feet[i])
                    self.max_landing_speed[i] = max(self.max_landing_speed[i], abs(float((jac @ r.data.qvel)[2])))
            if r.data.time >= CRITERIA['settle_s']:
                checks = dict(tilt=di['torso_tilt_rad'] <= STEP_CRITERIA['max_torso_tilt_rad'],
                    com_speed=di['com_speed_m_s'] <= STEP_CRITERIA['max_com_speed_m_s'],
                    tracking_com=error <= STEP_CRITERIA['max_tracking_com_m'],
                    tracking_foot=foot_error <= STEP_CRITERIA['max_tracking_foot_m'],
                    support_margin=margin >= STEP_CRITERIA['min_support_margin_m'],
                    pelvis_height=STEP_CRITERIA['min_pelvis_z_m'] <= di['pelvis_z'] <= STEP_CRITERIA['max_pelvis_z_m'],
                    stance_drift=drift <= STEP_CRITERIA['max_stance_drift_m'],
                    no_nonfoot_support=di['nonfoot_floor_contacts'] == 0,
                    joint_ranges=joint_margin >= 0.)
                self.failed.update(k for k, passed in checks.items() if not passed)
            if r.limit_events:
                self.failed.add('nominal_joint_limit_event')
            row = dict(di, planned_support='left' if ref['support'] == (0,) else ('right' if ref['support'] == (1,) else 'both'),
                reference_com_x_m=float(c.home_com[0]+now['com'][0]), reference_com_y_m=float(c.home_com[1]+now['com'][1]),
                reference_com_z_m=float(c.home_com[2]+now['com'][2]), tracking_com_m=error,
                tracking_foot_m=foot_error, support_margin_m=margin, joint_margin_rad=joint_margin,
                stance_drift_m=drift, left_sole_clearance_m=float(clearance[0]), right_sole_clearance_m=float(clearance[1]),
                left_foot_x_m=float(feet[0, 0]), right_foot_x_m=float(feet[1, 0]),
                max_applied_torque_nm=float(np.max(abs(applied))), absolute_mechanical_work_j=self.work,
                root_balance_residual=c.last.get('root_residual', 0.),
                predicted_wrench_bound_margin=c.last.get('minimum_inequality_margin', 0.),
                stopped=r.stopped, reason=r.stop_reason or '')
            for i, name in enumerate(r.joint_names):
                row[name+'_q_rad'] = float(q[i])
                row[name+'_v_rad_s'] = float(r.data.qvel[r.vadr[i]])
                row[name+'_torque_nm'] = float(applied[i])
            self.after_sample(t, ref, di, feet, loads, clearance, margin, row)
            self.rows.append(row); self.last = row
            if self.failed:
                r.emergency_stop('stepping envelope: ' + ', '.join(sorted(self.failed)))
        self.done = bool(r.stopped or r.data.time >= self.seconds-1e-10)
        return self.last.copy()

    def summary(self):
        r, c = self.r, self.c
        distance = self.specification['step_distance_m']
        final_feet = np.asarray([r.data.xpos[b].copy() for b in c.feet])
        expected_feet = c.home_feet + [distance, 0., 0.]
        placements = np.linalg.norm((final_feet-expected_feet)[:, :2], axis=1)
        displacement = final_feet[:, 0] - c.home_feet[:, 0]
        final_error = float(np.linalg.norm(r.data.subtree_com[c.pelvis]-c.home_com-[distance, 0., 0.]))
        checks = dict(duration=self.seconds >= 30. and r.data.time >= self.seconds-1e-8,
            safety=self.done and not r.stopped and not self.failed and not r.limit_events,
            left_single_support=self.single_support_dwell[1] >= STEP_CRITERIA['min_single_support_s'],
            right_single_support=self.single_support_dwell[0] >= STEP_CRITERIA['min_single_support_s'],
            both_foot_clearances=bool(np.all(self.max_clearance >= STEP_CRITERIA['min_clearance_m'])),
            forward_placements=bool(np.all(placements <= STEP_CRITERIA['max_final_placement_error_m'])),
            forward_displacement=bool(np.all(displacement >= distance-STEP_CRITERIA['max_final_placement_error_m'])),
            return_to_standing=final_error <= STEP_CRITERIA['max_final_com_error_m'],
            bilateral_final_load=bool(np.all(c.contacts()[0] >= 1.)),
            landing_speed=bool(np.all(self.max_landing_speed <= STEP_CRITERIA['max_landing_vertical_speed_m_s'])),
            no_torque_saturation=self.max_saturated == 0,
            no_generalized_external_force=self.generalized_external_max == 0.,
            no_constraints=int(r.model.neq) == 0)
        if self.log and self.rows:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0]), lineterminator='\n')
                writer.writeheader(); writer.writerows(self.rows)
        extrema = {}
        for key, operation in (('torso_tilt_rad', max), ('com_speed_m_s', max), ('tracking_com_m', max),
                               ('tracking_foot_m', max), ('support_margin_m', min), ('joint_margin_rad', min),
                               ('stance_drift_m', max), ('root_balance_residual', max), ('predicted_wrench_bound_margin', min)):
            extrema[key] = operation((row[key] for row in self.rows), default=None)
        return dict(skill='two_step_advance', continuous_walking_verified=False, hardware_verified=False,
            program=asdict(self.program), specification=self.specification,
            criteria=STEP_CRITERIA, acceptance={k:bool(v) for k,v in checks.items()},
            passed=bool(all(checks.values())), seconds_requested=self.seconds, elapsed_s=float(r.data.time),
            stopped=r.stopped, reason=r.stop_reason, failed_criteria=sorted(self.failed),
            foot_order=['left', 'right'], foot_displacements_x_m=displacement.tolist(),
            foot_placement_errors_xy_m=placements.tolist(), max_sole_clearance_m=self.max_clearance.tolist(),
            single_support_dwell_by_swing_foot_s=self.single_support_dwell.tolist(),
            max_landing_vertical_speed_m_s=self.max_landing_speed.tolist(), final_com_error_m=final_error,
            extrema=extrema, max_requested_torque_nm=self.max_requested, max_applied_torque_nm=self.max_applied,
            max_saturated_actuators=int(self.max_saturated), absolute_mechanical_work_j=self.work,
            max_external_force_component_n=self.external_max, max_generalized_external_force=self.generalized_external_max,
            equality_constraints=int(r.model.neq), joint_limit_events=r.limit_events, stop_snapshot=r.stop_snapshot,
            samples=len(self.rows), wall_s=time.perf_counter()-self.wall_start, log=str(self.log) if self.log else None)


def run_program(program=None, seconds=30., log=None, stop_check=None, view=False):
    task = StepTask(program, seconds, log, stop_check)
    viewer = None
    if view:
        import mujoco.viewer as mjviewer
        viewer = mjviewer.launch_passive(task.r.model, task.r.data)
    try:
        while not task.done:
            task.step()
            if viewer:
                viewer.sync()
    finally:
        if viewer:
            viewer.close()
    return task.summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/v06')
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--suite', action='store_true'); parser.add_argument('--view', action='store_true')
    parser.add_argument('--random-commands', action='store_true'); parser.add_argument('--disturbed', action='store_true')
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    programs = ([StepProgram(), *[StepProgram(s, True) for s in (101, 102, 103)],
                 *[StepProgram(s, True, True) for s in (201, 202)]] if args.suite else
                [StepProgram(args.seed, args.random_commands, args.disturbed)])
    results = []
    for program in programs:
        label = f'seed{program.seed}_random{int(program.random_commands)}_push{int(program.disturbed)}'
        result = run_program(program, args.seconds, output/(label+'.csv'), view=args.view)
        results.append(result)
        print(json.dumps({k:result[k] for k in ('program', 'elapsed_s', 'passed', 'reason', 'failed_criteria')}), flush=True)
    report = dict(platform=platform.platform(), python=platform.python_version(), mujoco=mujoco.__version__,
        numpy=np.__version__, model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('core.py', 'whole_body.py', 'verification.py', 'contact_control.py', 'stepping.py')},
        results=results)
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not all(result['passed'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
