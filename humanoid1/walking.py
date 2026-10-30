"""Slow straight walking and controlled stopping on the existing MuJoCo body.

This is a six-to-eight-step flat-floor gait. It does not provide turning, terrain
adaptation or general navigation. Only joint motors drive the robot. A controlled
stop completes a current swing before returning to standing; emergency stop is
the independent existing latch and takes priority at every physics substep.
"""
from dataclasses import dataclass, asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy as np
import mujoco
from .core import MODEL
from .stepping import StepTask, STEP_CRITERIA, interpolate


WALK_CRITERIA = dict(min_repeated_steps=6, min_body_progress_m=.18,
    min_airborne_dwell_per_step_s=.5, min_landing_dwell_per_step_s=.10,
    max_landing_error_m=.005, max_landing_vertical_speed_m_s=.08,
    min_final_standing_dwell_s=2., max_final_com_error_m=.003,
    max_final_com_speed_m_s=.003, max_final_foot_speed_m_s=.003,
    max_final_torso_tilt_rad=.01, max_controlled_stop_latency_s=15.)


def source_fingerprint():
    return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('core.py', 'whole_body.py', 'verification.py', 'contact_control.py', 'stepping.py', 'walking.py')}


@dataclass(frozen=True)
class WalkProgram:
    seed: int = 0
    steps: int = 6
    random_commands: bool = False
    disturbed: bool = False
    stop_at_s: float | None = None

    def specification(self):
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError('seed must be an integer')
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or not 6 <= self.steps <= 8:
            raise ValueError('walking program requires six to eight steps')
        if self.stop_at_s is not None and (not isinstance(self.stop_at_s, (int, float)) or isinstance(self.stop_at_s, bool)
                or not np.isfinite(self.stop_at_s) or not 0. <= self.stop_at_s <= 90.):
            raise ValueError('controlled stop time must be finite and between 0 and 90 seconds')
        rng = np.random.default_rng(self.seed)
        increment = float(rng.uniform(.035, .045)) if self.random_commands else .04
        shift = float(rng.uniform(.095, .105)) if self.random_commands else .10
        height = float(rng.uniform(.028, .035)) if self.random_commands else .03
        pulses = []
        if self.disturbed:
            for start in (8.8, 24.3):
                angle = rng.uniform(-np.pi, np.pi); magnitude = rng.uniform(2., 4.)
                pulses.append(dict(start_s=start, end_s=start+.10,
                    force_n=[float(magnitude*np.cos(angle)), float(magnitude*np.sin(angle)), 0.]))
        return dict(step_distance_m=increment, lateral_shift_m=shift, commanded_clearance_m=height,
                    steps=self.steps, transfer_s=4.5, swing_s=3., landing_s=.25,
                    stop_at_s=self.stop_at_s, pulses=pulses)


@dataclass(frozen=True)
class Phase:
    start: float
    end: float
    kind: str
    com_start: np.ndarray
    com_end: np.ndarray
    feet_start: np.ndarray
    feet_end: np.ndarray
    support: tuple = (0, 1)
    step_index: int = -1
    swing: int = -1

    def evaluate(self, t):
        com, velocity, acceleration = interpolate(t, self.start, self.end, self.com_start, self.com_end)
        feet, foot_velocity, foot_acceleration = interpolate(t, self.start, self.end, self.feet_start, self.feet_end)
        return dict(com=com, com_velocity=velocity, com_acceleration=acceleration,
            feet=feet, foot_velocity=foot_velocity, foot_acceleration=foot_acceleration,
            support=self.support, phase=self.kind, step_index=self.step_index, swing=self.swing)


class GaitPlan:
    def __init__(self, specification):
        self.specification = specification
        self.phases = []; self.steps = []; self.stop_requested_at = None
        self.stop_reason = None; self.active_step_count = specification['steps']
        increment = specification['step_distance_m']; shift = specification['lateral_shift_m']
        height = specification['commanded_clearance_m']
        feet = np.zeros((2, 3)); com = np.zeros(3); t = 0.

        def add(duration, kind, next_com, next_feet, support=(0, 1), index=-1, swing=-1):
            nonlocal t, com, feet
            self.phases.append(Phase(t, t+duration, kind, com.copy(), np.asarray(next_com).copy(),
                feet.copy(), np.asarray(next_feet).copy(), support, index, swing))
            t += duration; com = np.asarray(next_com).copy(); feet = np.asarray(next_feet).copy()

        add(1., 'initial_settle', com, feet)
        add(2., 'crouch', [0., 0., -.03], feet)
        for index in range(specification['steps']):
            swing = 1 if index % 2 == 0 else 0; stance = 1-swing
            target_com = [feet[stance, 0]+.5*increment, shift if stance == 0 else -shift, -.03]
            add(specification['transfer_s'], 'transfer', target_com, feet, index=index, swing=swing)
            initial_feet = feet.copy(); lift = feet.copy()
            lift[swing] = [(index+1)*increment, 0., height]
            start = t
            add(1., 'lift', com, lift, (stance,), index, swing)
            add(1., 'swing_hold', com, lift, (stance,), index, swing)
            land = lift.copy(); land[swing, 2] = 0.
            add(1., 'lower', com, land, (stance,), index, swing)
            add(.25, 'landing', com, land, (0, 1), index, swing)
            self.steps.append(dict(index=index, swing=swing, start_s=start, end_s=t,
                from_x_m=float(initial_feet[swing, 0]), target_x_m=float(land[swing, 0])))
        self.automatic_stop_at = t
        self._append_standing(t, com, feet)
        self.minimum_seconds = max(60., 10.*math.ceil(self.phases[-1].end/10.))
        if self.specification['stop_at_s'] is not None and self.specification['stop_at_s'] >= self.minimum_seconds:
            raise ValueError('controlled stop must occur before the episode ends')

    def _append_standing(self, t, com, feet):
        center = np.array([float(feet[:, 0].mean()), 0., com[2]])
        if np.linalg.norm(com-center) > 1e-12:
            self.phases.append(Phase(t, t+4., 'center', com.copy(), center.copy(), feet.copy(), feet.copy()))
            t += 4.; com = center
        upright = com.copy(); upright[2] = 0.
        if abs(com[2]) > 1e-12:
            self.phases.append(Phase(t, t+3., 'stand_up', com.copy(), upright.copy(), feet.copy(), feet.copy()))
            t += 3.; com = upright
        self.phases.append(Phase(t, t+3., 'standing_hold', com.copy(), com.copy(), feet.copy(), feet.copy()))

    def phase_at(self, t):
        return next((phase for phase in self.phases if t < phase.end-1e-10), self.phases[-1])

    def reference(self, t):
        return self.phase_at(t).evaluate(t)

    def request_stop(self, t, reason='operator controlled stop'):
        if self.stop_requested_at is not None:
            return False
        if not np.isfinite(t) or t < 0.:
            raise ValueError('stop timestamp must be finite and nonnegative')
        self.stop_requested_at = float(t); self.stop_reason = reason
        current = self.phase_at(t)
        if current.kind in ('lift', 'swing_hold', 'lower', 'landing'):
            boundary = next(phase for phase in self.phases
                            if phase.kind == 'landing' and phase.step_index == current.step_index)
            cut = boundary.end; endpoint = boundary
        else:
            cut = t if current.kind in ('initial_settle', 'standing_hold') else current.end
            endpoint = current
        self.phases = [phase for phase in self.phases if phase.end <= cut+1e-10
                       and not (phase is current and current.kind in ('initial_settle', 'standing_hold'))]
        # For constant hold phases, cutting at the request time preserves all derivatives.
        if current.kind in ('initial_settle', 'standing_hold') and cut > current.start:
            self.phases.append(Phase(current.start, cut, current.kind, current.com_start.copy(),
                current.com_end.copy(), current.feet_start.copy(), current.feet_end.copy()))
        self.active_step_count = sum(step['end_s'] <= cut+1e-10 for step in self.steps)
        self._append_standing(cut, endpoint.com_end.copy(), endpoint.feet_end.copy())
        return True


class WalkTask(StepTask):
    max_seconds = 120.

    def __init__(self, program=None, seconds=None, log=None, stop_check=None, controlled_stop_check=None,
                 progress_callback=None):
        program = program or WalkProgram()
        plan = GaitPlan(program.specification())
        seconds = plan.minimum_seconds if seconds is None else seconds
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or not .1 < seconds <= 120.:
            raise ValueError('walking duration must be in (0.1, 120] seconds')
        super().__init__(program, seconds, log, stop_check)
        self.plan = plan; self.requested_seconds = float(seconds)
        self.controlled_stop_check = controlled_stop_check
        self.progress_callback = progress_callback; self.next_progress_time = 0.; self.previous_phase = None
        self.home_root = self.r.data.qpos[:3].copy()
        self.step_measurements = [dict(index=step['index'], swing_foot='left' if step['swing'] == 0 else 'right',
            target_x_m=step['target_x_m'], max_clearance_m=0., airborne_dwell_s=0.,
            landing_dwell_s=0., landing_error_m=None, max_landing_vertical_speed_m_s=0., completed=False)
            for step in plan.steps]
        self.airborne_run = np.zeros(program.steps); self.landing_run = np.zeros(program.steps)
        self.standing_run = 0.; self.standing_dwell = 0.; self.stop_complete_at = None

    def request_stop(self, reason='operator controlled stop'):
        changed = self.plan.request_stop(float(self.r.data.time), reason)
        if changed:
            self.seconds = min(self.max_seconds, self.plan.phases[-1].end)
            self.standing_run = 0.; self.standing_dwell = 0.; self.stop_complete_at = None
        return changed

    def before_substep(self, t):
        scheduled = self.specification['stop_at_s']
        if scheduled is not None and t >= scheduled-1e-10:
            self.request_stop('scheduled controlled stop')
        if self.controlled_stop_check is not None and self.controlled_stop_check():
            self.request_stop()

    def motion_reference(self, t):
        return self.plan.reference(t)

    def landing_phase(self, t, foot):
        phase = self.plan.phase_at(t)
        return phase.kind == 'lower' and phase.swing == foot

    def after_sample(self, t, reference, diagnostics, feet, loads, clearance, margin, row):
        dt = float(self.r.model.opt.timestep); index = reference['step_index']; swing = reference['swing']
        velocities = []
        for body in self.c.feet:
            jac = np.zeros((3, self.r.model.nv))
            mujoco.mj_jacBody(self.r.model, self.r.data, jac, None, body)
            velocities.append(jac @ self.r.data.qvel)
        if index >= 0:
            measured = self.step_measurements[index]
            measured['max_clearance_m'] = max(measured['max_clearance_m'], float(clearance[swing]))
            eligible = (swing not in reference['support'] and clearance[swing] >= STEP_CRITERIA['min_clearance_m']
                and loads[swing] < STEP_CRITERIA['max_swing_load_n']
                and loads[1-swing] >= STEP_CRITERIA['min_stance_load_fraction']*32.8*9.81
                and margin >= STEP_CRITERIA['min_support_margin_m'])
            self.airborne_run[index] = self.airborne_run[index]+dt if eligible else 0.
            measured['airborne_dwell_s'] = max(measured['airborne_dwell_s'], float(self.airborne_run[index]))
            if reference['phase'] == 'lower' and loads[swing] >= 1.:
                measured['max_landing_vertical_speed_m_s'] = max(measured['max_landing_vertical_speed_m_s'],
                    abs(float(velocities[swing][2])))
            if reference['phase'] == 'landing':
                desired = self.c.home_feet[swing] + reference['feet'][swing]
                error = float(np.linalg.norm((feet[swing]-desired)[:2]))
                landed = bool(np.all(loads >= 1.) and clearance[swing] <= .003
                    and error <= WALK_CRITERIA['max_landing_error_m'])
                self.landing_run[index] = self.landing_run[index]+dt if landed else 0.
                measured['landing_error_m'] = error
                measured['landing_dwell_s'] = max(measured['landing_dwell_s'], float(self.landing_run[index]))
                measured['completed'] = bool(measured['airborne_dwell_s'] >= WALK_CRITERIA['min_airborne_dwell_per_step_s']
                    and self.landing_run[index] >= WALK_CRITERIA['min_landing_dwell_per_step_s']
                    and measured['max_landing_vertical_speed_m_s'] <= WALK_CRITERIA['max_landing_vertical_speed_m_s'])
        standing = (reference['phase'] == 'standing_hold' and np.all(loads >= 1.)
            and row['tracking_com_m'] <= WALK_CRITERIA['max_final_com_error_m']
            and diagnostics['com_speed_m_s'] <= WALK_CRITERIA['max_final_com_speed_m_s']
            and diagnostics['torso_tilt_rad'] <= WALK_CRITERIA['max_final_torso_tilt_rad']
            and max(np.linalg.norm(v) for v in velocities) <= WALK_CRITERIA['max_final_foot_speed_m_s'])
        self.standing_run = self.standing_run+dt if standing else 0.
        self.standing_dwell = max(self.standing_dwell, self.standing_run)
        if self.standing_run >= WALK_CRITERIA['min_final_standing_dwell_s'] and self.stop_complete_at is None:
            self.stop_complete_at = float(self.r.data.time)
        row.update(gait_phase=reference['phase'], gait_step=index+1,
            completed_steps=sum(m['completed'] for m in self.step_measurements),
            body_progress_m=float(self.r.data.qpos[0]-self.home_root[0]),
            controlled_stop_requested=self.plan.stop_requested_at is not None,
            standing_dwell_s=self.standing_run)
        if self.progress_callback is not None and (self.r.data.time >= self.next_progress_time
                                                   or reference['phase'] != self.previous_phase):
            self.progress_callback({key:row[key] for key in ('time_s', 'gait_phase', 'gait_step',
                'completed_steps', 'body_progress_m', 'controlled_stop_requested',
                'left_sole_clearance_m', 'right_sole_clearance_m', 'left_normal_n', 'right_normal_n')})
            self.next_progress_time = float(self.r.data.time)+.2
            self.previous_phase = reference['phase']

    def summary(self):
        # Reuse original logging/extrema, then replace the two-step-specific acceptance.
        result = super().summary()
        completed = sum(m['completed'] for m in self.step_measurements)
        active = self.plan.active_step_count
        final_reference = self.motion_reference(float(self.r.data.time))
        final_feet = np.asarray([self.r.data.xpos[b].copy() for b in self.c.feet])
        placement = np.linalg.norm((final_feet-self.c.home_feet-final_reference['feet'])[:, :2], axis=1)
        final_error = float(np.linalg.norm(self.r.data.subtree_com[self.c.pelvis]-self.c.home_com-final_reference['com']))
        progress = float(self.r.data.qpos[0]-self.home_root[0])
        requested = self.plan.stop_requested_at
        latency = None if requested is None or self.stop_complete_at is None else max(0., self.stop_complete_at-requested)
        checks = dict(completed_horizon=self.done and self.r.data.time >= self.seconds-1e-8,
            safety=not self.r.stopped and not self.failed and not self.r.limit_events,
            all_active_steps=completed == active and all(m['completed'] for m in self.step_measurements[:active]),
            final_standing=self.standing_dwell >= WALK_CRITERIA['min_final_standing_dwell_s'],
            final_com=final_error <= WALK_CRITERIA['max_final_com_error_m'],
            final_placements=bool(np.all(placement <= WALK_CRITERIA['max_landing_error_m'])),
            no_torque_saturation=self.max_saturated == 0,
            no_generalized_external_force=self.generalized_external_max == 0., no_constraints=int(self.r.model.neq) == 0)
        if requested is None:
            checks.update(full_program=active == self.program.steps and completed >= WALK_CRITERIA['min_repeated_steps'],
                full_duration=self.requested_seconds >= self.plan.minimum_seconds,
                body_progress=progress >= WALK_CRITERIA['min_body_progress_m'])
        else:
            checks['controlled_stop_latency'] = latency is not None and latency <= WALK_CRITERIA['max_controlled_stop_latency_s']
        passed = bool(all(checks.values()))
        result.update(skill='walk_forward', walking_scope='slow_straight_flat_floor',
            continuous_walking_verified=False, general_walking_verified=False,
            repeated_gait_verified=bool(passed and completed >= 6 and progress >= .18),
            controlled_stop_verified=bool(passed and requested is not None),
            success_scope='controlled_stop' if requested is not None else 'repeated_gait',
            goal_completed=bool(passed and requested is None),
            passed=passed, acceptance={key:bool(value) for key,value in checks.items()},
            criteria=dict(motion=STEP_CRITERIA, gait=WALK_CRITERIA),
            seconds_requested=self.requested_seconds, actual_program_horizon_s=self.seconds,
            planned_steps=self.program.steps, active_steps=active, completed_steps=completed,
            per_step_measurements=self.step_measurements,
            body_progress_m=progress, foot_placement_errors_xy_m=placement.tolist(), final_com_error_m=final_error,
            final_standing_dwell_s=self.standing_dwell, stop_requested_at_s=requested,
            stop_request_reason=self.plan.stop_reason, stop_complete_at_s=self.stop_complete_at,
            controlled_stop_latency_s=latency)
        return result


def run_program(program=None, seconds=None, log=None, stop_check=None, controlled_stop_check=None, view=False,
                progress_callback=None):
    task = WalkTask(program, seconds, log, stop_check, controlled_stop_check, progress_callback)
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
    parser.add_argument('--output', default='results/v07')
    parser.add_argument('--seconds', type=float); parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=6); parser.add_argument('--stop-at', type=float)
    parser.add_argument('--suite', action='store_true'); parser.add_argument('--view', action='store_true')
    parser.add_argument('--random-commands', action='store_true'); parser.add_argument('--disturbed', action='store_true')
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    sources = source_fingerprint(); model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    programs = ([WalkProgram(), *[WalkProgram(s, random_commands=True) for s in (101, 102, 103)],
        *[WalkProgram(s, random_commands=True, disturbed=True) for s in (201, 202)],
        *[WalkProgram(stop_at_s=t) for t in (5., 8.8, 24.4)]] if args.suite else
        [WalkProgram(args.seed, args.steps, args.random_commands, args.disturbed, args.stop_at)])
    results = []
    for program in programs:
        label = f'seed{program.seed}_random{int(program.random_commands)}_push{int(program.disturbed)}_stop{program.stop_at_s}'
        result = run_program(program, args.seconds, output/(label+'.csv'), view=args.view)
        results.append(result)
        print(json.dumps({key:result[key] for key in ('program', 'elapsed_s', 'passed', 'completed_steps', 'body_progress_m', 'reason')}), flush=True)
    if sources != source_fingerprint() or model_sha != hashlib.sha256(MODEL.read_bytes()).hexdigest():
        raise RuntimeError('controller/model sources changed during evaluation; rerun for matched evidence')
    report = dict(platform=platform.platform(), python=platform.python_version(), mujoco=mujoco.__version__,
        numpy=np.__version__, model_sha256=model_sha, source_sha256=sources, results=results)
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not all(result['passed'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
