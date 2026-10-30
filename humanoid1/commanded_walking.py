"""Bounded distance commands in four fixed-facing flat-floor directions.

Only joint motors move the unchanged reference body. Direction describes world
translation from the episode start, not yaw turning or obstacle navigation.
The v0.7 per-step contact, landing, standing and stop checks remain active.
"""
from dataclasses import dataclass, replace
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy as np
import mujoco
from .core import MODEL
from .walking import WalkProgram, GaitPlan, Phase, WalkTask, WALK_CRITERIA, source_fingerprint


DIRECTIONS = dict(forward=(1., 0.), backward=(-1., 0.), left=(0., 1.), right=(0., -1.))
COMMAND_CRITERIA = dict(min_distance_m=.20, max_distance_m=.45,
    max_endpoint_error_m=.005, max_cross_track_m=.005, max_torso_yaw_error_rad=.02)


@dataclass(frozen=True)
class CommandProgram:
    distance_m: float = .22
    direction: str = 'forward'
    seed: int = 0
    disturbed: bool = False
    stop_at_s: float | None = None

    @property
    def steps(self):
        return max(6, min(12, math.ceil(self.distance_m/.04+.5)))

    def specification(self):
        if (not isinstance(self.distance_m, (int, float)) or isinstance(self.distance_m, bool)
                or not np.isfinite(self.distance_m) or not .20 <= self.distance_m <= .45):
            raise ValueError('commanded distance must be finite and between 0.20 and 0.45 metres')
        if not isinstance(self.direction, str) or self.direction not in DIRECTIONS:
            raise ValueError('direction must be forward, backward, left or right; turning is unsupported')
        base = WalkProgram(seed=self.seed, disturbed=self.disturbed, stop_at_s=self.stop_at_s).specification()
        # The last two alternating foot targets are N*d and (N-1)*d. Their
        # midpoint is (N-.5)*d, so the final standing target matches the command.
        return dict(base, steps=self.steps, step_distance_m=self.distance_m/(self.steps-.5),
            distance_m=float(self.distance_m), direction=self.direction, direction_xy=list(DIRECTIONS[self.direction]))


class CommandGaitPlan(GaitPlan):
    def __init__(self, specification):
        super().__init__(specification)
        direction = np.asarray(specification['direction_xy'])

        def transform(values):
            result = values.copy()
            result[..., :2] = values[..., 0, None]*direction
            # Support transfers remain lateral relative to the fixed-facing
            # body; the commanded path alone is mapped to the selected axis.
            result[..., 1] += values[..., 1]
            return result

        self.phases = [replace(p, com_start=transform(p.com_start), com_end=transform(p.com_end),
            feet_start=transform(p.feet_start), feet_end=transform(p.feet_end)) for p in self.phases]
        for step in self.steps:
            target = step['target_x_m']*direction
            step.update(target_xy_m=target.tolist(), target_x_m=float(target[0]))

    def _append_standing(self, t, com, feet):
        center = np.r_[feet[:, :2].mean(axis=0), com[2]]
        if np.linalg.norm(com-center) > 1e-12:
            self.phases.append(Phase(t, t+4., 'center', com.copy(), center.copy(), feet.copy(), feet.copy()))
            t += 4.; com = center
        upright = com.copy(); upright[2] = 0.
        if abs(com[2]) > 1e-12:
            self.phases.append(Phase(t, t+3., 'stand_up', com.copy(), upright.copy(), feet.copy(), feet.copy()))
            t += 3.; com = upright
        self.phases.append(Phase(t, t+3., 'standing_hold', com.copy(), com.copy(), feet.copy(), feet.copy()))


class CommandTask(WalkTask):
    plan_type = CommandGaitPlan

    def __init__(self, program=None, **kwargs):
        program = program or CommandProgram()
        specification = program.specification()
        self.direction = np.asarray(specification['direction_xy'])
        self.max_cross_track = 0.; self.max_yaw_error = 0.
        super().__init__(program, **kwargs)
        self.home_yaw = self._yaw()
        for measurement, step in zip(self.step_measurements, self.plan.steps):
            measurement['target_xy_m'] = step['target_xy_m']

    def _yaw(self):
        rotation = self.r.data.xmat[self.c.torso].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def body_progress(self):
        return float((self.r.data.qpos[:2]-self.home_root[:2]) @ self.direction)

    def after_sample(self, *args):
        super().after_sample(*args)
        row = args[-1]
        displacement = self.r.data.qpos[:2]-self.home_root[:2]
        # Compare path tracking to the COM's intentional lateral balance
        # reference, rather than calling the 0.10 m support transfer a drift.
        reference = self.motion_reference(float(self.r.data.time))
        error = self.r.data.subtree_com[self.c.pelvis][:2]-self.c.home_com[:2]-reference['com'][:2]
        cross = float(error[0]*self.direction[1]-error[1]*self.direction[0])
        yaw_error = math.atan2(math.sin(self._yaw()-self.home_yaw), math.cos(self._yaw()-self.home_yaw))
        self.max_cross_track = max(self.max_cross_track, abs(cross))
        self.max_yaw_error = max(self.max_yaw_error, abs(yaw_error))
        row.update(body_x_m=float(displacement[0]), body_y_m=float(displacement[1]),
            reference_cross_track_error_m=cross, torso_yaw_error_rad=yaw_error)
        if abs(cross) > COMMAND_CRITERIA['max_cross_track_m']:
            self.failed.add('command_cross_track')
        if abs(yaw_error) > COMMAND_CRITERIA['max_torso_yaw_error_rad']:
            self.failed.add('command_yaw')

    def summary(self):
        result = super().summary()
        displacement = self.r.data.qpos[:2]-self.home_root[:2]
        target = self.motion_reference(self.plan.phases[-1].end)['com'][:2]
        endpoint_error = float(np.linalg.norm(displacement-target))
        checks = result['acceptance']
        checks.update(command_endpoint=endpoint_error <= COMMAND_CRITERIA['max_endpoint_error_m'],
            reference_cross_track=self.max_cross_track <= COMMAND_CRITERIA['max_cross_track_m'],
            fixed_facing=self.max_yaw_error <= COMMAND_CRITERIA['max_torso_yaw_error_rad'])
        if self.plan.stop_requested_at is None:
            checks['command_distance'] = abs(self.body_progress()-self.program.distance_m) <= COMMAND_CRITERIA['max_endpoint_error_m']
        passed = bool(all(checks.values()))
        result.update(skill='commanded_walk', walking_scope='bounded_fixed_facing_four_direction_flat_floor',
            passed=passed, goal_completed=bool(passed and self.plan.stop_requested_at is None),
            repeated_gait_verified=bool(passed and result['completed_steps'] >= 6 and self.body_progress() >= .18),
            controlled_stop_verified=bool(passed and self.plan.stop_requested_at is not None),
            command=dict(distance_m=self.program.distance_m, direction=self.program.direction,
                desired_body_displacement_xy_m=(self.program.distance_m*self.direction).tolist()),
            body_displacement_xy_m=displacement.tolist(), active_endpoint_xy_m=target.tolist(),
            endpoint_error_m=endpoint_error, max_reference_cross_track_error_m=self.max_cross_track,
            max_torso_yaw_error_rad=self.max_yaw_error, turning_verified=False,
            criteria=dict(result['criteria'], command=COMMAND_CRITERIA))
        return result


def run_program(program=None, seconds=None, log=None, stop_check=None, controlled_stop_check=None,
                progress_callback=None, view=False):
    task = CommandTask(program, seconds=seconds, log=log, stop_check=stop_check,
        controlled_stop_check=controlled_stop_check, progress_callback=progress_callback)
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


def fingerprint():
    return dict(source_fingerprint(), commanded_walking=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance', type=float, default=.22)
    parser.add_argument('--direction', choices=tuple(DIRECTIONS), default='forward')
    parser.add_argument('--stop-at', type=float); parser.add_argument('--seconds', type=float)
    parser.add_argument('--seed', type=int, default=0); parser.add_argument('--disturbed', action='store_true')
    parser.add_argument('--view', action='store_true'); parser.add_argument('--suite', action='store_true')
    parser.add_argument('--output', default='results/v08')
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    programs = ([CommandProgram(.22, d) for d in DIRECTIONS] + [CommandProgram(.45),
        CommandProgram(.20), CommandProgram(.22, 'backward', seed=201, disturbed=True),
        CommandProgram(.22, 'left', stop_at_s=8.8)] if args.suite else
        [CommandProgram(args.distance, args.direction, args.seed, args.disturbed, args.stop_at)])
    sources = fingerprint(); model = hashlib.sha256(MODEL.read_bytes()).hexdigest(); results = []
    for index, program in enumerate(programs):
        result = run_program(program, seconds=args.seconds, log=output/f'{index}_{program.direction}_{program.distance_m}.csv', view=args.view)
        results.append(result)
        print(json.dumps({k:result[k] for k in ('command', 'elapsed_s', 'passed', 'completed_steps',
            'body_displacement_xy_m', 'endpoint_error_m', 'reason', 'failed_criteria')}), flush=True)
    if fingerprint() != sources or hashlib.sha256(MODEL.read_bytes()).hexdigest() != model:
        raise RuntimeError('source/model changed during evaluation; rerun for matched evidence')
    (output/'summary.json').write_text(json.dumps(dict(platform=platform.platform(), python=platform.python_version(),
        mujoco=mujoco.__version__, numpy=np.__version__, model_sha256=model, source_sha256=sources,
        results=results), indent=2, allow_nan=False)+'\n')
    if not all(r['passed'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
