"""Bounded double-support weight shifting and recovery, with physics evidence.

This extends standing inside its existing home-COM envelope. It neither lifts a
foot nor provides walking. Only joint motors act on the robot; optional pushes
are explicit disturbances. No Gymnasium/Torch dependency is needed for baseline.
"""
from dataclasses import dataclass, asdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import mujoco
from .core import Humanoid, MODEL
from .whole_body import WholeBodyController
from .verification import CRITERIA, standing_checks, support_margin


TASK_CRITERIA = dict(min_duration_s=30., max_rms_tracking_xy_m=.003,
    max_peak_tracking_xy_m=.008, max_final_center_error_m=.002,
    min_com_y_range_m=.028, min_left_hold_load_fraction=.53,
    max_right_hold_load_fraction=.47, max_sole_drift_m=.003,
    recovery_xy_m=.001, recovery_speed_m_s=.003,
    recovery_dwell_s=.25, max_recovery_s=1.5)


@dataclass(frozen=True)
class ShiftProgram:
    seed: int = 0
    random_commands: bool = False
    disturbed: bool = False

    def build(self):
        rng = np.random.default_rng(self.seed)
        segments = []; holds = []; t = 1.
        for i in range(4):
            amplitude = float(rng.uniform(.016, .020)) if self.random_commands else .020
            target = amplitude * (1. if i % 2 == 0 else -1.)
            durations = np.array([2., 1., 2., 1.])
            if self.random_commands:
                durations *= rng.uniform(.8, 1.2, 4)
                durations *= 6./durations.sum()
            up, hold, down, center = durations
            segments.extend([(t, t+up, 0., target), (t+up, t+up+hold, target, target),
                             (t+up+hold, t+up+hold+down, target, 0.),
                             (t+up+hold+down, t+6., 0., 0.)])
            holds.append((t+up+.25, t+up+hold, int(np.sign(target))))
            t += 6.
        pulses = []
        if self.disturbed:
            # Truly different directions, amplitudes and durations for each seed.
            for start in (10.5, 18.5):
                angle = rng.uniform(-np.pi, np.pi)
                magnitude = rng.uniform(4., 8.)
                pulses.append(dict(start_s=start, end_s=start+float(rng.uniform(.10, .15)),
                    force_n=[float(magnitude*np.cos(angle)), float(magnitude*np.sin(angle)), 0.]))
        return segments, holds, pulses


def smooth_reference(t, segments):
    """Minimum-jerk interpolation with analytic position/velocity/acceleration."""
    for start, end, a, b in segments:
        if start <= t < end:
            u = (t-start)/(end-start); duration = end-start
            p = 10*u**3-15*u**4+6*u**5
            v = (30*u**2-60*u**3+30*u**4)/duration
            acc = (60*u-180*u**2+120*u**3)/duration**2
            return np.array([a+(b-a)*p, (b-a)*v, (b-a)*acc])
    return np.zeros(3)


class ShiftController(WholeBodyController):
    def __init__(self, robot):
        super().__init__(robot)
        self.reference_velocity = np.zeros(3)
        self.reference_acceleration = np.zeros(3)

    def com_task_acceleration(self, jacobian):
        return (super().com_task_acceleration(jacobian)
                + self.config.com_kd*self.reference_velocity + self.reference_acceleration)


class ShiftTask:
    def __init__(self, program=None, seconds=30., log=None, stop_check=None):
        if not np.isfinite(seconds) or seconds <= .1:
            raise ValueError('seconds must exceed contact settling')
        self.program = program or ShiftProgram()
        self.seconds = float(seconds)
        self.r = Humanoid(); self.c = ShiftController(self.r)
        self.home_com = self.c.target_com.copy()
        self.segments, self.holds, self.pulses = self.program.build()
        self.foot_start = np.array([self.r.data.xpos[b].copy() for b in self.c.feet])
        self.failed = set(); self.rows = []; self.recoveries = [None]*len(self.pulses)
        self.recovery_start = [None]*len(self.pulses)
        self.previous_action = np.zeros(self.r.nu); self.done = False
        self.slip_distance = np.zeros(2); self.work_j = 0.; self.reward_sum = 0.
        self.max_torque = 0.; self.max_saturated = 0; self.log = Path(log) if log else None
        self.last = {}; self.physics_steps = 0
        self.stop_check = stop_check

    def observation(self):
        r, c = self.r, self.c
        reference = smooth_reference(r.data.time, self.segments)
        normal, _, _ = c.contacts()
        jac = np.zeros((3, r.model.nv)); mujoco.mj_jacSubtreeCom(r.model, r.data, jac, c.pelvis)
        return np.nan_to_num(np.r_[r.observe(), reference,
            r.data.subtree_com[c.pelvis]-self.home_com, jac@r.data.qvel,
            normal/(32.8*9.81), r.data.time-c.last_loaded,
            self.previous_action, r.data.time/self.seconds].astype(np.float32))

    def _slip_speeds(self):
        r, c = self.r, self.c
        speeds = np.zeros(2); loads = np.zeros(2)
        for ci in range(r.data.ncon):
            contact = r.data.contact[ci]; pair = (int(contact.geom1), int(contact.geom2))
            if c.floor not in pair:
                continue
            other = pair[1] if pair[0] == c.floor else pair[0]
            if other not in c.foot_geoms:
                continue
            i = c.foot_geoms.index(other)
            force = np.zeros(6); mujoco.mj_contactForce(r.model, r.data, ci, force)
            if force[0] <= 0:
                continue
            jp = np.zeros((3, r.model.nv))
            mujoco.mj_jac(r.model, r.data, jp, None, contact.pos, c.feet[i])
            speeds[i] += force[0]*np.linalg.norm((jp@r.data.qvel)[:2])
            loads[i] += force[0]
        return np.divide(speeds, loads, out=np.zeros(2), where=loads >= 1.)

    def step(self, action=None, frames=10):
        if self.done:
            raise RuntimeError('task has finished; create a new task to reset')
        if not isinstance(frames,int) or frames < 1:
            raise ValueError('frames must be a positive integer')
        r, c = self.r, self.c; dt = r.model.opt.timestep
        action = np.zeros(r.nu) if action is None else np.asarray(action, dtype=float)
        if action.shape != (r.nu,) or not np.isfinite(action).all():
            r.emergency_stop('invalid weight-shift policy action'); self.done = True
            return self.observation(), -10., True, False, self.info()
        action = np.clip(action, -1., 1.)
        reward = 0.
        for _ in range(frames):
            if self.stop_check is not None and self.stop_check():
                r.emergency_stop('operator runtime stop')
            if r.stopped or r.data.time >= self.seconds-1e-10:
                break
            t = self.physics_steps*dt
            ref = smooth_reference(t, self.segments)
            c.target_com = self.home_com + [0., ref[0], 0.]
            c.reference_velocity[:] = [0., ref[1], 0.]
            c.reference_acceleration[:] = [0., ref[2], 0.]
            r.data.xfrc_applied[:] = 0.
            for pulse in self.pulses:
                if pulse['start_s'] <= t < pulse['end_s']:
                    r.data.xfrc_applied[c.torso, :3] = pulse['force_n']
            torque = c.act(action)
            self.max_saturated = max(self.max_saturated, c.last.get('n_saturated', 0))
            r.step(torque, frames=1); self.physics_steps += 1
            applied = r.data.qfrc_actuator[r.vadr].copy()
            self.max_torque = max(self.max_torque, float(np.max(abs(applied))))
            self.work_j += dt*float(np.sum(np.abs(applied*r.data.qvel[r.vadr])))
            di = c.diagnostics()
            # Keep the ORIGINAL absolute home-COM envelope, not a moving safety boundary.
            com = r.data.subtree_com[c.pelvis]
            di['com_error_m'] = float(np.linalg.norm(com-self.home_com))
            jac = np.zeros((3, r.model.nv)); mujoco.mj_jacSubtreeCom(r.model, r.data, jac, c.pelvis)
            com_vel = jac@r.data.qvel
            now_ref = smooth_reference(self.physics_steps*dt, self.segments)
            target = self.home_com + [0., now_ref[0], 0.]
            error = float(np.linalg.norm((com-target)[:2]))
            speed_error = float(np.linalg.norm((com_vel-np.array([0., now_ref[1], 0.]))[:2]))
            margin = support_margin(r, com[:2], [di['left_normal_n'] >= 1., di['right_normal_n'] >= 1.])
            q = r.data.qpos[r.qadr]
            joint_margin = float(np.min(np.minimum(q-r.limits[:, 0], r.limits[:, 1]-q)))
            drift = float(max(np.linalg.norm((r.data.xpos[b]-p)[:2]) for b,p in zip(c.feet,self.foot_start)))
            slips = self._slip_speeds(); self.slip_distance += slips*dt
            if r.data.time >= CRITERIA['settle_s']:
                self.failed.update(k for k,v in standing_checks(di, margin, joint_margin).items() if not v)
            if drift > TASK_CRITERIA['max_sole_drift_m']:
                self.failed.add('sole_drift')
            if r.limit_events:
                self.failed.add('nominal_joint_limit_event')
            for i, pulse in enumerate(self.pulses):
                if r.data.time >= pulse['end_s'] and self.recoveries[i] is None:
                    recovered = error <= TASK_CRITERIA['recovery_xy_m'] and speed_error <= TASK_CRITERIA['recovery_speed_m_s']
                    if not recovered:
                        self.recovery_start[i] = None
                    elif self.recovery_start[i] is None:
                        self.recovery_start[i] = float(r.data.time)
                    elif r.data.time-self.recovery_start[i] >= TASK_CRITERIA['recovery_dwell_s']:
                        self.recoveries[i] = max(0., self.recovery_start[i]-pulse['end_s'])
            total_normal = di['left_normal_n']+di['right_normal_n']
            row = dict(di, command_y_m=float(now_ref[0]), command_vy_m_s=float(now_ref[1]),
                actual_vy_m_s=float(com_vel[1]), tracking_xy_m=error, tracking_speed_error_m_s=speed_error,
                left_load_fraction=float(di['left_normal_n']/total_normal) if total_normal >= 1 else .5,
                support_margin_m=margin, joint_margin_rad=joint_margin, sole_drift_m=drift,
                left_slip_speed_m_s=float(slips[0]), right_slip_speed_m_s=float(slips[1]),
                mechanical_work_j=self.work_j, max_torque_nm=float(np.max(abs(applied))),
                residual_max_nm=float(2.*np.max(abs(action))), stopped=r.stopped, reason=r.stop_reason or '')
            self.last = row; self.rows.append(row)
            reward += dt*(1.-1000.*error**2-5.*di['torso_tilt_rad']**2-.001*np.mean(action**2))
            if self.failed:
                r.emergency_stop('weight-shift envelope: '+', '.join(sorted(self.failed)))
        reward -= .0001*float(np.mean((action-self.previous_action)**2))
        self.previous_action = action.copy()
        terminated = bool(r.stopped); truncated = bool(not terminated and r.data.time >= self.seconds-1e-10)
        self.done = terminated or truncated
        if terminated:
            reward -= 10.
        self.reward_sum += reward
        return self.observation(), float(reward), terminated, truncated, self.info()

    def info(self):
        return dict(time_s=float(self.r.data.time), reason=self.r.stop_reason,
            failed_criteria=sorted(self.failed), metrics=self.last.copy())

    def summary(self):
        rows = [r for r in self.rows if r['time_s'] >= CRITERIA['settle_s']]
        hold_fractions = {1: [], -1: []}
        for row in rows:
            for start,end,sign in self.holds:
                if start <= row['time_s'] < end:
                    hold_fractions[sign].append(row['left_load_fraction'])
        left = float(np.median(hold_fractions[1])) if hold_fractions[1] else None
        right = float(np.median(hold_fractions[-1])) if hold_fractions[-1] else None
        errors = [r['tracking_xy_m'] for r in rows]
        final = self.rows[-1] if self.rows else {}
        criteria = dict(duration=self.seconds >= 30. and self.r.data.time >= 30.-1e-8,
            completed=self.done and not self.r.stopped and self.r.data.time >= self.seconds-1e-8,
            safety=not self.failed and not self.r.stopped and not self.r.limit_events,
            rms_tracking=bool(errors) and np.sqrt(np.mean(np.square(errors))) <= TASK_CRITERIA['max_rms_tracking_xy_m'],
            peak_tracking=bool(errors) and max(errors) <= TASK_CRITERIA['max_peak_tracking_xy_m'],
            left_load=left is not None and left >= TASK_CRITERIA['min_left_hold_load_fraction'],
            right_load=right is not None and right <= TASK_CRITERIA['max_right_hold_load_fraction'],
            return_center=bool(final) and final['tracking_xy_m'] <= TASK_CRITERIA['max_final_center_error_m'],
            displacement=bool(rows) and max(r['com_y'] for r in rows)-min(r['com_y'] for r in rows) >= TASK_CRITERIA['min_com_y_range_m'],
            recovery=all(v is not None and v <= TASK_CRITERIA['max_recovery_s'] for v in self.recoveries))
        if self.log and self.rows:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(self.rows[0]),lineterminator='\n');w.writeheader();w.writerows(self.rows)
        return dict(program=asdict(self.program), seconds_requested=self.seconds, elapsed_s=float(self.r.data.time),
            passed=bool(all(criteria.values())), criteria={k:bool(v) for k,v in criteria.items()},
            safety_failed_criteria=sorted(self.failed), stopped=self.r.stopped, reason=self.r.stop_reason,
            rms_tracking_xy_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            peak_tracking_xy_m=max(errors) if errors else None, left_hold_load_fraction=left,
            right_hold_load_fraction=right, final_center_error_m=final.get('tracking_xy_m'),
            com_y_range_m=max((r['com_y'] for r in rows),default=0)-min((r['com_y'] for r in rows),default=0),
            mechanical_absolute_work_j=self.work_j, max_motor_torque_nm=self.max_torque,
            max_saturated_actuators=int(self.max_saturated), max_sole_drift_m=max((r['sole_drift_m'] for r in rows),default=0.),
            cumulative_contact_slip_m=self.slip_distance.tolist(), recovery_times_s=self.recoveries,
            joint_limit_events=self.r.limit_events, total_return=self.reward_sum,
            segments=self.segments, disturbances=self.pulses, log=str(self.log) if self.log else None)


def run_program(program, policy=None, log=None, seconds=30., view=False, stop_check=None):
    task = ShiftTask(program, seconds, log, stop_check)
    viewer = None
    if view:
        import mujoco.viewer as mv
        viewer = mv.launch_passive(task.r.model, task.r.data)
    try:
        while not task.done:
            action = None if policy is None else policy.predict(task.observation(), deterministic=True)[0]
            task.step(action)
            if viewer:
                viewer.sync()
    finally:
        if viewer:
            viewer.close()
    return task.summary()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', action='store_true');p.add_argument('--view',action='store_true')
    p.add_argument('--output', default='results/v05/baseline');p.add_argument('--policy')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--random-commands',action='store_true')
    p.add_argument('--disturbed',action='store_true');p.add_argument('--seconds',type=float,default=30.)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    policy = None
    if a.policy:
        from stable_baselines3 import PPO
        policy = PPO.load(a.policy, device='cpu')
    jobs = [ShiftProgram(a.seed,a.random_commands,a.disturbed)]
    if a.suite:
        jobs=[ShiftProgram(),*[ShiftProgram(s,True) for s in (101,102,103)],*[ShiftProgram(s,True,True) for s in (201,202)]]
    results=[]
    for job in jobs:
        label=f'seed{job.seed}_random{int(job.random_commands)}_push{int(job.disturbed)}'
        result=run_program(job,policy,out/(label+'.csv'),a.seconds,a.view);results.append(result)
        print(json.dumps({k:result[k] for k in ('program','elapsed_s','passed','rms_tracking_xy_m','left_hold_load_fraction','right_hold_load_fraction','reason')}),flush=True)
    report=dict(schema='humanoid1-weight-shift-v1',standing_criteria=CRITERIA,task_criteria=TASK_CRITERIA,
        model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(), platform=platform.platform(),
        mujoco=mujoco.__version__,numpy=np.__version__,
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('core.py','whole_body.py','verification.py','weight_shift.py')},
        policy_sha256=hashlib.sha256(Path(a.policy).read_bytes()).hexdigest() if a.policy else None,results=results)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
