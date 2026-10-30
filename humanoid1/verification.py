"""Reproducible physics-step standing verification and baseline diagnosis.

Run: python -m humanoid1.verification --suite --output results/v03
All thresholds are declared here, before evaluating a rollout. Survival alone
is insufficient: every post-settling sample must pass the standing envelope.
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from .core import Humanoid, StandingPD, MODEL
from .balance import BalanceController
from .whole_body import WholeBodyController

CRITERIA = dict(settle_s=.10, max_torso_tilt_rad=.10, max_com_speed_m_s=.10,
                min_support_margin_m=.02, min_normal_per_foot_n=1.,
                min_pelvis_z_m=.95, max_pelvis_z_m=1.05,
                max_com_error_m=.03)


def standing_checks(di, margin, joint_margin):
    """One envelope shared by physics verification and policy evaluation."""
    return dict(tilt=di['torso_tilt_rad'] <= CRITERIA['max_torso_tilt_rad'],
                com_speed=di['com_speed_m_s'] <= CRITERIA['max_com_speed_m_s'],
                com_error=di['com_error_m'] <= CRITERIA['max_com_error_m'],
                support_margin=margin >= CRITERIA['min_support_margin_m'],
                bilateral_load=min(di['left_normal_n'], di['right_normal_n']) >= CRITERIA['min_normal_per_foot_n'],
                pelvis_height=CRITERIA['min_pelvis_z_m'] <= di['pelvis_z'] <= CRITERIA['max_pelvis_z_m'],
                no_nonfoot_support=di['nonfoot_floor_contacts'] == 0,
                no_joint_violation=joint_margin >= 0)


def support_margin(robot, com_xy, loaded):
    """Signed distance inside the convex hull of loaded sole box corners.

    A geometric support-polygon measure, not a dynamic ZMP/CoP certificate.
    Uses the foot geoms (including their +65 mm offset), not body origins.
    """
    m, d = robot.model, robot.data
    points = []
    for name, use in zip(('left_foot', 'right_foot'), loaded):
        if not use:
            continue
        bid = m.body(name).id
        g = int(np.flatnonzero(m.geom_bodyid == bid)[0])
        rot = d.geom_xmat[g].reshape(3, 3)
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = d.geom_xpos[g] + rot @ (m.geom_size[g] * [x, y, -1])
            points.append(tuple(p[:2]))
    if len(points) < 3:
        return -1.
    pts = sorted(set(points))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return -1.
    edges = np.roll(hull, -1, axis=0) - hull
    delta = np.asarray(com_xy) - hull
    distances = (edges[:, 0]*delta[:, 1]-edges[:, 1]*delta[:, 0])/np.linalg.norm(edges, axis=1)
    return float(np.min(distances))


def run_trial(mode='v03', seconds=30., control_ms=None, scenario='nominal', seed=0, log=None, view=False, stop_check=None):
    if mode not in ('v01', 'v02', 'v03'):
        raise ValueError('mode must be v01, v02 or v03')
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be positive and finite')
    r = Humanoid()
    monitor = WholeBodyController(r)
    ctl = {'v01': StandingPD, 'v02': BalanceController, 'v03': WholeBodyController}[mode](r)
    dt = r.model.opt.timestep
    period = (2. if mode == 'v03' else 20.) if control_ms is None else control_ms
    frames = int(round(period / (1000.*dt)))
    if frames < 1 or not np.isclose(frames*dt*1000, period):
        raise ValueError('control period must be a positive multiple of physics timestep')
    # References are captured BEFORE disturbances, so the target does not tilt
    # along with a perturbed robot. Randomness is explicit, local and seeded.
    rng = np.random.default_rng(seed)
    if scenario == 'initial_velocity':
        r.data.qvel[:2] = rng.uniform(-.015, .015, 2)
        r.data.qvel[3:5] = rng.uniform(-.01, .01, 2)
    elif scenario not in ('nominal', 'push_x', 'push_y'):
        raise ValueError('unknown scenario')
    mujoco.mj_forward(r.model, r.data)
    rows = []; failures = set(); worst = {}; violation_samples = 0; extrema = {}
    max_torque = 0.; max_saturation = 0; max_external = 0.; max_generalized_external = 0.
    q = r.data.qpos[r.qadr]
    min_joint_margin = float(np.min(np.minimum(q-r.limits[:, 0], r.limits[:, 1]-q)))
    min_ground_force = float('inf')
    start = time.perf_counter()
    viewer = None
    if view:
        import mujoco.viewer as mjviewer
        viewer = mjviewer.launch_passive(r.model, r.data)
    try:
        for k in range(int(np.ceil(seconds / dt))):
            if stop_check is not None and stop_check():
                r.emergency_stop('operator runtime stop')
            if r.stopped:
                break
            # Deliberate disturbance only, never a stabilizing external force.
            r.data.xfrc_applied[:] = 0.
            if scenario.startswith('push_') and 5. <= k*dt < 5.10:
                axis = 0 if scenario == 'push_x' else 1
                r.data.xfrc_applied[r.model.body('torso').id, axis] = 5.
            max_external = max(max_external, float(np.max(np.abs(r.data.xfrc_applied))))
            max_generalized_external = max(max_generalized_external, float(np.max(np.abs(r.data.qfrc_applied))))
            if k % frames == 0:
                tau = ctl.act()
                max_saturation = max(max_saturation, int(getattr(ctl, 'last', {}).get('n_saturated', 0)))
            _, info = r.step(tau, frames=1)
            # Capture last applied torque before forward refreshes derived state.
            applied = r.data.qfrc_actuator[r.vadr].copy()
            for event in r.joint_limit_violations():
                excess = max(event['lower_rad']-event['position_rad'], event['position_rad']-event['upper_rad'])
                if excess > extrema.get(event['joint'], {}).get('excess_rad', 0.):
                    extrema[event['joint']] = dict(event, excess_rad=excess)
            max_torque = max(max_torque, float(np.max(np.abs(applied))))
            di = monitor.diagnostics()
            margin = support_margin(r, [di['com_x'], di['com_y']],
                                    [di['left_normal_n'] >= 1., di['right_normal_n'] >= 1.])
            q = r.data.qpos[r.qadr]
            joint_margin = float(np.min(np.minimum(q-r.limits[:, 0], r.limits[:, 1]-q)))
            min_joint_margin = min(min_joint_margin, joint_margin)
            if joint_margin < 0:
                violation_samples += 1
            if r.data.time >= CRITERIA['settle_s']:
                checks = standing_checks(di, margin, joint_margin)
                failures.update(key for key, passed in checks.items() if not passed)
                for key in ('torso_tilt_rad', 'com_speed_m_s', 'com_error_m'):
                    worst[key] = max(worst.get(key, 0.), di[key])
                worst['min_support_margin_m'] = min(worst.get('min_support_margin_m', 1.), margin)
                min_ground_force = min(min_ground_force, di['left_normal_n'] + di['right_normal_n'])
            row = dict(di, support_margin_m=margin, min_joint_margin_rad=joint_margin,
                       max_applied_torque_nm=float(np.max(np.abs(applied))), stopped=r.stopped,
                       reason=info['reason'] or '')
            # Per-joint values are tied to the SAME integration timestamp.
            for i, name in enumerate(r.joint_names):
                row[name+'_q_rad'] = float(q[i])
                row[name+'_v_rad_s'] = float(r.data.qvel[r.vadr[i]])
                row[name+'_torque_nm'] = float(applied[i])
            rows.append(row)
            if viewer and k % 10 == 0:
                viewer.sync()
            if r.stopped:
                break
    finally:
        if viewer:
            viewer.close()
    elapsed = float(r.data.time)
    if log and rows:
        path = Path(log); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    complete = elapsed + dt*1e-6 >= seconds and not r.stopped
    return dict(mode=mode, scenario=scenario, seed=seed, control_ms=frames*dt*1000.,
                seconds_requested=seconds, elapsed_s=elapsed, samples=len(rows),
                wall_s=time.perf_counter()-start, stopped=r.stopped, reason=r.stop_reason,
                completed=bool(complete), stable=bool(complete and not failures and not r.limit_events),
                achieved_30s=bool(seconds >= 30. and elapsed+dt*1e-6 >= 30. and complete and not failures and not r.limit_events),
                criteria=CRITERIA, failed_criteria=sorted(failures), metrics=worst,
                min_joint_margin_rad=min_joint_margin, min_total_ground_force_n=min_ground_force if np.isfinite(min_ground_force) else None,
                max_applied_torque_nm=max_torque, max_saturated_actuators=max_saturation,
                nominal_joint_violation_samples=violation_samples,
                max_external_force_component_n=max_external,
                max_generalized_external_force=max_generalized_external,
                no_stabilizing_external_forces=bool(max_generalized_external == 0. and (scenario != 'nominal' or max_external == 0.)),
                equality_constraints=int(r.model.neq),
                stop_snapshot=r.stop_snapshot, joint_limit_events=r.limit_events,
                joint_limit_extrema=extrema,
                final=monitor.diagnostics(), log=str(log) if log else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['v01', 'v02', 'v03'], default='v03')
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--control-ms', type=float)
    p.add_argument('--scenario', choices=['nominal', 'initial_velocity', 'push_x', 'push_y'], default='nominal')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--suite', action='store_true')
    p.add_argument('--view', action='store_true')
    p.add_argument('--output', default='results/v03')
    a = p.parse_args()
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    try:
        base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        base = None
    jobs = [(a.mode, a.scenario, a.seed, a.control_ms)]
    if a.suite:
        jobs = [('v01', 'nominal', 0, 20.), ('v02', 'nominal', 0, 20.),
                ('v01', 'nominal', 0, 2.), ('v02', 'nominal', 0, 2.),
                ('v03', 'nominal', 0, 2.), ('v03', 'nominal', 0, 20.)]
        jobs += [('v03', 'initial_velocity', seed, 2.) for seed in (1, 2, 3)]
        jobs += [('v03', s, 0, 2.) for s in ('push_x', 'push_y')]
    results = []
    for mode, scenario, seed, control_ms in jobs:
        label = f'{mode}_{scenario}_seed{seed}_{control_ms or "default"}ms'
        result = run_trial(mode, a.seconds, control_ms, scenario, seed, output/(label+'.csv'), a.view)
        results.append(result)
        print(json.dumps({k: result[k] for k in ('mode', 'scenario', 'seed', 'control_ms', 'elapsed_s', 'stable', 'achieved_30s', 'reason', 'failed_criteria')}), flush=True)
    sources = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path('humanoid1').glob('*.py'))}
    report = dict(platform=platform.platform(), python=platform.python_version(), mujoco=mujoco.__version__,
                  numpy=np.__version__, base_commit=base, model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                  source_sha256=sources, results=results)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Evidence:', output/'summary.json')


if __name__ == '__main__':
    main()
