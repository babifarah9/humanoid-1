import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
from humanoid1.walking import WalkProgram, GaitPlan, run_program
from humanoid1.runtime import LocalRuntime, UnsupportedCapability


def test_repeated_gait_advances_alternating_feet_without_reference_resets():
    plan = GaitPlan(WalkProgram().specification())
    assert [step['swing'] for step in plan.steps] == [1, 0, 1, 0, 1, 0]
    assert np.allclose([step['target_x_m'] for step in plan.steps], [.04, .08, .12, .16, .20, .24])
    final = plan.reference(60.)
    assert np.allclose(final['feet'][:, 0], [.24, .20])
    assert final['support'] == (0, 1) and final['phase'] == 'standing_hold'
    for phase in plan.phases[:-1]:
        before, after = plan.reference(phase.end-1e-8), plan.reference(phase.end+1e-8)
        for key in ('com', 'com_velocity', 'com_acceleration', 'feet', 'foot_velocity', 'foot_acceleration'):
            assert np.allclose(before[key], after[key], atol=1e-6)


@pytest.mark.parametrize('when,active', [(5., 0), (8.8, 1), (24.4, 3), (58., 6), (65., 6)])
def test_controlled_stop_keeps_prefix_and_lands_airborne_foot(when, active):
    plan = GaitPlan(WalkProgram().specification())
    times = np.linspace(0., when, 100)
    original = [plan.reference(t) for t in times]
    assert plan.request_stop(when)
    assert not plan.request_stop(when+.1)
    assert plan.active_step_count == active
    for t, old in zip(times, original):
        new = plan.reference(t)
        assert np.allclose(new['com'], old['com']) and np.allclose(new['feet'], old['feet'])
    for previous, following in zip(plan.phases, plan.phases[1:]):
        assert previous.end == pytest.approx(following.start)
    final = plan.reference(plan.phases[-1].end)
    assert final['support'] == (0, 1) and np.allclose(final['feet'][:, 2], 0.)
    assert final['com'][2] == 0. and final['com'][1] == 0.


@pytest.mark.parametrize('program', [WalkProgram(steps=5), WalkProgram(steps=9), WalkProgram(seed=True),
    WalkProgram(stop_at_s=np.nan), WalkProgram(stop_at_s=-1.), WalkProgram(stop_at_s='invalid')])
def test_invalid_motion_programs_fail_before_simulation(program):
    with pytest.raises(ValueError):
        program.specification()


def test_short_episode_and_immediate_emergency_cannot_claim_walking():
    result = run_program(seconds=.2)
    assert not result['passed'] and not result['repeated_gait_verified'] and result['completed_steps'] == 0
    result = run_program(stop_check=lambda: True, controlled_stop_check=lambda: True)
    assert not result['passed'] and result['stopped'] and result['elapsed_s'] == 0.
    assert result['reason'] == 'operator runtime stop' and result['stop_requested_at_s'] is None
    json.dumps(result, allow_nan=False)


def test_controlled_stop_without_a_step_is_not_reported_as_a_walk():
    result = run_program(WalkProgram(stop_at_s=0.))
    assert result['passed'] and result['controlled_stop_verified']
    assert not result['goal_completed'] and not result['repeated_gait_verified']
    assert result['completed_steps'] == 0 and result['elapsed_s'] == pytest.approx(3.)
    assert not result['stopped']


def test_runtime_forward_goal_is_bounded_and_stop_requests_are_episode_scoped(tmp_path):
    runtime = LocalRuntime(tmp_path)
    assert runtime.plan('walk forward')['steps'][0] == dict(skill='walk_forward', seconds=60., seed=0)
    assert runtime.plan('stand')['steps'][0]['seconds'] == 30.
    assert not runtime.request_walking_stop()['accepted']
    with pytest.raises(ValueError):
        runtime.plan('walk forward', seconds=30.)
    for goal in ('walk', 'walk backward', 'walk forward and jump', 'turn left'):
        with pytest.raises(UnsupportedCapability):
            runtime.plan(goal)
    assert not runtime.memory.list('task')
    runtime.close()


def test_actual_six_step_forward_gait_and_final_standing():
    result = run_program()
    assert result['passed'], result
    assert result['elapsed_s'] == pytest.approx(60.) and result['samples'] == 30000
    assert result['completed_steps'] == 6 and result['body_progress_m'] >= .18
    assert result['repeated_gait_verified'] and result['goal_completed']
    assert result['final_standing_dwell_s'] >= 2. and result['final_com_error_m'] <= .003
    assert all(m['airborne_dwell_s'] >= .5 and m['landing_dwell_s'] >= .1
               and m['landing_error_m'] <= .005 for m in result['per_step_measurements'])
    assert not result['joint_limit_events'] and result['max_saturated_actuators'] == 0
    assert result['max_external_force_component_n'] == result['max_generalized_external_force'] == 0.
    assert not result['hardware_verified'] and not result['general_walking_verified']


def test_actual_cross_process_stop_during_measured_airborne_step_and_restart_recall(tmp_path):
    process = subprocess.Popen([sys.executable, '-m', 'humanoid1.runtime', '--state-dir', str(tmp_path),
        'execute', 'walk forward'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    observer = LocalRuntime(tmp_path)
    try:
        deadline = time.monotonic()+90.
        while time.monotonic() < deadline:
            tasks = observer.memory.list('task')
            record = tasks[-1]['value'] if tasks else {}
            progress = record.get('progress', {})
            if (progress.get('gait_phase') == 'swing_hold' and progress.get('gait_step') == 1
                    and progress['right_sole_clearance_m'] >= .015 and progress['right_normal_n'] < 1.):
                break
            if process.poll() is not None:
                raise AssertionError('walking worker ended before measured airborne phase')
            time.sleep(.03)
        else:
            raise AssertionError('no measured airborne step observed')
        request = json.loads(subprocess.check_output([sys.executable, '-m', 'humanoid1.runtime',
            '--state-dir', str(tmp_path), 'stop-walking', '--task-id', record['task_id']], text=True))
        assert request['accepted'] and request['task_ids'] == [record['task_id']]
        output, error = process.communicate(timeout=90.)
        assert process.returncode == 0, error
        result = json.loads(output)
        assert result['status'] == 'controlled_stop' and not result['goal_completed']
        evidence = json.loads(Path(result['evidence']).read_text())
        assert evidence['passed'] and evidence['controlled_stop_verified']
        assert evidence['completed_steps'] == 1 and evidence['active_steps'] == 1
        assert evidence['stop_requested_at_s'] >= 8.5
        assert evidence['controlled_stop_latency_s'] <= 15. and evidence['final_standing_dwell_s'] >= 2.
        assert not evidence['stopped'] and not evidence['joint_limit_events']
        recalled = json.loads(subprocess.check_output([sys.executable, '-m', 'humanoid1.runtime',
            '--state-dir', str(tmp_path), 'recall', result['task_id']], text=True))
        assert recalled == result
        assert not observer.status()['motion_safety_latch']['latched']
        assert not observer.request_walking_stop()['accepted']
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        observer.close()
