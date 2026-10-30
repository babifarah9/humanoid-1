"""Contact feasibility, measured motion, phase boundaries and failure behavior."""
import json
import numpy as np
import pytest
from humanoid1.core import Humanoid
from humanoid1.contact_control import ContactController, contact_bounds, allocate_wrenches, InfeasibleContact
from humanoid1.stepping import StepProgram, StepTask, reference, run_program
from humanoid1.runtime import LocalRuntime, UnsupportedCapability


def test_wrench_allocation_respects_unilateral_contact_and_rejects_impossible_root_force():
    r = Humanoid(); c = ContactController(r)
    C = contact_bounds(r, c.foot_geoms, (0, 1))
    A = np.hstack((np.eye(6), np.eye(6)))
    target = np.zeros(12); target[2] = -10.; target[8] = 110.
    b = np.array([0., 0., 100., 0., 0., 0.])
    wrench, info = allocate_wrenches(np.eye(12), target, A, b, C)
    assert np.allclose(A @ wrench, b, atol=1e-8)
    assert np.min(C @ wrench) >= -1e-8
    assert wrench[2] == pytest.approx(0., abs=1e-8)
    assert info['root_residual'] < 1e-8
    for impossible in (np.array([0., 0., -100., 0., 0., 0.]),
                       np.array([1000., 0., 100., 0., 0., 0.])):
        with pytest.raises(InfeasibleContact):
            allocate_wrenches(np.eye(12), target, A, impossible, C)


def test_sole_offset_and_foot_frame_are_accounted_for_in_bounds():
    r = Humanoid(); c = ContactController(r)
    C = contact_bounds(r, c.foot_geoms, (0,))
    inside = np.array([0., 0., 100., 0., -20., 0.])  # CoP x = +.20 m
    outside = np.array([0., 0., 100., 0., -23., 0.])  # +.23 exceeds +.215
    assert np.min(C @ inside) >= 0.
    assert np.min(C @ outside) < 0.


def test_program_is_seeded_continuous_and_only_releases_scheduled_foot():
    specification = StepProgram(101, True, True).specification()
    assert specification == StepProgram(101, True, True).specification()
    assert specification != StepProgram(102, True, True).specification()
    for t in (1., 3., 6., 7., 8., 9., 12., 15., 16., 17., 18., 21., 22., 25.):
        before, after = reference(t-1e-8, specification), reference(t+1e-8, specification)
        for key in ('com', 'com_velocity', 'com_acceleration', 'feet', 'foot_velocity', 'foot_acceleration'):
            assert np.allclose(before[key], after[key], atol=1e-6)
    assert reference(7., specification)['support'] == (0,)
    assert reference(16., specification)['support'] == (1,)
    assert reference(26., specification)['support'] == (0, 1)
    final = reference(30., specification)
    assert np.allclose(final['feet'][:, 0], specification['step_distance_m'])
    assert np.allclose(final['feet'][:, 2], 0.)


@pytest.mark.parametrize('fault', ['position', 'height', 'velocity', 'nan'])
def test_contact_controller_preserves_core_stop_limits(fault):
    r = Humanoid(); c = ContactController(r)
    if fault == 'position':
        r.data.qpos[r.qadr[0]] = r.limits[0, 1]+.071
    elif fault == 'height':
        r.data.qpos[2] = .59
    elif fault == 'velocity':
        r.data.qvel[r.vadr[0]] = 91.
    else:
        r.data.qvel[0] = np.nan
    assert not c.act().any() and r.stopped
    r.step(np.zeros(16))
    assert r.data.time == 0.


def test_expected_contact_loss_and_invalid_residual_fail_closed():
    r = Humanoid(); c = ContactController(r)
    r.data.qpos[2] += .5; r.data.time = .2
    assert not c.act().any() and r.stopped
    assert r.stop_reason == 'planned support contact lost'
    r = Humanoid(); c = ContactController(r)
    assert not c.act(np.zeros(16)).any() and r.stopped
    r = Humanoid(); c = ContactController(r)
    c.planned_support = ()
    assert not c.act().any() and r.stopped


def test_short_or_cancelled_motion_cannot_claim_two_step_success():
    result = run_program(seconds=.2)
    assert not result['passed'] and not result['stopped']
    result = run_program(stop_check=lambda: True)
    assert not result['passed'] and result['elapsed_s'] == 0.
    assert result['reason'] == 'operator runtime stop'
    json.dumps(result, allow_nan=False)


def test_runtime_exposes_bounded_steps_and_preserves_stop_and_walking_rejection(tmp_path):
    runtime = LocalRuntime(tmp_path)
    assert runtime.plan('take two steps')['steps'][0]['skill'] == 'two_step_advance'
    with pytest.raises(ValueError):
        runtime.plan('take two steps', seconds=2.)
    for goal in ('walk', 'walk and take two steps', 'take two steps and jump'):
        with pytest.raises(UnsupportedCapability):
            runtime.execute(goal)
    runtime.emergency_stop()
    with pytest.raises(RuntimeError):
        runtime.execute('take two steps')
    assert not runtime.memory.list('task')
    runtime.close()


def test_complete_two_step_physics_evidence():
    # A real full-horizon regression, independently checking measured outcomes.
    result = run_program()
    assert result['passed'], result
    assert result['elapsed_s'] == pytest.approx(30.) and result['samples'] == 15000
    assert min(result['foot_displacements_x_m']) >= .035
    assert max(result['foot_placement_errors_xy_m']) <= .005
    assert min(result['max_sole_clearance_m']) >= .015
    assert min(result['single_support_dwell_by_swing_foot_s']) >= .5
    assert result['max_saturated_actuators'] == 0 and not result['joint_limit_events']
    assert result['max_external_force_component_n'] == 0. and result['max_generalized_external_force'] == 0.
    assert not result['continuous_walking_verified'] and not result['hardware_verified']
