import json
import numpy as np
import pytest
from humanoid1.commanded_walking import CommandProgram, CommandGaitPlan, DIRECTIONS, run_program


@pytest.mark.parametrize('direction', DIRECTIONS)
@pytest.mark.parametrize('distance', [.20, .22, .40, .45])
def test_distance_endpoint_continuity_and_interrupted_plan(direction, distance):
    program = CommandProgram(distance, direction)
    plan = CommandGaitPlan(program.specification())
    assert 6 <= len(plan.steps) <= 12 and plan.minimum_seconds <= 120.
    final = plan.reference(plan.minimum_seconds)
    assert np.allclose(final['com'][:2], distance*np.asarray(DIRECTIONS[direction]))
    assert np.allclose(final['com'][:2], final['feet'][:, :2].mean(axis=0))
    assert np.allclose(final['feet'][:, 2], 0.)
    for phase in plan.phases[:-1]:
        before, after = plan.reference(phase.end-1e-8), plan.reference(phase.end+1e-8)
        for key in ('com', 'com_velocity', 'com_acceleration', 'feet', 'foot_velocity', 'foot_acceleration'):
            assert np.allclose(before[key], after[key], atol=1e-6)
    before = [plan.reference(t) for t in np.linspace(0., 8.8, 80)]
    assert plan.request_stop(8.8) and plan.active_step_count == 1
    for t, original in zip(np.linspace(0., 8.8, 80), before):
        assert np.allclose(plan.reference(t)['com'], original['com'])
        assert np.allclose(plan.reference(t)['feet'], original['feet'])
    final = plan.reference(plan.phases[-1].end)
    assert np.allclose(final['com'][:2], final['feet'][:, :2].mean(axis=0))
    assert np.allclose(final['feet'][:, 2], 0.)


@pytest.mark.parametrize('distance', [True, False, None, '.22', np.nan, np.inf, -.22, .1999, .4501])
def test_out_of_envelope_commands_fail_before_motion(distance):
    with pytest.raises(ValueError):
        run_program(CommandProgram(distance))


@pytest.mark.parametrize('direction', [None, True, [], 'turn left', 'north'])
def test_unsupported_direction_rejected_before_motion(direction):
    with pytest.raises(ValueError):
        run_program(CommandProgram(direction=direction))


def test_short_episode_and_emergency_stop_do_not_claim_command_success():
    result = run_program(CommandProgram(direction='backward'), seconds=.2)
    assert not result['passed'] and not result['goal_completed'] and result['completed_steps'] == 0
    result = run_program(CommandProgram(direction='left'), stop_check=lambda: True,
        controlled_stop_check=lambda: True)
    assert result['stopped'] and result['elapsed_s'] == 0. and not result['passed']
    assert result['stop_requested_at_s'] is None
    json.dumps(result, allow_nan=False)
