import numpy as np
import pytest
import mujoco
from humanoid1.core import Humanoid
from humanoid1.whole_body import WholeBodyController, full_mass
from humanoid1.verification import run_trial
from humanoid1.learning import promotion_decision


def test_model_and_compensation():
    r = Humanoid(); c = WholeBodyController(r)
    assert r.nu == 16
    assert r.model.body_mass.sum() == pytest.approx(32.8)
    assert r.model.neq == 0  # no weld or equality support
    assert np.linalg.eigvalsh(full_mass(r.model, r.data)).min() > 0
    tau = c.act()
    assert np.isfinite(tau).all()
    assert not r.data.qfrc_applied.any() and not r.data.xfrc_applied.any()
    assert c.last['root_dynamics_residual'] < 1e-8


@pytest.mark.parametrize('fault', ['position', 'velocity', 'height', 'nan'])
def test_safety_latch_and_reset(fault):
    r = Humanoid()
    if fault == 'position':
        r.data.qpos[r.qadr[0]] = r.limits[0, 1]+.071
    elif fault == 'velocity':
        r.data.qvel[r.vadr[0]] = 91.
    elif fault == 'height':
        r.data.qpos[2] = .59
    else:
        r.data.qpos[0] = np.nan
    before = r.data.time
    _, info = r.step(np.ones(r.nu))
    assert info['stopped'] and r.data.time == before
    assert not r.data.ctrl.any()
    snapshot = r.stop_snapshot
    r.step(np.zeros(r.nu))
    assert r.data.time == before and r.stop_snapshot is snapshot
    r.reset()
    assert not r.stopped and not r.limit_events and r.stop_snapshot is None


def test_last_substep_stop_and_joint_snapshot():
    r = Humanoid()
    # Cross the safety tolerance during exactly one final substep.
    r.data.qpos[r.qadr[0]] = r.limits[0, 1]+.069
    r.data.qvel[r.vadr[0]] = 5.
    _, info = r.step(np.full(r.nu, 1000.), frames=1)
    assert info['stopped']
    ev = r.stop_snapshot['safety_joint_violations'][0]
    assert ev['joint'] == 'left_hip_pitch'
    assert ev['time_s'] == pytest.approx(.002)
    assert ev['position_rad'] > r.limits[0, 1]+.07
    assert abs(ev['torque_nm']) <= 100.
    assert ev['requested_torque_nm'] == 1000.
    assert ev['command_nm'] == 100.
    assert not r.data.ctrl.any()


def test_contact_and_external_estop():
    r = Humanoid(); c = WholeBodyController(r)
    r.data.qpos[2] += .5
    r.data.time = .2
    assert not c.act().any() and r.stopped
    assert r.stop_reason == 'bilateral ground support lost'
    r.reset(); c.reset()
    r.emergency_stop('operator stop')
    c.act(); r.step(np.ones(r.nu))
    assert r.data.time == 0. and r.stop_reason == 'operator stop'


def test_invalid_residual_latches():
    r = Humanoid(); c = WholeBodyController(r)
    assert not c.act(np.full(r.nu, np.nan)).any()
    assert r.stopped


def test_standing_envelope_short():
    result = run_trial(seconds=1.)
    assert result['stable'] and not result['achieved_30s']
    assert not result['joint_limit_events']
    assert result['metrics']['min_support_margin_m'] >= .02
    assert result['max_external_force_component_n'] == 0.


def test_promotion_rejects_incomplete_or_failed_evidence():
    assert not promotion_decision({}, {})['eligible']
    jobs = [('nominal', 101), ('initial_velocity', 101), ('initial_velocity', 102),
            ('initial_velocity', 103), ('push_x', 101), ('push_y', 101)]
    report = dict(schema='humanoid1-standing-evaluation-v1', evaluation_split='held_out',
        physics_timestep_s=.002, policy_period_s=.02,
        criteria={}, balance_config={}, source_sha256={'core.py': 'abc'},
        model_sha256='abc', episodes=[dict(scenario=s, seed=n, stable=True,
        elapsed_s=30., total_return=0., terminated=False, joint_limit_events=[]) for s, n in jobs])
    assert promotion_decision(report, report)['eligible']
    report['episodes'][0]['stable'] = False
    assert not promotion_decision(report, report)['eligible']
