from copy import deepcopy
import numpy as np
import pytest
from humanoid1.learning import promotion_decision
from humanoid1.verification import CRITERIA


def report():
    jobs = [('nominal', 101), ('initial_velocity', 101), ('initial_velocity', 102),
            ('initial_velocity', 103), ('push_x', 101), ('push_y', 101)]
    return dict(schema='humanoid1-standing-evaluation-v1', evaluation_split='held_out',
        physics_timestep_s=.002, policy_period_s=.02, criteria=CRITERIA,
        balance_config={'residual_scale_nm': 2.}, source_sha256={'core.py': 'test'},
        model_sha256='model', policy_sha256=None,
        episodes=[dict(scenario=s, seed=n, stable=True, elapsed_s=30., total_return=29.9,
                       terminated=False, failed_criteria=[], joint_limit_events=[]) for s, n in jobs])


@pytest.mark.parametrize('fault', ['duplicate_baseline', 'nan_duration', 'nan_baseline_duration',
    'nan_return', 'validation', 'baseline_stopped', 'joint_event', 'regression', 'criteria_mismatch'])
def test_rejects_incomplete_or_inconsistent_promotion_evidence(fault):
    candidate, baseline = report(), report()
    if fault == 'duplicate_baseline':
        baseline['episodes'].append(deepcopy(baseline['episodes'][0]))
    elif fault == 'nan_duration':
        candidate['episodes'][0]['elapsed_s'] = np.nan
    elif fault == 'nan_baseline_duration':
        baseline['episodes'][0]['elapsed_s'] = np.nan
    elif fault == 'nan_return':
        candidate['episodes'][0]['total_return'] = np.nan
    elif fault == 'validation':
        candidate['evaluation_split'] = 'validation'
    elif fault == 'baseline_stopped':
        baseline['episodes'][0]['terminated'] = True
    elif fault == 'joint_event':
        candidate['episodes'][0]['joint_limit_events'] = [{'joint': 'left_knee'}]
    elif fault == 'regression':
        candidate['episodes'][0]['total_return'] -= .001
    else:
        candidate['criteria'] = {**CRITERIA, 'max_torso_tilt_rad': 1.}
    assert not promotion_decision(candidate, baseline)['eligible']


def test_identical_eligible_evidence_and_small_numerical_tolerance():
    candidate, baseline = report(), report()
    assert promotion_decision(candidate, baseline)['eligible']
    candidate['episodes'][0]['total_return'] -= .5e-6
    assert promotion_decision(candidate, baseline)['eligible']
