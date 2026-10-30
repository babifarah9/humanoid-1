from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from humanoid1.weight_shift_learning import shift_promotion

ROOT=Path(__file__).resolve().parents[1]


def fixtures():
    baseline=json.loads((ROOT/'evidence/v0.5/engineered/summary.json').read_text())
    candidate=deepcopy(baseline);candidate['policy_sha256']='unit-test-only'
    for r in candidate['results']:r['rms_tracking_xy_m']*=.9
    return candidate,baseline


def test_gate_accepts_a_matched_improved_fixture_but_rejects_actual_policy():
    candidate,baseline=fixtures()
    assert shift_promotion(candidate,baseline)['eligible']
    actual=json.loads((ROOT/'evidence/v0.5/ppo/summary.json').read_text())
    assert not shift_promotion(actual,baseline)['eligible']


@pytest.mark.parametrize('fault',['nan_metric','duplicate','missing_case','changed_command','unsafe','short','work','source'])
def test_shift_gate_rejects_invalid_or_regressed_evidence(fault):
    candidate,baseline=fixtures();r=candidate['results'][0]
    if fault=='nan_metric':r['rms_tracking_xy_m']=np.nan
    elif fault=='duplicate':candidate['results'].append(deepcopy(r))
    elif fault=='missing_case':candidate['results'].pop()
    elif fault=='changed_command':r['segments'][0][3]=.03
    elif fault=='unsafe':r['stopped']=True
    elif fault=='short':r['elapsed_s']=29.
    elif fault=='work':r['mechanical_absolute_work_j']*=1.5
    else:candidate['source_sha256']={'wrong':'source'}
    assert not shift_promotion(candidate,baseline)['eligible']
