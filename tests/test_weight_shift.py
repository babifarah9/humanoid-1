import json
from pathlib import Path
import numpy as np
import pytest
from humanoid1.weight_shift import ShiftProgram,ShiftTask,smooth_reference,run_program
from humanoid1.verification import run_trial
from humanoid1.core import Humanoid
from humanoid1.whole_body import WholeBodyController


def test_reference_is_bounded_smooth_and_returns_home():
    segments,_,_=ShiftProgram(101,True,True).build()
    values=np.array([smooth_reference(t,segments) for t in np.linspace(0,30,3001)])
    assert np.max(np.abs(values[:,0]))<=.02
    assert np.allclose(smooth_reference(30,segments),0.)
    for start,end,a,b in segments:
        assert np.allclose(smooth_reference(start,segments)[1:],0,atol=1e-10)
        assert np.allclose(smooth_reference(end,segments)[1:],0,atol=1e-10)
    assert ShiftProgram(101,True,True).build()==ShiftProgram(101,True,True).build()
    assert ShiftProgram(101,True,True).build()!=ShiftProgram(102,True,True).build()


def test_standing_default_com_task_remains_identical():
    r=Humanoid();c=WholeBodyController(r)
    j=np.arange(3*r.model.nv).reshape(3,-1)*.001
    r.data.qvel[:]=np.linspace(-.02,.02,r.model.nv)
    expected=c.config.com_kp*(c.target_com-r.data.subtree_com[c.pelvis])-c.config.com_kd*j@r.data.qvel
    assert np.array_equal(c.com_task_acceleration(j),expected)


def test_operator_stop_is_polled_and_short_trial_cannot_claim_task_success():
    task=ShiftTask(seconds=.2)
    while not task.done:task.step()
    assert not task.r.stopped and not task.summary()['passed']
    calls=[0]
    def stop():
        calls[0]+=1
        return calls[0]>5
    result=run_program(ShiftProgram(),seconds=30,stop_check=stop)
    assert result['stopped'] and result['elapsed_s']==pytest.approx(.01)
    assert result['reason']=='operator runtime stop'
    stand=run_trial(seconds=30,stop_check=lambda:True)
    assert stand['elapsed_s']==0 and stand['stopped'] and not stand['stable']
    json.dumps(stand,allow_nan=False)


def test_shift_invalid_policy_action_latches_stop():
    task=ShiftTask()
    _,_,terminated,_,_=task.step(np.full(16,np.nan))
    assert terminated and task.r.data.time==0
    assert task.r.stopped
