import json
import subprocess
import sys
from pathlib import Path
import pytest
from humanoid1.runtime import LocalRuntime,MemoryStore,UnsupportedCapability,CATALOG


def test_every_toc_entry_is_registered_without_false_completion():
    catalog=json.loads(CATALOG.read_text())
    assert len(catalog['capabilities'])==45
    entries=[s['id'] for c in catalog['capabilities'] for s in c['sections']]+[s['id'] for s in catalog['appendices']]
    assert len(entries)==587 and len(set(entries))==587
    assert not catalog['overall_complete']
    assert all(c['acceptance'] and c['evidence_boundary'] and not c['complete'] for c in catalog['capabilities'])


def test_memory_and_stop_survive_new_process(tmp_path):
    runtime=LocalRuntime(tmp_path)
    runtime.memory.put('semantic','project',dict(name='Humanoid-1'))
    runtime.emergency_stop('test operator')
    runtime.close()
    other=LocalRuntime(tmp_path)
    with pytest.raises(RuntimeError):other.execute('stand',seconds=.2)
    other.reset_stop()
    result=other.execute('stand',seconds=.2)
    other.close()
    proc=subprocess.run([sys.executable,'-m','humanoid1.runtime','--state-dir',str(tmp_path),
        'recall',result['task_id']],capture_output=True,text=True,check=True)
    assert json.loads(proc.stdout)==result
    assert result['status']=='succeeded' and result['actual_simulation_seconds']==pytest.approx(.2)
    db=MemoryStore(tmp_path/'memory.sqlite')
    assert db.get('semantic','project')==dict(name='Humanoid-1')
    assert db.get('spatial','last_simulation_observation')['source']=='mujoco_ground_truth'
    db.close()


def test_unsupported_requests_take_no_motion_action(tmp_path):
    r=LocalRuntime(tmp_path)
    for command in ['walk','pick up cup','smell the kitchen','run a shell command']:
        with pytest.raises(UnsupportedCapability):r.execute(command)
    assert not r.memory.list('task')
    assert not r.status()['generative_local_brain_verified']
    assert not r.status()['learned_policy_active']
    r.close()


def test_memory_recall_detects_changed_evidence(tmp_path):
    r=LocalRuntime(tmp_path)
    task=r.execute('stand',seconds=.2)
    Path(task['evidence']).write_text('{}')
    with pytest.raises(ValueError):r.recall(task['task_id'])
    r.close()


def test_quick_reset_cannot_hide_stop_from_already_running_motion(tmp_path):
    r=LocalRuntime(tmp_path)
    assert not r.stop_requested_since(0)
    r.emergency_stop('operator')
    r.reset_stop()
    assert r.stop_requested_since(0)
    assert not r.stop_requested_since(1)
    r.close()


@pytest.mark.parametrize('goal', ['stand', 'take two steps'])
def test_operator_stop_from_another_process_interrupts_motion(tmp_path, goal):
    import time
    proc=subprocess.Popen([sys.executable,'-m','humanoid1.runtime','--state-dir',str(tmp_path),
        'execute',goal,'--seconds','30'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    observer=LocalRuntime(tmp_path)
    try:
        for _ in range(250):
            if observer.memory.list('task'):break
            time.sleep(.02)
        else:raise AssertionError('worker never started task')
        time.sleep(.15)
        observer.emergency_stop('test external operator')
        output,error=proc.communicate(timeout=20)
        assert proc.returncode==0,error
        result=json.loads(output)
        assert result['status']=='failed' and result['actual_simulation_seconds']<30.
        evidence=json.loads(Path(result['evidence']).read_text())
        assert evidence['reason']=='operator runtime stop' and evidence['stopped']
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        observer.close()
