"""First local sensor-plan-act-remember integration slice for Humanoid-1.

The task planner is a deterministic bounded grammar, not a local generative
model. World observations are labeled MuJoCo ground truth, not visual inference.
No unsupported TOC capability is silently substituted with a scripted success.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT/'requirements'/'book_features.json'
MEMORY_TYPES = ('episodic', 'semantic', 'procedural', 'spatial', 'task')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10.)
        self.db.execute('CREATE TABLE IF NOT EXISTS memories (kind TEXT, key TEXT, payload TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL, PRIMARY KEY (kind,key))')
        self.db.commit()

    def put(self, kind, key, value):
        if kind not in MEMORY_TYPES or not isinstance(key, str) or not key:
            raise ValueError('invalid memory type/key')
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        now = utc_now()
        self.db.execute('INSERT INTO memories VALUES (?,?,?,?,?) ON CONFLICT(kind,key) DO UPDATE SET payload=excluded.payload,updated=excluded.updated',
                        (kind,key,encoded,now,now))
        self.db.commit()

    def get(self, kind, key):
        row = self.db.execute('SELECT payload FROM memories WHERE kind=? AND key=?',(kind,key)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind):
        return [dict(key=k, value=json.loads(v)) for k,v in self.db.execute('SELECT key,payload FROM memories WHERE kind=? ORDER BY updated',(kind,))]

    def close(self):
        self.db.close()


class UnsupportedCapability(ValueError):
    pass


class LocalRuntime:
    def __init__(self, directory='results/humanoid1'):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.memory = MemoryStore(self.directory/'memory.sqlite')
        self.catalog = json.loads(CATALOG.read_text())
        for name, reference in [('stand', 'evidence/v0.3/main/summary.json'),
                                ('weight_shift', 'evidence/v0.5/engineered/summary.json'),
                                ('two_step_advance', 'evidence/v0.6/engineered/summary.json')]:
            self.memory.put('procedural', name, dict(skill=name, controller='engineered',
                evidence=reference, execution='bounded MuJoCo simulation', hardware_verified=False))

    def status(self):
        latch = self.memory.get('semantic','safety_latch') or dict(latched=False)
        return dict(project='Humanoid-1', overall_complete=False, simulation_only=True,
            planner='deterministic_bounded_grammar', generative_local_brain_verified=False,
            executable_skills=['stand','weight_shift','two_step_advance'], motion_safety_latch=latch,
            toc_chapters=self.catalog['chapter_count'], toc_entries=self.catalog['toc_section_count'],
            complete_chapters=sum(c['complete'] for c in self.catalog['capabilities']),
            capabilities=[{k:c[k] for k in ('id','title','status','complete','evidence_boundary')}
                          for c in self.catalog['capabilities']],
            learned_policy_active=False, memory_types=MEMORY_TYPES,
            latest_world_observation=self.memory.get('spatial','last_simulation_observation'))

    def plan(self, goal, seconds=30., seed=0):
        if not isinstance(goal,str) or len(goal)>1000:
            raise ValueError('goal must be a short text instruction')
        text=goal.lower().strip()
        if re.search(r'\b(walk|run|jump|climb|crawl|grasp|pick|smell|taste|consciousness|surgery|implant)\b', text):
            raise UnsupportedCapability('Requested capability is not verified or executable; see requirements/book_features.json')
        if text in ('take two steps', 'demonstrate stepping', 'two step advance'):
            skill='two_step_advance'
            if not isinstance(seconds,(int,float)) or not 30.<=float(seconds)<=60.:
                raise ValueError('the verified two-step program requires 30 to 60 seconds')
        elif ('weight' in text and ('shift' in text or 'shifting' in text)) or text in ('shift','balance left and right'):
            skill='weight_shift'
        elif text in ('stand','stand upright','hold standing'):
            skill='stand'
        else:
            raise UnsupportedCapability('Supported motion requests: stand; demonstrate weight shifting; take two steps. No action was taken.')
        if not isinstance(seconds,(int,float)) or not .1<float(seconds)<=60.:
            raise ValueError('simulation duration must be in (0.1,60] seconds')
        if not isinstance(seed,int):
            raise ValueError('seed must be an integer')
        return dict(goal=goal, planner='deterministic_bounded_grammar',
            steps=[dict(skill=skill, seconds=float(seconds), seed=seed)],
            contract=dict(no_direct_torque_from_planner=True, independent_safety=True,
                          simulation_only=True, learned_policy=False))

    def emergency_stop(self, reason='operator emergency stop'):
        old=self.memory.get('semantic','safety_latch') or {}
        generation=int(old.get('generation',0))+1
        self.memory.put('semantic','safety_latch',dict(latched=True,generation=generation,reason=reason,time=utc_now()))
        # Running control loops poll this persisted latch at physics substeps,
        # including a stop written by a second CLI process.
        return dict(latched=True,reason=reason)

    def reset_stop(self):
        old=self.memory.get('semantic','safety_latch') or {}
        self.memory.put('semantic','safety_latch',dict(latched=False,generation=int(old.get('generation',0)),reset_at=utc_now()))
        return dict(latched=False,scope='fresh simulation episodes only')

    def stop_requested_since(self, generation):
        current=self.memory.get('semantic','safety_latch') or {}
        return bool(current.get('latched') or int(current.get('generation',0))!=generation)

    def execute(self, goal, seconds=30., seed=0):
        latch=self.memory.get('semantic','safety_latch')
        if latch and latch['latched']:
            raise RuntimeError('motion is latched off; explicitly reset the simulation stop first')
        initial_stop_generation=int((latch or {}).get('generation',0))
        plan=self.plan(goal,seconds,seed)
        task_id=uuid.uuid4().hex
        record=dict(task_id=task_id,goal=goal,status='running',plan=plan,started_at=utc_now())
        self.memory.put('task',task_id,record)
        self.memory.put('episodic',task_id+':planned',dict(task_id=task_id,event='planned',plan=plan))
        step=plan['steps'][0]; log=self.directory/'episodes'/(task_id+'.csv')
        def stop_check():
            return self.stop_requested_since(initial_stop_generation)
        try:
            if step['skill']=='two_step_advance':
                from .stepping import run_program, StepProgram
                evidence=run_program(StepProgram(seed=seed),log=log,seconds=seconds,stop_check=stop_check)
                success=evidence['passed']
                world=dict(source='mujoco_ground_truth',timestamp=utc_now(),simulation_time_s=evidence['elapsed_s'],
                    task_id=task_id,reference_frame='flat-floor episode start',
                    foot_order=evidence['foot_order'],foot_displacements_x_m=evidence['foot_displacements_x_m'],
                    max_sole_clearance_m=evidence['max_sole_clearance_m'],
                    single_support_dwell_by_swing_foot_s=evidence['single_support_dwell_by_swing_foot_s'],
                    final_com_error_m=evidence['final_com_error_m'],continuous_walking_verified=False)
            elif step['skill']=='weight_shift':
                from .weight_shift import run_program, ShiftProgram
                evidence=run_program(ShiftProgram(seed=seed),log=log,seconds=seconds,stop_check=stop_check)
                success=evidence['passed']
                world=dict(source='mujoco_ground_truth',timestamp=utc_now(),simulation_time_s=evidence['elapsed_s'],
                    task_id=task_id,reference_frame='flat-floor standing home',
                    com_y_range_m=evidence['com_y_range_m'],left_hold_load_fraction=evidence['left_hold_load_fraction'],
                    right_hold_load_fraction=evidence['right_hold_load_fraction'],
                    final_center_error_m=evidence['final_center_error_m'])
            else:
                from .verification import run_trial
                evidence=run_trial(seconds=seconds,log=log,stop_check=stop_check)
                success=evidence['stable']
                world=dict(source='mujoco_ground_truth',timestamp=utc_now(),simulation_time_s=evidence['elapsed_s'],
                    task_id=task_id,observation=evidence['final'])
            report_path=self.directory/'episodes'/(task_id+'.json')
            report_path.parent.mkdir(parents=True,exist_ok=True)
            report_path.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
            record.update(status='succeeded' if success else 'failed',finished_at=utc_now(),
                evidence=str(report_path.resolve()), evidence_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                actual_simulation_seconds=evidence['elapsed_s'],simulation_only=True)
            if evidence['stopped']:
                self.emergency_stop(evidence.get('reason') or 'motion safety stop')
            self.memory.put('spatial','last_simulation_observation',world)
            self.memory.put('episodic',task_id+':result',dict(task_id=task_id,event=record['status'],world=world))
            self.memory.put('task',task_id,record)
            return record
        except Exception as exc:
            record.update(status='failed',finished_at=utc_now(),error=str(exc))
            self.memory.put('task',task_id,record)
            self.emergency_stop('runtime execution failure')
            raise

    def recall(self, task_id):
        record=self.memory.get('task',task_id)
        if record is None:
            raise KeyError('task not found')
        path=Path(record.get('evidence',''))
        if record.get('evidence_sha256'):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['evidence_sha256']:
                raise ValueError('task evidence is missing or its hash changed')
        return record

    def close(self):
        self.memory.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir',default='results/humanoid1')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('status');sub.add_parser('stop');sub.add_parser('reset-stop')
    execute=sub.add_parser('execute');execute.add_argument('goal');execute.add_argument('--seconds',type=float,default=30.)
    execute.add_argument('--seed',type=int,default=0)
    recall=sub.add_parser('recall');recall.add_argument('task_id')
    remember=sub.add_parser('remember');remember.add_argument('key');remember.add_argument('value')
    recall_memory=sub.add_parser('recall-memory');recall_memory.add_argument('key')
    sub.add_parser('demo')
    a=p.parse_args();runtime=LocalRuntime(a.state_dir)
    try:
        if a.command=='status': result=runtime.status()
        elif a.command=='execute': result=runtime.execute(a.goal,a.seconds,a.seed)
        elif a.command=='recall': result=runtime.recall(a.task_id)
        elif a.command=='stop': result=runtime.emergency_stop()
        elif a.command=='reset-stop': result=runtime.reset_stop()
        elif a.command=='remember':
            if a.key=='safety_latch':
                raise ValueError('safety state is reserved; use stop/reset-stop')
            runtime.memory.put('semantic',a.key,dict(text=a.value,source='explicit_user_instruction'))
            result=dict(remembered=a.key)
        elif a.command=='recall-memory': result=runtime.memory.get('semantic',a.key)
        else:
            result=runtime.execute('demonstrate weight shifting')
            # A real second process reads the committed SQLite/evidence state.
            other=subprocess.run([sys.executable,'-m','humanoid1.runtime','--state-dir',str(runtime.directory.resolve()),
                'recall',result['task_id']],capture_output=True,text=True,check=True)
            restored=json.loads(other.stdout)
            if restored != result:
                raise RuntimeError('restart recall differs from recorded task')
            result=dict(task=result,recall_after_process_restart=True,
                integration=['bounded_plan','physics_control','ground_truth_observation','persistent_memory','restart_recall'],
                complete_humanoid=False)
        print(json.dumps(result,indent=2,allow_nan=False))
    finally:
        runtime.close()


if __name__=='__main__':
    main()
