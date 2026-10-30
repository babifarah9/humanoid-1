"""Record real runtime execution, optional cross-process stop, and restart recall."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid1.runtime import LocalRuntime
from humanoid1.walking import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/v07/runtime')
    parser.add_argument('--controlled-stop', action='store_true')
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    sources = source_fingerprint()
    state = output/'state'; observer = LocalRuntime(state)
    process = subprocess.Popen([sys.executable, '-m', 'humanoid1.runtime', '--state-dir', str(state),
        'execute', 'walk forward'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    observation = None; request = None
    try:
        if args.controlled_stop:
            deadline = time.monotonic()+120.
            while time.monotonic() < deadline:
                tasks = observer.memory.list('task')
                record = tasks[-1]['value'] if tasks else {}
                progress = record.get('progress', {})
                if (record.get('status') == 'running' and progress.get('gait_phase') == 'swing_hold'
                        and progress.get('gait_step') == 1 and progress['right_sole_clearance_m'] >= .015
                        and progress['right_normal_n'] < 1.):
                    observation = progress
                    break
                if process.poll() is not None:
                    raise RuntimeError('worker ended before a measured airborne step')
                time.sleep(.03)
            else:
                raise RuntimeError('no measured airborne step observed')
            request = json.loads(subprocess.check_output([sys.executable, '-m', 'humanoid1.runtime',
                '--state-dir', str(state), 'stop-walking', '--task-id', record['task_id']], text=True))
            if not request['accepted']:
                raise RuntimeError('controlled stop request was not accepted')
        text, error = process.communicate(timeout=180.)
        if process.returncode:
            raise RuntimeError(error)
        task = json.loads(text); episode = Path(task['evidence'])
        evidence = json.loads(episode.read_text())
        if not evidence['passed'] or evidence['stopped']:
            raise RuntimeError('runtime motion did not pass acceptance')
        expected_status = 'controlled_stop' if args.controlled_stop else 'succeeded'
        if task['status'] != expected_status or task['goal_completed'] != (not args.controlled_stop):
            raise RuntimeError('runtime outcome misrepresents goal completion')
        recalled = json.loads(subprocess.check_output([sys.executable, '-m', 'humanoid1.runtime',
            '--state-dir', str(state), 'recall', task['task_id']], text=True))
        if recalled != task or hashlib.sha256(episode.read_bytes()).hexdigest() != task['evidence_sha256']:
            raise RuntimeError('task/evidence restart recall mismatch')
        if args.controlled_stop and (evidence['completed_steps'] != 1 or not evidence['controlled_stop_verified']):
            raise RuntimeError('airborne stop did not complete exactly the current step')
        if sources != source_fingerprint():
            raise RuntimeError('controller sources changed during runtime validation')
        validation = dict(passed=True, task=task, observed_airborne_state_at_request=observation,
            controlled_stop_request=request, recall_after_process_restart=True,
            emergency_latched=observer.status()['motion_safety_latch']['latched'], source_sha256=sources)
        (output/'runtime_task.json').write_text(json.dumps(task, indent=2)+'\n')
        (output/'runtime_episode.json').write_bytes(episode.read_bytes())
        (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        print(json.dumps(dict(passed=True,status=task['status'],elapsed_s=evidence['elapsed_s'],
            completed_steps=evidence['completed_steps'],recall_after_process_restart=True)))
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        observer.close()


if __name__ == '__main__':
    main()
