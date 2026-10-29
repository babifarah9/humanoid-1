"""Run a bounded, reproducible two-candidate CPU PPO standing experiment.

python scripts/run_ppo_experiment.py --output results/v04
Use --reuse-training only with hash-matched completed checkpoints. Validation
uses separate seeds. Exactly one selected policy sees the held-out suite.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from humanoid1.learning import promotion_decision
from humanoid1.core import MODEL


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log):
    print('RUN', ' '.join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        subprocess.run(command, cwd=ROOT, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1',
                       'OMP_NUM_THREADS': '1'}, stdout=f, stderr=subprocess.STDOUT, check=True)


def read_report(directory):
    report = json.loads((directory/'evaluation.json').read_text())
    expected = {name: digest(ROOT/'humanoid1'/name)
                for name in ('core.py', 'whole_body.py', 'ppo_env.py', 'verification.py')}
    if report.get('source_sha256') != expected or report.get('model_sha256') != digest(MODEL):
        raise ValueError('evaluation source/model hash mismatch')
    for episode in report['episodes']:
        if digest(directory/episode['trace']) != episode['trace_sha256']:
            raise ValueError('evaluation trace hash mismatch')
    return report


def completed_training(prefix, seed, steps, log_std):
    record = json.loads(Path(str(prefix)+'.json').read_text())
    if (record.get('seed') != seed or record.get('requested_timesteps') != steps
            or record.get('log_std_init') != log_std or not record.get('training_completed')
            or record.get('actual_timesteps', 0) < steps
            or record.get('model_sha256') != digest(MODEL)
            or record.get('policy_sha256') != digest(Path(str(prefix)+'.zip'))):
        raise ValueError('training checkpoint metadata mismatch')
    for name in ('learning.py', 'core.py', 'whole_body.py', 'ppo_env.py', 'verification.py'):
        if record.get('source_sha256', {}).get(name) != digest(ROOT/'humanoid1'/name):
            raise ValueError('training source hash mismatch: '+name)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='results/v04')
    p.add_argument('--policy-steps', type=int, default=16384)
    p.add_argument('--reuse-training', action='store_true')
    p.add_argument('--reuse-baseline', action='store_true')
    p.add_argument('--promote', action='store_true', help='record skill only if the unchanged promotion gate passes')
    a = p.parse_args()
    if a.policy_steps < 256:
        p.error('policy-steps must be at least one PPO rollout (256)')
    output = Path(a.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    configs = [(0, -3.), (1, -4.)]
    plan = dict(training=[dict(seed=s, log_std_init=std, policy_steps=a.policy_steps) for s, std in configs],
        validation=dict(seconds=10., jobs=[['nominal', 11], ['initial_velocity', 12],
            ['initial_velocity', 13], ['push_x', 14], ['push_y', 15]]),
        selection='Among candidates with all validation episodes stable, highest mean validation return. If none qualify, choose longest mean duration for final failure diagnosis, and forbid promotion.',
        held_out='Only the selected candidate receives the unchanged six-episode >=30s promotion test.',
        safety='Unchanged model, whole-body controller, torque bounds, envelope and emergency checks.',
        source_sha256={str(f.relative_to(ROOT)): digest(f) for f in (ROOT/'humanoid1').glob('*.py')})
    plan_file = output/'experiment_plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError('existing experiment plan differs; use a new output directory')
    plan_file.write_text(json.dumps(plan, indent=2)+'\n')
    training = []
    for seed, log_std in configs:
        prefix = output/f'seed{seed}'
        if not a.reuse_training:
            run([sys.executable, '-m', 'humanoid1.learning', 'train', '--timesteps', str(a.policy_steps),
                 '--seed', str(seed), '--log-std-init', str(log_std), '--output', str(prefix)], output/f'train_seed{seed}.log')
        training.append(completed_training(prefix, seed, a.policy_steps, log_std))
    validation = []
    for seed, _ in configs:
        directory = output/f'validation_seed{seed}'
        run([sys.executable, '-m', 'humanoid1.learning', 'evaluate', '--policy', str(output/f'seed{seed}.zip'),
             '--validation', '--seconds', '10', '--output', str(directory)], output/f'validation_seed{seed}.log')
        report = read_report(directory)
        if report.get('evaluation_split') != 'validation':
            raise ValueError('wrong candidate selection split')
        episodes = report['episodes']
        validation.append(dict(seed=seed, all_stable=all(e['stable'] for e in episodes),
            mean_return=sum(e['total_return'] for e in episodes)/len(episodes),
            mean_elapsed_s=sum(e['elapsed_s'] for e in episodes)/len(episodes),
            policy_sha256=report['policy_sha256']))
    eligible = [row for row in validation if row['all_stable']]
    selected = max(eligible, key=lambda x: x['mean_return']) if eligible else max(validation, key=lambda x: x['mean_elapsed_s'])
    selection = dict(candidates=validation, selected_seed=selected['seed'], selection_qualified=bool(eligible))
    (output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    # Freeze selection BEFORE running the held-out suite.
    if not a.reuse_baseline:
        run([sys.executable, '-m', 'humanoid1.learning', 'evaluate', '--output', str(output/'baseline')], output/'baseline.log')
    baseline = read_report(output/'baseline')
    candidate_dir = output/'selected_held_out'
    selected_policy = output/f"seed{selected['seed']}.zip"
    run([sys.executable, '-m', 'humanoid1.learning', 'evaluate', '--policy', str(selected_policy),
         '--output', str(candidate_dir)], output/'selected_held_out.log')
    candidate = read_report(candidate_dir)
    if candidate.get('policy_sha256') != selected['policy_sha256']:
        raise ValueError('selected policy identity changed')
    decision = promotion_decision(candidate, baseline)
    if not eligible:
        decision = dict(eligible=False, reasons=decision['reasons']+['no candidate passed validation'])
    (output/'decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    if a.promote and decision['eligible']:
        run([sys.executable, '-m', 'humanoid1.learning', 'promote', '--policy', str(selected_policy),
             '--candidate-report', str(candidate_dir/'evaluation.json'), '--baseline-report',
             str(output/'baseline'/'evaluation.json'), '--registry', str(output/'skills.json')], output/'promotion.log')
    print(json.dumps(dict(selection=selection, decision=decision,
                         promoted=bool(a.promote and decision['eligible'])), indent=2))


if __name__ == '__main__':
    main()
