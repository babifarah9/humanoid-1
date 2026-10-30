"""Publish completed experiment evidence with exact 50 Hz actions/rewards/times.

Observation checkpoints are retained at 10 Hz to limit repository size. The
full 50 Hz observation archives regenerate from scripts/run_ppo_experiment.py.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('destination')
    a = p.parse_args()
    source, dest = Path(a.source), Path(a.destination)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('experiment_plan.json', 'selection.json', 'decision.json'):
        shutil.copyfile(source/name, dest/name)
    selection = json.loads((source/'selection.json').read_text())
    for seed in (0, 1):
        for suffix in ('.zip', '.json', '_episodes.json'):
            shutil.copyfile(source/f'seed{seed}{suffix}', dest/f'seed{seed}{suffix}')
        shutil.copyfile(source/f'seed{seed}_logs'/'progress.csv', dest/f'seed{seed}_progress.csv')
        report = json.loads((source/f'validation_seed{seed}'/'evaluation.json').read_text())
        (dest/f'validation_seed{seed}.json').write_text(json.dumps(report, indent=2)+'\n')
    selected = source/f"seed{selection['selected_seed']}.zip"
    shutil.copyfile(selected, dest/'selected_policy.zip')
    manifest = dict(observation_rate_hz=10, action_reward_time_rate_hz=50,
        note='Observation indices refer to original 50 Hz transitions. All actions/rewards/times are retained; full observations regenerate using source and seeds. No training replay is fed into PPO.',
        policy_sha256=sha(selected), episodes=[])
    with zipfile.ZipFile(dest/'experience.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
        for label in ('baseline', 'selected_held_out'):
            directory = source/label
            report = json.loads((directory/'evaluation.json').read_text())
            (dest/f'{label}.json').write_text(json.dumps(report, indent=2)+'\n')
            for episode in report['episodes']:
                path = directory/episode['trace']
                assert sha(path) == episode['trace_sha256']
                with np.load(path, allow_pickle=False) as trace:
                    n = len(trace['actions'])
                    indices = np.unique(np.r_[np.arange(0, n+1, 5), n]).astype(np.int64)
                    stream = io.BytesIO()
                    np.savez_compressed(stream, observations_10hz=trace['observations'][indices],
                        observation_indices=indices, actions=trace['actions'], rewards=trace['rewards'], time_s=trace['time_s'])
                    item = dict(split=label, scenario=episode['scenario'], seed=episode['seed'],
                        policy_steps=n, observation_checkpoints=len(indices), original_trace_sha256=sha(path),
                        archive_path=label+'/'+path.name, archive_entry_sha256=hashlib.sha256(stream.getvalue()).hexdigest())
                archive.writestr(item['archive_path'], stream.getvalue())
                manifest['episodes'].append(item)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2)+'\n')
    (dest/'experience_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    files = {str(f.relative_to(dest)): sha(f) for f in sorted(dest.rglob('*')) if f.is_file() and f.name != 'SHA256SUMS.json'}
    (dest/'SHA256SUMS.json').write_text(json.dumps(files, indent=2)+'\n')
    print('Published bytes:', sum(f.stat().st_size for f in dest.iterdir() if f.is_file()))


if __name__ == '__main__':
    main()
