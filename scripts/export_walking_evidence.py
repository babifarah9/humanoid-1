"""Export complete review traces from finished walking/controlled-stop trials."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('destination')
    args = parser.parse_args(); source, dest = Path(args.source), Path(args.destination)
    dest.mkdir(parents=True, exist_ok=True)
    report = json.loads((source/'summary.json').read_text())
    fields = ['trial', 'time_s', 'gait_phase', 'gait_step', 'completed_steps', 'body_progress_m',
        'controlled_stop_requested', 'standing_dwell_s', 'planned_support', 'com_x', 'com_y', 'com_z',
        'tracking_com_m', 'tracking_foot_m', 'torso_tilt_rad', 'pelvis_z',
        'left_sole_clearance_m', 'right_sole_clearance_m', 'left_foot_x_m', 'right_foot_x_m',
        'left_normal_n', 'right_normal_n', 'support_margin_m', 'joint_margin_rad', 'stance_drift_m',
        'max_applied_torque_nm', 'root_balance_residual', 'predicted_wrench_bound_margin', 'stopped', 'reason']
    counts = {}
    with (dest/'trace_10hz.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for result in report['results']:
            raw = Path(result['log']); count = 0; selected = []
            if raw.stem in counts:
                raise ValueError('duplicate trial label: '+raw.stem)
            with raw.open(newline='') as f:
                for index, row in enumerate(csv.DictReader(f)):
                    count += 1
                    if index % 50 == 0:
                        selected.append(row)
                    last = row
            if not count or count != result['samples'] or abs(float(last['time_s'])-result['elapsed_s']) > 1e-8:
                raise ValueError('raw physics trace incomplete: '+str(raw))
            if selected[-1] is not last:
                selected.append(last)
            for row in selected:
                writer.writerow(dict(trial=raw.stem, **{key:row[key] for key in fields if key != 'trial'}))
            counts[raw.stem] = len(selected)
            result['raw_csv_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest(); result['log'] = raw.name
    with (dest/'trace_10hz.csv').open(newline='') as f:
        published = list(csv.DictReader(f))
    if sum(counts.values()) != len(published) or any(
            sum(row['trial'] == trial for row in published) != count for trial,count in counts.items()):
        raise ValueError('review export incomplete')
    report['trace_note'] = 'Complete 10 Hz review traces; all safety/step/stop criteria evaluated at 2 ms. Full joint CSVs regenerate locally.'
    (dest/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(verified_rows=counts)))


if __name__ == '__main__':
    main()
