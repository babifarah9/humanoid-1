"""Save compact review evidence while retaining hashes of full raw rollout CSVs.

Usage: python scripts/export_evidence.py results/v03_final evidence/v0.3/main
The acceptance tests and extrema run at 500 Hz; the committed trace is 10 Hz.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path


def export(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    report = json.loads((source/'summary.json').read_text())
    trace_fields = ['trial', 'time_s', 'pelvis_z', 'torso_tilt_rad', 'com_error_m',
                    'com_speed_m_s', 'left_normal_n', 'right_normal_n', 'left_contacts',
                    'right_contacts', 'nonfoot_floor_contacts', 'support_margin_m',
                    'min_joint_margin_rad', 'max_applied_torque_nm', 'stopped', 'reason']
    with (destination/'trace_10hz.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=trace_fields, lineterminator='\n'); writer.writeheader()
        for result in report['results']:
            path = Path(result['log'])
            result['raw_csv_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            result['log'] = path.name
            result['trace_id'] = path.stem
            last = None
            with path.open(newline='') as f:
                for k, row in enumerate(csv.DictReader(f)):
                    compact = {field: row.get(field, '') for field in trace_fields if field != 'trial'}
                    compact['trial'] = path.stem
                    if k % 50 == 0:
                        writer.writerow(compact)
                        last = None
                    else:
                        last = compact
                if last:
                    writer.writerow(last)
    report['trace_note'] = 'Trace downsampled to 10 Hz for review. Criteria, events and extrema evaluated at every 2 ms physics step. Raw CSVs regenerate via documented command; SHA256 retained.'
    (destination/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    export(*sys.argv[1:])
