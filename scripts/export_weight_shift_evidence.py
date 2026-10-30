"""Export complete 10 Hz review traces from finished 500 Hz experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('destination')
    a=p.parse_args();source=Path(a.source);dest=Path(a.destination);dest.mkdir(parents=True,exist_ok=True)
    report=json.loads((source/'summary.json').read_text())
    fields=['trial','time_s','command_y_m','com_y','tracking_xy_m','left_load_fraction','torso_tilt_rad',
        'support_margin_m','joint_margin_rad','mechanical_work_j','sole_drift_m','stopped','reason']
    expected=[]
    with (dest/'trace_10hz.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for result in report['results']:
            raw=Path(result['log'])
            with raw.open(newline='') as f:rows=list(csv.DictReader(f))
            if not rows or abs(float(rows[-1]['time_s'])-result['elapsed_s'])>1e-8:
                raise ValueError('raw trace incomplete: '+str(raw))
            selected=[row for i,row in enumerate(rows) if i%50==0 or i==len(rows)-1]
            for row in selected:
                writer.writerow({**{k:row.get(k,'') for k in fields if k!='trial'},'trial':raw.stem})
            result['raw_csv_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest();result['log']=raw.name
            expected.append((raw.stem,len(selected)))
    with (dest/'trace_10hz.csv').open(newline='') as f:
        published=list(csv.DictReader(f))
    for trial,count in expected:
        if sum(row['trial']==trial for row in published)!=count:
            raise ValueError('review export is incomplete: '+trial)
    report['trace_note']='Complete 10 Hz review traces; all criteria/extrema were checked at each 2 ms physics step. Full CSVs regenerate locally.'
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified exported trials:',expected)


if __name__=='__main__':main()
