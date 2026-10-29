"""CPU-friendly exploratory gain search, NOT reinforcement learning."""
import csv
from pathlib import Path
import numpy as np
from .cli import run

def main():
    rng=np.random.default_rng(42)
    p=Path('results/gain_search.csv');p.parent.mkdir(exist_ok=True,parents=True)
    rows=[]
    for trial in range(12):
        kp=float(rng.uniform(35,110));kd=float(rng.uniform(5,20))
        result=run(seconds=2.,log=f'results/trial_{trial:02}.csv',kp=kp,kd=kd)
        rows.append({'trial':trial,'kp':kp,'kd':kd,'duration_s':result['elapsed_s'],'stopped':result['safety_stop']})
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('gain search saved:',p)
if __name__=='__main__':main()
