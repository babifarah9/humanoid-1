"""Comparative standing experiment, designed to run with only numpy + mujoco."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .core import Humanoid,StandingPD
from .balance import BalanceController

FIELDS=['time_s','pelvis_z','root_x','root_y','pitch_rad','roll_rad','com_x','com_y','com_err_x','com_err_y','com_vx','com_vy','foot_contacts','normal_force_n','max_requested_torque_nm','n_saturated','stopped','reason']

def trial(seconds=30., mode='baseline', log=None, pitch_sign=1.,com_sign=1.,
          kp=85.,kd=14.,ankle_gain=90.,com_gain=110.,hip_gain=35.,derivative_gain=14., use_bias=True):
    robot=Humanoid()
    controller=BalanceController(robot,tilt_sign=pitch_sign,com_sign=com_sign,kp=kp,kd=kd,
                                 ankle_gain=ankle_gain,com_gain=com_gain,hip_gain=hip_gain,
                                 derivative_gain=derivative_gain,use_bias=use_bias)
    baseline=StandingPD(robot)
    path=Path(log) if log else None
    if path: path.parent.mkdir(parents=True,exist_ok=True)
    rows=[]; frame=int(round(.02/robot.model.opt.timestep));frame=max(frame,1)
    for _ in range(int(np.ceil(seconds/(frame*robot.model.opt.timestep)))):
        if mode=='baseline':
            tau=baseline.act(); di=controller.diagnostics()
            di['max_requested_torque_nm']=float(np.max(abs(tau)));di['n_saturated']=0
        else:
            tau=controller.act();di=controller.last.copy()
        _,info=robot.step(tau,frames=frame)
        di['stopped']=bool(robot.stopped);di['reason']=info.get('reason') or ''
        rows.append({k:di.get(k,'') for k in FIELDS})
        if robot.stopped:break
    if path:
        with path.open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=FIELDS);wr.writeheader();wr.writerows(rows)
    final=controller.diagnostics()
    return dict(mode=mode,elapsed_s=round(float(robot.data.time),3),seconds_requested=seconds,
                achieved_30s=bool(robot.data.time>=30. and not robot.stopped),
                stopped=bool(robot.stopped),reason=robot.safety_reason(),
                pelvis_z=round(float(robot.data.qpos[2]),4),root_x=round(float(robot.data.qpos[0]),4),
                max_abs_pitch_rad=round(max(abs(float(r['pitch_rad'])) for r in rows),4),
                max_abs_roll_rad=round(max(abs(float(r['roll_rad'])) for r in rows),4),
                foot_contacts_at_end=int(final['foot_contacts']),log=str(path) if path else '')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--mode',choices=['baseline','balance','sweep'],default='balance')
    p.add_argument('--log',default='results/v02_balance.csv')
    a=p.parse_args()
    if a.mode!='sweep':
        print(json.dumps(trial(seconds=a.seconds,mode=a.mode,log=a.log),indent=2));return
    # Compare signs and strengths without claiming their optimality.
    candidates=[('baseline',dict(mode='baseline'))]
    for bias in [True,False]:
        for sign in [-1.,1.]:
            for cs in [-1.,1.]:
                for amp in [0.5,1.0]:
                    candidates.append((f'bias{int(bias)}_tilt{sign:+g}_com{cs:+g}_gain{amp}',
                                       dict(mode='balance',use_bias=bias,pitch_sign=sign,com_sign=cs,
                                            ankle_gain=90*amp,com_gain=110*amp)))
    results=[]
    for name,kwargs in candidates:
        try:
            r=trial(seconds=a.seconds,**kwargs);r['candidate']=name;results.append(r)
            print(name,'duration=',r['elapsed_s'],'stop=',r['stopped'],flush=True)
        except Exception as e:
            results.append(dict(candidate=name,error=repr(e)));print(name,'ERROR',e,flush=True)
    path=Path('results/v02_sweep.json');path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(results,indent=2))
    valid=[r for r in results if 'elapsed_s' in r]
    if valid:
        best=max(valid,key=lambda r:(r['elapsed_s'],-r['max_abs_pitch_rad']))
        print('BEST EXPERIMENTAL CANDIDATE:',json.dumps(best,indent=2))
    print('Full sweep saved:',path)
if __name__=='__main__':main()
