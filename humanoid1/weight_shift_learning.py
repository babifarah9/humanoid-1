"""Optional CPU PPO experiment for the same bounded weight-shifting task.

This module never changes robot limits, standing criteria or task commands to
favor a learned candidate. Promotion requires >=5% mean RMS tracking improvement
without substantial per-case, work or slip regressions.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np


def shift_promotion(candidate, baseline):
    reasons=[]
    for field in ('schema','model_sha256','standing_criteria','task_criteria','source_sha256'):
        if not candidate.get(field) or candidate.get(field)!=baseline.get(field):
            reasons.append('configuration mismatch: '+field)
    if candidate.get('schema')!='humanoid1-weight-shift-v1' or not candidate.get('policy_sha256') or baseline.get('policy_sha256') is not None:
        reasons.append('invalid candidate/reference identity')
    def index(report):
        return {(r['program']['seed'],r['program']['random_commands'],r['program']['disturbed']):r for r in report.get('results',[])}
    ca,ba=index(candidate),index(baseline)
    required={(0,False,False),(101,True,False),(102,True,False),(103,True,False),(201,True,True),(202,True,True)}
    if set(ca)!=required or set(ba)!=required or len(ca)!=len(candidate.get('results',[])) or len(ba)!=len(baseline.get('results',[])):
        reasons.append('missing/duplicate matched final scenarios')
    for key,result in ca.items():
        ref=ba.get(key,{})
        for value in (result,ref):
            if (not value.get('passed') or value.get('stopped') or value.get('joint_limit_events')
                    or not np.isfinite(value.get('elapsed_s',np.nan)) or value.get('elapsed_s',0)<30.-1e-8):
                reasons.append('failed final scenario: '+str(key))
        if result.get('segments')!=ref.get('segments') or result.get('disturbances')!=ref.get('disturbances'):
            reasons.append('command/disturbance mismatch: '+str(key))
        for name in ('rms_tracking_xy_m','mechanical_absolute_work_j'):
            if not np.isfinite(result.get(name,np.nan)) or not np.isfinite(ref.get(name,np.nan)):
                reasons.append('non-finite metric: '+str(key))
        if result.get('rms_tracking_xy_m',np.inf)>1.02*ref.get('rms_tracking_xy_m',0):
            reasons.append('per-case tracking regression: '+str(key))
        if result.get('mechanical_absolute_work_j',np.inf)>1.10*ref.get('mechanical_absolute_work_j',0):
            reasons.append('mechanical work regression: '+str(key))
        slip=result.get('cumulative_contact_slip_m',[]);ref_slip=ref.get('cumulative_contact_slip_m',[])
        if (len(slip)!=2 or len(ref_slip)!=2 or not np.isfinite(slip).all() or not np.isfinite(ref_slip).all()
                or any(a>max(.001,1.10*b) for a,b in zip(slip,ref_slip))):
            reasons.append('slip evidence/regression: '+str(key))
    mean_candidate=np.mean([r.get('rms_tracking_xy_m',np.inf) for r in ca.values()]) if ca else np.inf
    mean_baseline=np.mean([r.get('rms_tracking_xy_m',np.nan) for r in ba.values()]) if ba else np.nan
    if not np.isfinite(mean_candidate) or not np.isfinite(mean_baseline) or mean_candidate>.95*mean_baseline:
        reasons.append('no >=5% mean RMS tracking improvement')
    return dict(eligible=not reasons,reasons=reasons,mean_candidate_rms_m=float(mean_candidate) if np.isfinite(mean_candidate) else None,
                mean_baseline_rms_m=float(mean_baseline) if np.isfinite(mean_baseline) else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train');train.add_argument('--steps',type=int,default=16384)
    train.add_argument('--seed',type=int,default=7);train.add_argument('--output',default='results/v05/ppo_candidate')
    gate=sub.add_parser('compare');gate.add_argument('--candidate',required=True);gate.add_argument('--baseline',required=True)
    gate.add_argument('--output',default='results/v05/promotion.json')
    a=p.parse_args()
    if a.command=='compare':
        result=shift_promotion(json.loads(Path(a.candidate).read_text()),json.loads(Path(a.baseline).read_text()))
        dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return
    if a.steps<256:
        p.error('steps must be at least 256')
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.logger import configure
    from .weight_shift_env import WeightShiftEnv
    torch.set_num_threads(1)
    env=WeightShiftEnv();check_env(env)
    model=PPO('MlpPolicy',env,device='cpu',seed=a.seed,n_steps=256,batch_size=64,n_epochs=5,
              target_kl=.02,policy_kwargs=dict(net_arch=[64,64],log_std_init=-3.),verbose=1)
    initial={n:v.detach().cpu().clone() for n,v in model.policy.named_parameters()}
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    model.set_logger(configure(str(dest)+'_logs',['stdout','csv']))
    sources={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
             for n in ('core.py','whole_body.py','verification.py','weight_shift.py','weight_shift_env.py','weight_shift_learning.py')}
    try:
        model.learn(total_timesteps=a.steps);model.save(str(dest))
        info=dict(actual_steps=model.num_timesteps,requested_steps=a.steps,seed=a.seed,device='cpu',
            source_sha256=sources,policy_sha256=hashlib.sha256(Path(str(dest)+'.zip').read_bytes()).hexdigest(),
            actor_max_abs_parameter_change=float((model.policy.action_net.weight.detach().cpu()-initial['action_net.weight']).abs().max()),
            normalized_residual_bound=1.,residual_torque_bound_nm=2.,promoted=False)
        Path(str(dest)+'.json').write_text(json.dumps(info,indent=2)+'\n')
    finally:
        env.close()


if __name__=='__main__':main()
