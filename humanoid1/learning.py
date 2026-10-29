"""Local candidate training, experience checkpoints and explicit skill promotion.

PPO's on-policy rollout buffer is owned by SB3. Saved evaluation trajectories
support auditing and later imitation/analysis; they are NOT replayed into PPO.
No model is trained, promoted or executed merely by importing this module.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np


def promotion_decision(candidate, baseline):
    """Require matched >=30 s evaluations, all envelope passes, no regression.

    A gate for the standing skill only; it is not evidence of walking or general
    autonomous learning. Reports must be produced by evaluate_policy below.
    """
    required = {('nominal', 101), ('initial_velocity', 101),
                ('initial_velocity', 102), ('initial_velocity', 103),
                ('push_x', 101), ('push_y', 101)}
    def index(report):
        return {(r['scenario'], r['seed']): r for r in report.get('episodes', [])}
    ca, ba = index(candidate), index(baseline)
    reasons = []
    for field in ('schema', 'physics_timestep_s', 'policy_period_s', 'criteria', 'balance_config', 'source_sha256'):
        if field not in candidate or candidate.get(field) != baseline.get(field):
            reasons.append(f'evaluation configuration mismatch: {field}')
    if candidate.get('model_sha256') != baseline.get('model_sha256') or not candidate.get('model_sha256'):
        reasons.append('model mismatch')
    if baseline.get('policy_sha256') is not None:
        reasons.append('reference must be the zero-residual baseline')
    if set(ca) != required or set(ca) != set(ba) or len(candidate.get('episodes', [])) != len(ca):
        reasons.append('missing or unmatched held-out scenarios')
    for key, result in ca.items():
        if (not result.get('stable') or result.get('elapsed_s', 0.) < 30.-1e-8
                or result.get('joint_limit_events') or result.get('terminated')):
            reasons.append(f'failed episode {key}')
        score = result.get('total_return', -np.inf)
        if not np.isfinite(score):
            reasons.append(f'invalid reward {key}')
        reference = ba.get(key, {})
        if (not reference.get('stable') or not np.isfinite(reference.get('total_return', np.nan))
                or reference.get('elapsed_s', 0.) < 30.-1e-8):
            reasons.append(f'invalid baseline episode {key}')
        if key in ba and score + 1e-6 < ba[key].get('total_return', np.inf):
            reasons.append(f'reward regression {key}')
    return dict(eligible=not reasons, reasons=reasons)


def evaluate_policy(policy, output, seconds=30.):
    from .ppo_env import ResidualStandEnv
    from .core import MODEL
    from .verification import CRITERIA
    directory = Path(output); directory.mkdir(parents=True, exist_ok=True)
    episodes = []
    jobs = [('nominal', 101)] + [('initial_velocity', s) for s in (101, 102, 103)]
    jobs += [('push_x', 101), ('push_y', 101)]
    for scenario, seed in jobs:
        env = ResidualStandEnv(seconds=seconds, scenario=scenario)
        obs, _ = env.reset(seed=seed)
        observations, actions, rewards = [obs.copy()], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            action = np.zeros(env.action_space.shape, dtype=np.float32) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs.copy()); actions.append(np.asarray(action).copy()); rewards.append(reward)
        trace = f'{scenario}_{seed}.npz'
        np.savez_compressed(directory/trace, observations=np.asarray(observations),
                            actions=np.asarray(actions), rewards=np.asarray(rewards))
        episodes.append(dict(scenario=scenario, seed=seed, elapsed_s=float(env.robot.data.time),
                             stable=info['episode_stable'], terminated=terminated,
                             total_return=float(sum(rewards)), joint_limit_events=env.robot.limit_events,
                             final_metrics=info.get('metrics', {}), trace=trace))
        env.close()
        print(scenario, seed, 'stable=', episodes[-1]['stable'], flush=True)
    sources = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
               for name in ('core.py', 'whole_body.py', 'ppo_env.py', 'verification.py')}
    report = dict(schema='humanoid1-standing-evaluation-v1',
                  model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                  physics_timestep_s=float(env.robot.model.opt.timestep), policy_period_s=.02,
                  criteria=CRITERIA, balance_config=asdict(env.controller.config),
                  source_sha256=sources, episodes=episodes)
    (directory/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--timesteps', type=int, default=10000)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--output', default='results/learning/candidate')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--policy', help='SB3 PPO .zip; omitted means zero residual baseline')
    evaluate.add_argument('--output', default='results/learning/evaluation')
    promote = sub.add_parser('promote')
    promote.add_argument('--candidate-report', required=True)
    promote.add_argument('--baseline-report', required=True)
    promote.add_argument('--policy', required=True)
    promote.add_argument('--registry', default='results/learning/skills.json')
    a = p.parse_args()
    if a.command == 'train':
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        import torch
        from .ppo_env import ResidualStandEnv
        if a.timesteps < 1:
            p.error('timesteps must be positive')
        torch.set_num_threads(1)
        env = ResidualStandEnv(scenario='initial_velocity')
        check_env(env)
        model = PPO('MlpPolicy', env, device='cpu', seed=a.seed, n_steps=256,
                    batch_size=64, n_epochs=5, policy_kwargs=dict(net_arch=[64, 64]), verbose=1)
        try:
            model.learn(total_timesteps=a.timesteps)
            out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(out))
            Path(str(out)+'.json').write_text(json.dumps(dict(seed=a.seed,
                requested_timesteps=a.timesteps, actual_timesteps=model.num_timesteps,
                device='cpu', policy='bounded residual', promoted=False), indent=2)+'\n')
        finally:
            env.close()
    elif a.command == 'evaluate':
        policy = None
        if a.policy:
            from stable_baselines3 import PPO
            policy = PPO.load(a.policy, device='cpu')
        report = evaluate_policy(policy, a.output)
        report['policy_sha256'] = hashlib.sha256(Path(a.policy).read_bytes()).hexdigest() if a.policy else None
        (Path(a.output)/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    else:
        ca = json.loads(Path(a.candidate_report).read_text())
        ba = json.loads(Path(a.baseline_report).read_text())
        decision = promotion_decision(ca, ba)
        policy = Path(a.policy)
        digest = hashlib.sha256(policy.read_bytes()).hexdigest()
        if digest != ca.get('policy_sha256'):
            decision = dict(eligible=False, reasons=decision['reasons']+['policy hash mismatch'])
        if not decision['eligible']:
            print(json.dumps(decision, indent=2))
            raise SystemExit(1)
        registry = Path(a.registry); registry.parent.mkdir(parents=True, exist_ok=True)
        records = json.loads(registry.read_text()) if registry.exists() else []
        records.append(dict(skill='standing', policy=str(policy.resolve()), sha256=digest,
                            evaluation=str(Path(a.candidate_report).resolve()), decision=decision))
        registry.write_text(json.dumps(records, indent=2)+'\n')
        print('Standing skill recorded. Runtime activation remains explicit.')


if __name__ == '__main__':
    main()
