"""Local candidate training, experience checkpoints and explicit skill promotion.

PPO's on-policy rollout buffer is owned by SB3. Saved evaluation trajectories
support auditing and later imitation/analysis; they are NOT replayed into PPO.
No model is trained, promoted or executed merely by importing this module.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import time
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
    if candidate.get('schema') != 'humanoid1-standing-evaluation-v1':
        reasons.append('unsupported evaluation schema')
    if candidate.get('evaluation_split') != 'held_out' or baseline.get('evaluation_split') != 'held_out':
        reasons.append('promotion requires held-out evidence')
    for field in ('schema', 'physics_timestep_s', 'policy_period_s', 'criteria', 'balance_config', 'source_sha256'):
        if field not in candidate or candidate.get(field) != baseline.get(field):
            reasons.append(f'evaluation configuration mismatch: {field}')
    if candidate.get('model_sha256') != baseline.get('model_sha256') or not candidate.get('model_sha256'):
        reasons.append('model mismatch')
    if baseline.get('policy_sha256') is not None:
        reasons.append('reference must be the zero-residual baseline')
    if (set(ca) != required or set(ca) != set(ba)
            or len(candidate.get('episodes', [])) != len(ca)
            or len(baseline.get('episodes', [])) != len(ba)):
        reasons.append('missing or unmatched held-out scenarios')
    for key, result in ca.items():
        if (not result.get('stable') or not np.isfinite(result.get('elapsed_s', np.nan))
                or result.get('elapsed_s', 0.) < 30.-1e-8 or result.get('failed_criteria')
                or result.get('joint_limit_events') or result.get('terminated')):
            reasons.append(f'failed episode {key}')
        score = result.get('total_return', -np.inf)
        if not np.isfinite(score):
            reasons.append(f'invalid reward {key}')
        reference = ba.get(key, {})
        if (not reference.get('stable') or not np.isfinite(reference.get('total_return', np.nan))
                or not np.isfinite(reference.get('elapsed_s', np.nan))
                or reference.get('elapsed_s', 0.) < 30.-1e-8
                or reference.get('terminated') or reference.get('joint_limit_events') or reference.get('failed_criteria')):
            reasons.append(f'invalid baseline episode {key}')
        if key in ba and score + 1e-6 < ba[key].get('total_return', np.inf):
            reasons.append(f'reward regression {key}')
    return dict(eligible=not reasons, reasons=reasons)


def evaluate_policy(policy, output, seconds=30., validation=False):
    from .ppo_env import ResidualStandEnv
    from .core import MODEL
    from .verification import CRITERIA
    directory = Path(output); directory.mkdir(parents=True, exist_ok=True)
    episodes = []
    jobs = [('nominal', 101)] + [('initial_velocity', s) for s in (101, 102, 103)]
    jobs += [('push_x', 101), ('push_y', 101)]
    if validation:
        # Validation seeds are separate from the final promotion suite.
        jobs = [('nominal', 11), ('initial_velocity', 12), ('initial_velocity', 13),
                ('push_x', 14), ('push_y', 15)]
    for scenario, seed in jobs:
        env = ResidualStandEnv(seconds=seconds, scenario=scenario)
        obs, _ = env.reset(seed=seed)
        observations, actions, rewards = [obs.copy()], [], []
        max_metrics = {}; min_metrics = {}; frames = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = np.zeros(env.action_space.shape, dtype=np.float32) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs.copy()); actions.append(np.asarray(action).copy()); rewards.append(reward)
            metrics = info.get('episode_metrics', {})
            for key, value in metrics.get('maximum', {}).items():
                max_metrics[key] = max(max_metrics.get(key, -np.inf), value)
            for key, value in metrics.get('minimum', {}).items():
                min_metrics[key] = min(min_metrics.get(key, np.inf), value)
            frames.append(float(info.get('time_s', env.robot.data.time)))
        trace = f'{scenario}_{seed}.npz'
        np.savez_compressed(directory/trace, observations=np.asarray(observations),
                            actions=np.asarray(actions), rewards=np.asarray(rewards), time_s=np.asarray(frames))
        episodes.append(dict(scenario=scenario, seed=seed, elapsed_s=float(env.robot.data.time),
                             stable=info['episode_stable'], terminated=terminated,
                             total_return=float(sum(rewards)), joint_limit_events=env.robot.limit_events,
                             reason=info.get('reason'), failed_criteria=info.get('failed_criteria', []),
                             episode_metrics=dict(maximum=max_metrics, minimum=min_metrics),
                             max_abs_residual_action=float(np.max(np.abs(actions))),
                             final_metrics=info.get('metrics', {}), trace=trace,
                             trace_sha256=hashlib.sha256((directory/trace).read_bytes()).hexdigest()))
        env.close()
        print(scenario, seed, 'stable=', episodes[-1]['stable'], flush=True)
    sources = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
               for name in ('core.py', 'whole_body.py', 'ppo_env.py', 'verification.py')}
    report = dict(schema='humanoid1-standing-evaluation-v1',
                  model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                  physics_timestep_s=float(env.robot.model.opt.timestep), policy_period_s=.02,
                  criteria=CRITERIA, balance_config=asdict(env.controller.config),
                  evaluation_split='validation' if validation else 'held_out',
                  requested_seconds=seconds, source_sha256=sources, episodes=episodes)
    (directory/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--timesteps', type=int, default=10000)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--log-std-init', type=float, default=-3.)
    train.add_argument('--output', default='results/learning/candidate')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--policy', help='SB3 PPO .zip; omitted means zero residual baseline')
    evaluate.add_argument('--output', default='results/learning/evaluation')
    evaluate.add_argument('--validation', action='store_true', help='separate seeds for candidate selection, never promotion')
    evaluate.add_argument('--seconds', type=float, default=30.)
    promote = sub.add_parser('promote')
    promote.add_argument('--candidate-report', required=True)
    promote.add_argument('--baseline-report', required=True)
    promote.add_argument('--policy', required=True)
    promote.add_argument('--registry', default='results/learning/skills.json')
    a = p.parse_args()
    if a.command == 'train':
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.callbacks import BaseCallback
        import torch
        from .ppo_env import ResidualStandEnv
        if a.timesteps < 1 or not np.isfinite(a.log_std_init) or not -6. <= a.log_std_init <= 0.:
            p.error('timesteps must be positive and log-std-init must be in [-6, 0]')
        torch.set_num_threads(1)
        env = ResidualStandEnv(scenario='initial_velocity')
        check_env(env)
        model = PPO('MlpPolicy', env, device='cpu', seed=a.seed, n_steps=256,
                    batch_size=64, n_epochs=5, policy_kwargs=dict(net_arch=[64, 64], log_std_init=a.log_std_init), verbose=1)
        initial = {name: param.detach().cpu().clone() for name, param in model.policy.named_parameters()}
        out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(out.parent/(out.name+'_logs')), ['stdout', 'csv']))
        class ProgressArchive(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episodes = []

            def _on_step(self):
                for done, info in zip(self.locals['dones'], self.locals['infos']):
                    if done:
                        self.episodes.append(dict(timesteps=self.num_timesteps,
                            elapsed_s=info.get('time_s'), stable=info.get('episode_stable', False),
                            reason=info.get('reason'), failed_criteria=info.get('failed_criteria', [])))
                if self.num_timesteps % 4096 == 0:
                    self.model.save(str(out)+f'_step{self.num_timesteps}')
                    Path(str(out)+'_episodes.json').write_text(json.dumps(self.episodes, indent=2)+'\n')
                return True
        archive = ProgressArchive()
        start = time.perf_counter()
        try:
            model.learn(total_timesteps=a.timesteps, callback=archive)
            model.save(str(out))
            Path(str(out)+'_episodes.json').write_text(json.dumps(archive.episodes, indent=2)+'\n')
            import mujoco, gymnasium, stable_baselines3
            from .core import MODEL
            policy_file = Path(str(out)+'.zip')
            changes = {name: float((param.detach().cpu()-initial[name]).abs().max())
                       for name, param in model.policy.named_parameters()}
            Path(str(out)+'.json').write_text(json.dumps(dict(seed=a.seed,
                requested_timesteps=a.timesteps, actual_timesteps=model.num_timesteps,
                wall_s=time.perf_counter()-start, log_std_init=a.log_std_init,
                policy_sha256=hashlib.sha256(policy_file.read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob('*.py')},
                versions=dict(python=platform.python_version(), torch=torch.__version__,
                              mujoco=mujoco.__version__, numpy=np.__version__, gymnasium=gymnasium.__version__,
                              stable_baselines3=stable_baselines3.__version__),
                parameter_max_abs_changes=changes, training_completed=True,
                device='cpu', policy='bounded residual', promoted=False), indent=2)+'\n')
        finally:
            env.close()
    elif a.command == 'evaluate':
        policy = None
        if a.policy:
            from stable_baselines3 import PPO
            policy = PPO.load(a.policy, device='cpu')
        report = evaluate_policy(policy, a.output, a.seconds, a.validation)
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
