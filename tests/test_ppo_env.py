import numpy as np
import pytest
pytest.importorskip('gymnasium')
from gymnasium.utils.env_checker import check_env
from humanoid1.ppo_env import ResidualStandEnv


def test_gym_contract_and_seeded_reset():
    env = ResidualStandEnv(seconds=.2, scenario='initial_velocity')
    check_env(env, skip_render_check=True)
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    assert np.array_equal(a, b)
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(np.zeros(16))
    assert truncated and not terminated and info['episode_stable']
    assert env.observation_space.contains(obs)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(16))


def test_invalid_action_and_safety_during_policy_substeps():
    env = ResidualStandEnv(seconds=.2)
    env.reset(seed=0)
    _, _, terminated, _, _ = env.step(np.full(16, np.nan))
    assert terminated and env.robot.data.time == 0.
    env.reset(seed=0)
    env.robot.data.qpos[2] = .59
    _, _, terminated, _, info = env.step(np.zeros(16))
    assert terminated and env.robot.data.time == 0.
    assert not info['episode_stable']


def test_episode_extrema_include_substeps_and_reset():
    env = ResidualStandEnv(seconds=.2)
    env.reset(seed=0)
    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros(16))
    metrics = info['episode_metrics']
    assert metrics['minimum']['support_margin_m'] >= .02
    assert metrics['minimum']['foot_normal_n'] >= 1.
    assert metrics['maximum']['com_speed_m_s'] >= info['metrics']['com_speed_m_s']
    env.reset(seed=0)
    assert not env.episode_maximum and not env.episode_minimum
