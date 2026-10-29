import numpy as np
import pytest
pytest.importorskip('gymnasium')
from gymnasium.utils.env_checker import check_env
from humanoid1.weight_shift_env import WeightShiftEnv


def test_weight_shift_environment_contract_and_program_seeds():
    env=WeightShiftEnv(seconds=.2)
    check_env(env,skip_render_check=True)
    x,a=env.reset(seed=42);segments=env.task.segments
    y,b=env.reset(seed=42)
    assert a==b and segments==env.task.segments and np.array_equal(x,y)
    for _ in range(10):obs,_,terminated,truncated,_=env.step(np.zeros(16))
    assert truncated and not terminated and env.observation_space.contains(obs)
    assert not env.task.summary()['passed']
