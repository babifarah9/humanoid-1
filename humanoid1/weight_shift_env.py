"""Optional Gymnasium interface; 50 Hz policy, 500 Hz safety/whole-body control."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .weight_shift import ShiftTask, ShiftProgram


class WeightShiftEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self, seconds=30.):
        self.seconds=seconds;self.task=None
        self.action_space=spaces.Box(-1.,1.,(16,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(75,),dtype=np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        program_seed=int(self.np_random.integers(0,2**31-1))
        self.task=ShiftTask(ShiftProgram(program_seed,True,True),self.seconds)
        return self.task.observation(),dict(program_seed=program_seed)

    def step(self,action):
        if self.task is None:
            raise RuntimeError('reset is required')
        return self.task.step(action)
