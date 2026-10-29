"""Dependency-light Gym-style environment for future RL integration."""
import numpy as np
from .core import Humanoid

class StandEnv:
    def __init__(self, max_steps=500):
        self.robot=Humanoid();self.max_steps=max_steps;self.steps=0
        self.observation_size=self.robot.model.nq+self.robot.model.nv
        self.action_size=self.robot.nu
    def reset(self, seed=None):
        self.steps=0;return self.robot.reset(seed), {}
    def step(self, action):
        self.steps+=1
        x,info=self.robot.step(action,frames=10)
        upright=np.clip(self.robot.data.qpos[2],0,1.0)
        effort=float(np.sum(np.square(np.asarray(action)/np.maximum(1,self.robot.ctrl_limits[:,1]))))
        reward=float(1.0*upright - .001*effort)
        terminated=bool(info['stopped']);truncated=self.steps>=self.max_steps
        return x,reward,terminated,truncated,info
