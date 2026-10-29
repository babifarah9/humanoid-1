"""Optional Gymnasium residual-PPO interface over the 500 Hz standing controller.

Install requirements-rl.txt. The policy runs at 50 Hz and can request only a
bounded +/-2 Nm residual. Safety checks run at each 2 ms physics substep.
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from .core import Humanoid
from .whole_body import WholeBodyController
from .verification import CRITERIA, support_margin, standing_checks


class ResidualStandEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seconds=30., scenario='nominal'):
        super().__init__()
        if not np.isfinite(seconds) or seconds <= .1:
            raise ValueError('seconds must exceed the contact settling period')
        if scenario not in ('nominal', 'initial_velocity', 'push_x', 'push_y'):
            raise ValueError('unknown scenario')
        self.robot = Humanoid()
        self.controller = WholeBodyController(self.robot)
        self.seconds = float(seconds)
        self.scenario = scenario
        self.action_space = spaces.Box(-1., 1., (self.robot.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
            (self.robot.model.nq + self.robot.model.nv + 3 + 2 + 2 + self.robot.nu + 1,), dtype=np.float32)
        self.done = True
        self.previous_action = np.zeros(self.robot.nu)
        self.failed_criteria = set()
        self.last_metrics = {}

    def _observation(self):
        r, c = self.robot, self.controller
        normal, _, _ = c.contacts()
        obs = np.r_[r.observe(), r.data.subtree_com[c.pelvis] - c.target_com,
                    normal/(r.model.body_mass.sum()*9.81),
                    r.data.time-c.last_loaded, self.previous_action,
                    r.data.time/self.seconds].astype(np.float32)
        return np.nan_to_num(obs, nan=0., posinf=1e6, neginf=-1e6)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.reset()
        self.controller.reset()
        self.done = False
        self.previous_action[:] = 0.
        self.failed_criteria = set()
        self.last_metrics = {}
        if self.scenario == 'initial_velocity':
            self.robot.data.qvel[:2] = self.np_random.uniform(-.015, .015, 2)
            self.robot.data.qvel[3:5] = self.np_random.uniform(-.01, .01, 2)
        mujoco.mj_forward(self.robot.model, self.robot.data)
        return self._observation(), {'scenario': self.scenario}

    def _check_envelope(self, di):
        r = self.robot
        q = r.data.qpos[r.qadr]
        joint_margin = float(np.min(np.minimum(q-r.limits[:, 0], r.limits[:, 1]-q)))
        margin = support_margin(r, [di['com_x'], di['com_y']],
            [di['left_normal_n'] >= 1., di['right_normal_n'] >= 1.])
        checks = standing_checks(di, margin, joint_margin)
        if r.data.time >= CRITERIA['settle_s']:
            self.failed_criteria.update(k for k, passed in checks.items() if not passed)
        return margin, joint_margin

    def step(self, action):
        if self.done:
            raise RuntimeError('reset is required before stepping a finished episode')
        r = self.robot
        action = np.asarray(action, dtype=float)
        if action.shape != (r.nu,) or not np.isfinite(action).all():
            r.emergency_stop('invalid policy action')
            self.done = True
            return self._observation(), -10., True, False, {'reason': r.stop_reason, 'episode_stable': False}
        action = np.clip(action, -1., 1.)
        reward = 0.
        for _ in range(10):
            if r.data.time >= self.seconds - 1e-10 or r.stopped:
                break
            r.data.xfrc_applied[:] = 0.
            if self.scenario.startswith('push_') and 5. <= r.data.time < 5.10:
                axis = 0 if self.scenario == 'push_x' else 1
                r.data.xfrc_applied[self.controller.torso, axis] = 5.
            torque = self.controller.act(action)
            r.step(torque, frames=1)
            di = self.controller.diagnostics()
            margin, joint_margin = self._check_envelope(di)
            self.last_metrics = dict(di, support_margin_m=margin, joint_margin_rad=joint_margin)
            reward += r.model.opt.timestep * (1. - 10.*di['torso_tilt_rad']**2
                - 100.*di['com_error_m']**2 - di['com_speed_m_s']**2 - .001*np.mean(action**2))
            if self.failed_criteria:
                r.emergency_stop('standing envelope: '+', '.join(sorted(self.failed_criteria)))
        reward -= .0001*float(np.mean((action-self.previous_action)**2))
        self.previous_action = action.copy()
        terminated = bool(r.stopped)
        truncated = bool(not terminated and r.data.time >= self.seconds - 1e-10)
        self.done = terminated or truncated
        if terminated:
            reward -= 10.
        info = dict(time_s=float(r.data.time), reason=r.stop_reason,
                    episode_stable=bool(truncated and not self.failed_criteria and not r.limit_events),
                    failed_criteria=sorted(self.failed_criteria), metrics=self.last_metrics.copy(),
                    joint_limit_events=list(r.limit_events))
        return self._observation(), float(reward), terminated, truncated, info
