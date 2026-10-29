from pathlib import Path
import numpy as np
import mujoco

MODEL = Path(__file__).resolve().parent.parent / 'models' / 'humanoid1.xml'

class Humanoid:
    """Experimental physics environment. Not a safety-certified robot controller."""
    def __init__(self, model_path=MODEL):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu
        self.joint_names = []
        self.qadr, self.vadr, self.limits = [], [], []
        for i in range(self.nu):
            jid = int(self.model.actuator_trnid[i, 0])
            self.joint_names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid))
            self.qadr.append(int(self.model.jnt_qposadr[jid]))
            self.vadr.append(int(self.model.jnt_dofadr[jid]))
            self.limits.append(self.model.jnt_range[jid].copy())
        self.qadr = np.array(self.qadr, dtype=int)
        self.vadr = np.array(self.vadr, dtype=int)
        self.limits = np.asarray(self.limits)
        self.ctrl_limits = self.model.actuator_ctrlrange.copy()
        self.reset()

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 1.0
        self.data.qpos[3] = 1.0  # root quaternion identity
        mujoco.mj_forward(self.model, self.data)
        self.stopped = False
        return self.observe()

    def observe(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()]).astype(np.float32)

    def safety_reason(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return 'non-finite state'
        if self.data.qpos[2] < 0.60:
            return 'pelvis below floor-clearance threshold'
        if np.linalg.norm(self.data.qvel) > 90:
            return 'excessive joint/root velocity'
        if np.any(self.data.qpos[self.qadr] < self.limits[:, 0] - .07) or np.any(self.data.qpos[self.qadr] > self.limits[:, 1] + .07):
            return 'joint position outside tolerance'
        return None

    def step(self, torque, frames=10):
        torque = np.asarray(torque, dtype=float)
        if torque.shape != (self.nu,) or not np.isfinite(torque).all():
            raise ValueError('torque must be a finite vector with one value per actuator')
        if self.stopped:
            return self.observe(), {'stopped': True, 'reason': 'latched stop'}
        for _ in range(frames):
            reason = self.safety_reason()
            if reason:
                self.stopped = True
                self.data.ctrl[:] = 0
                break
            self.data.ctrl[:] = np.clip(torque, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
            mujoco.mj_step(self.model, self.data)
        return self.observe(), {'stopped': self.stopped, 'reason': self.safety_reason(), 'time': float(self.data.time), 'pelvis_height': float(self.data.qpos[2])}

class StandingPD:
    def __init__(self, robot, kp=70., kd=10.):
        self.robot=robot
        self.kp=np.full(robot.nu, float(kp));self.kd=np.full(robot.nu,float(kd))
        # Lower gain for arms, increase leg resistance modestly
        for i,name in enumerate(robot.joint_names):
            if 'shoulder' in name or 'elbow' in name:
                self.kp[i]=25;self.kd[i]=5
        self.target=np.zeros(robot.nu)
        # Slightly bend knees to avoid hyperextension contact
        for i,name in enumerate(robot.joint_names):
            if name.endswith('_knee'): self.target[i] = .06
        self.target=np.clip(self.target,robot.limits[:,0]+.02,robot.limits[:,1]-.02)

    def act(self):
        r=self.robot
        q=r.data.qpos[r.qadr];v=r.data.qvel[r.vadr]
        tau=self.kp*(self.target-q)-self.kd*v
        # Simple sagittal root feedback; experimental, not a validated balancer.
        quat=r.data.qpos[3:7]
        rot=np.zeros(9)
        mujoco.mju_quat2Mat(rot,quat)
        pitch=np.arctan2(-rot[6], np.sqrt(rot[7]**2+rot[8]**2))
        for i,name in enumerate(r.joint_names):
            if name.endswith('_ankle_pitch'): tau[i] -= 25*pitch + 3*r.data.qvel[4]
        return np.clip(tau,r.ctrl_limits[:,0],r.ctrl_limits[:,1])
