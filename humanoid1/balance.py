"""Experimental v0.2 feedback controller. NOT validated for physical hardware."""
import numpy as np
import mujoco
from .core import StandingPD

class BalanceController:
    """Gravity compensated PD + contact-aware COM/tilt compensation.

    No claim of formal stability or guaranteed fall prevention. All gains are
    hypotheses; benchmark.py evaluates them against the existing baseline.
    """
    def __init__(self, robot, tilt_sign=1., com_sign=1., kp=85., kd=14.,
                 ankle_gain=90., com_gain=110., hip_gain=35., derivative_gain=14.,
                 use_bias=True):
        self.r=robot
        self.baseline=StandingPD(robot,kp=kp,kd=kd)
        self.tilt_sign=float(tilt_sign)
        self.com_sign=float(com_sign)
        self.ankle_gain=float(ankle_gain)
        self.com_gain=float(com_gain)
        self.hip_gain=float(hip_gain)
        self.derivative_gain=float(derivative_gain)
        self.use_bias=bool(use_bias)
        self.last={}
    def diagnostics(self):
        r=self.r; m=r.model; d=r.data
        mujoco.mj_forward(m,d)
        mat=d.xmat[1].reshape(3,3)
        # Tilt of pelvis body z-axis in the world XY directions, robust near upright.
        pitch=float(np.arctan2(mat[0,2],mat[2,2]))
        roll=float(np.arctan2(mat[1,2],mat[2,2]))
        # World angular velocity of free root, index 3:6 of qvel.
        pitch_rate=float(d.qvel[4]); roll_rate=float(d.qvel[3])
        com=d.subtree_com[1].copy()
        feet=[]; contacts=0; total_normal=0.
        for bid,name in enumerate(['left_foot','right_foot']):
            body_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
            feet.append(d.xpos[body_id].copy())
        feet=np.array(feet)
        support_center=feet[:,:2].mean(axis=0)
        com_error=np.asarray(com[:2]-support_center)
        # The explicit foot COM velocity estimate is finite-difference-free and
        # uses the time derivative of overall COM for damping.
        J=np.zeros((3,m.nv)); mujoco.mj_jacSubtreeCom(m,d,J,1)
        com_vel=J @ d.qvel
        for ci in range(d.ncon):
            c=d.contact[ci]
            n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom1)) or ''
            body2=int(m.geom_bodyid[c.geom2]); body1=int(m.geom_bodyid[c.geom1])
            b2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body2) or ''
            b1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body1) or ''
            if ('foot' in b1 or 'foot' in b2) and (n1=='floor' or m.geom_bodyid[c.geom1]==0 or m.geom_bodyid[c.geom2]==0):
                contacts+=1
                f=np.zeros(6); mujoco.mj_contactForce(m,d,ci,f);total_normal+=max(float(f[0]),0.)
        return dict(time_s=float(d.time), pelvis_z=float(d.qpos[2]), root_x=float(d.qpos[0]),root_y=float(d.qpos[1]),
                    pitch_rad=pitch,roll_rad=roll,pitch_rate=pitch_rate,roll_rate=roll_rate,
                    com_x=float(com[0]),com_y=float(com[1]),com_err_x=float(com_error[0]),com_err_y=float(com_error[1]),
                    com_vx=float(com_vel[0]),com_vy=float(com_vel[1]),foot_contacts=contacts,
                    normal_force_n=total_normal)
    def act(self):
        r=self.r;d=r.data
        di=self.diagnostics()
        q=d.qpos[r.qadr];v=d.qvel[r.vadr]
        tau=self.baseline.kp*(self.baseline.target-q)-self.baseline.kd*v
        # MuJoCo bias includes gravity and Coriolis terms. Only actuated DOFs
        # are compensated; the floating root is not treated as an actuator.
        if self.use_bias: tau+=d.qfrc_bias[r.vadr]
        # Sign conventions are tested explicitly in benchmark candidates because
        # the effect of equal ankle torque depends on the complete contact model.
        pitch_error=(di['pitch_rad'] + .4*di['pitch_rate'])
        roll_error=(di['roll_rad'] + .4*di['roll_rate'])
        sagittal=(self.ankle_gain*self.tilt_sign*pitch_error + self.com_gain*self.com_sign*di['com_err_x'] + self.derivative_gain*di['com_vx'])
        lateral=(self.ankle_gain*self.tilt_sign*roll_error + self.com_gain*self.com_sign*di['com_err_y'] + self.derivative_gain*di['com_vy'])
        for i,name in enumerate(r.joint_names):
            if name.endswith('ankle_pitch'): tau[i]+=sagittal
            if name.endswith('hip_pitch'): tau[i]+=self.hip_gain*self.tilt_sign*pitch_error
            if name.endswith('ankle_roll'): tau[i]-=lateral
            if name.endswith('hip_roll'): tau[i]-=.35*self.hip_gain*self.tilt_sign*roll_error
        self.last=dict(di)
        self.last['max_requested_torque_nm']=float(np.max(np.abs(tau)))
        self.last['n_saturated']=int(np.sum((tau<r.ctrl_limits[:,0])|(tau>r.ctrl_limits[:,1])))
        return np.clip(tau,r.ctrl_limits[:,0],r.ctrl_limits[:,1])
