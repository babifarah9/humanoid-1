"""v0.3 double-support inverse-dynamics standing (NumPy + MuJoCo, CPU only).

Contact wrenches are internal predictions used to compute joint torques. They
are NEVER applied to the floating root or the world. This is a local controller,
not a walking policy or a constrained-QP/hardware safety guarantee.
"""
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass(frozen=True)
class BalanceConfig:
    posture_kp: float = 40.
    posture_kd: float = 12.
    com_kp: float = 60.
    com_kd: float = 16.
    torso_kp: float = 80.
    torso_kd: float = 18.
    foot_kp: float = 100.
    foot_kd: float = 20.
    startup_s: float = .10
    contact_timeout_s: float = .10
    min_normal_n: float = 1.
    max_tilt_rad: float = .50
    residual_scale_nm: float = 2.


def rotation_error(current, target):
    """World-frame small-angle SO(3) feedback; supervisor bounds its domain."""
    return .5 * sum((np.cross(current[:, i], target[:, i]) for i in range(3)))


def full_mass(model, data):
    result = np.empty((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, result)  # MuJoCo >= 3.4
    except TypeError:
        mujoco.mj_fullM(model, result, data.qM)  # MuJoCo 3.2 / 3.3
    return result


class WholeBodyController:
    def __init__(self, robot, config=None):
        self.r = robot
        self.config = config or BalanceConfig()
        m, d = robot.model, robot.data
        # This controller's controls are Nm only for direct unit-gear motors.
        if not np.allclose(m.actuator_gear[:, 0], 1.) or not np.allclose(m.actuator_gear[:, 1:], 0.):
            raise ValueError('whole-body controller requires unit-gear joint motors')
        self.feet = [m.body(name).id for name in ('left_foot', 'right_foot')]
        self.foot_geoms = [int(np.flatnonzero(m.geom_bodyid == b)[0]) for b in self.feet]
        self.torso = m.body('torso').id
        self.pelvis = m.body('pelvis').id
        self.floor = m.geom('floor').id
        self.selection = np.eye(m.nv)[robot.vadr]
        self.reset()

    def reset(self):
        m, d = self.r.model, self.r.data
        mujoco.mj_forward(m, d)
        self.target_q = d.qpos[self.r.qadr].copy()
        self.target_com = d.subtree_com[self.pelvis].copy()
        self.target_torso = d.xmat[self.torso].reshape(3, 3).copy()
        self.foot_positions = [d.xpos[b].copy() for b in self.feet]
        self.foot_rotations = [d.xmat[b].reshape(3, 3).copy() for b in self.feet]
        self.start_time = float(d.time)
        self.last_loaded = np.full(2, d.time)
        self.last = {}

    def contacts(self):
        """Only floor/foot contacts count; self-collisions cannot fake support."""
        m, d = self.r.model, self.r.data
        normal = np.zeros(2)
        counts = np.zeros(2, dtype=int)
        nonfoot = 0
        for ci in range(d.ncon):
            c = d.contact[ci]
            pair = (int(c.geom1), int(c.geom2))
            if self.floor not in pair:
                continue
            other = pair[1] if pair[0] == self.floor else pair[0]
            if other in self.foot_geoms:
                i = self.foot_geoms.index(other)
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                normal[i] += max(0., float(f[0]))
                counts[i] += 1
            else:
                nonfoot += 1
        return normal, counts, nonfoot

    def diagnostics(self):
        m, d = self.r.model, self.r.data
        mujoco.mj_forward(m, d)
        normal, counts, nonfoot = self.contacts()
        jc = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jc, self.pelvis)
        mat = d.xmat[self.torso].reshape(3, 3)
        tilt = float(np.arccos(np.clip(mat[2, 2], -1., 1.)))
        com = d.subtree_com[self.pelvis]
        return dict(time_s=float(d.time), pelvis_z=float(d.qpos[2]),
                    torso_tilt_rad=tilt, com_x=float(com[0]), com_y=float(com[1]), com_z=float(com[2]),
                    com_error_m=float(np.linalg.norm(com - self.target_com)),
                    com_speed_m_s=float(np.linalg.norm(jc @ d.qvel)),
                    left_normal_n=float(normal[0]), right_normal_n=float(normal[1]),
                    left_contacts=int(counts[0]), right_contacts=int(counts[1]),
                    nonfoot_floor_contacts=nonfoot)

    def act(self, residual=None):
        r, c = self.r, self.config
        m, d = r.model, r.data
        if r.stopped:
            return np.zeros(r.nu)
        reason = r.safety_reason()
        if reason:
            r.emergency_stop(reason)
            return np.zeros(r.nu)
        self.last = self.diagnostics()
        normal, _, nonfoot = self.contacts()
        self.last_loaded[normal >= c.min_normal_n] = d.time
        if nonfoot:
            r.emergency_stop('non-foot ground contact')
        elif self.last['torso_tilt_rad'] > c.max_tilt_rad:
            r.emergency_stop('whole-body tilt envelope exceeded')
        elif d.time - self.start_time > c.startup_s and np.any(d.time - self.last_loaded > c.contact_timeout_s):
            r.emergency_stop('bilateral ground support lost')
        if r.stopped:
            return np.zeros(r.nu)
        if residual is None:
            residual = np.zeros(r.nu)
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (r.nu,) or not np.isfinite(residual).all():
            r.emergency_stop('invalid policy residual')
            return np.zeros(r.nu)
        # During initial contact settling only, a sole within 1 cm of the plane
        # may be predicted as support before the soft contact develops load.
        active = []
        for i, g in enumerate(self.foot_geoms):
            sole_z = d.geom_xpos[g, 2] - m.geom_size[g, 2]
            if normal[i] >= c.min_normal_n or (d.time - self.start_time <= c.startup_s and abs(sole_z) <= .01):
                active.append(i)
        if not active:
            # No fictitious airborne contact wrench: damp joints while the
            # contact timeout and existing fall/velocity checks remain active.
            return np.clip(-c.posture_kd * d.qvel[r.vadr], r.ctrl_limits[:, 0], r.ctrl_limits[:, 1])
        try:
            tau = self._inverse_dynamics(active)
        except (np.linalg.LinAlgError, FloatingPointError):
            r.emergency_stop('whole-body numerical solve failed')
            return np.zeros(r.nu)
        tau += c.residual_scale_nm * np.clip(residual, -1., 1.)
        if not np.isfinite(tau).all():
            r.emergency_stop('non-finite whole-body command')
            return np.zeros(r.nu)
        self.last['max_requested_torque_nm'] = float(np.max(np.abs(tau)))
        self.last['n_saturated'] = int(np.sum((tau < r.ctrl_limits[:, 0]) | (tau > r.ctrl_limits[:, 1])))
        return np.clip(tau, r.ctrl_limits[:, 0], r.ctrl_limits[:, 1])

    def _inverse_dynamics(self, active):
        r, c = self.r, self.config
        m, d = r.model, r.data
        rows, targets = [], []
        for i in active:
            body = self.feet[i]
            jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
            jdp = np.zeros_like(jp); jdr = np.zeros_like(jp)
            mujoco.mj_jacBody(m, d, jp, jr, body)
            mujoco.mj_jacDot(m, d, jdp, jdr, d.xpos[body], body)
            j = np.vstack((jp, jr))
            err = np.r_[self.foot_positions[i] - d.xpos[body],
                        rotation_error(d.xmat[body].reshape(3, 3), self.foot_rotations[i])]
            rows.append(j)
            targets.append(c.foot_kp * err - c.foot_kd * j @ d.qvel - np.vstack((jdp, jdr)) @ d.qvel)
        contact_j = np.vstack(rows)
        jc = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jc, self.pelvis)
        jp = np.zeros_like(jc); jt = np.zeros_like(jc)
        mujoco.mj_jacBody(m, d, jp, jt, self.torso)
        q = d.qpos[r.qadr]
        posture = c.posture_kp * (self.target_q - q) - c.posture_kd * d.qvel[r.vadr]
        # Repulsion inside the original joint ranges; no widening of limits.
        posture += 100. * (np.maximum(r.limits[:, 0] + .04 - q, 0.) - np.maximum(q - r.limits[:, 1] + .04, 0.))
        com_acc = self.com_task_acceleration(jc)
        torso_acc = c.torso_kp * rotation_error(d.xmat[self.torso].reshape(3, 3), self.target_torso) - c.torso_kd * jt @ d.qvel
        # Foot Jdot*v is explicit. COM/torso Jdot*v is neglected locally near
        # rest; this approximation limits the controller to slow standing.
        A = np.vstack((self.selection, 30. * jc, 10. * jt))
        b = np.r_[posture, 30. * com_acc, 10. * torso_acc]
        nc = contact_j.shape[0]
        K = np.block([[A.T @ A + 1e-6 * np.eye(m.nv), contact_j.T],
                      [contact_j, np.zeros((nc, nc))]])
        # Two flat feet have redundant yaw constraints: use rank-revealing SVD,
        # not an inverse of the singular KKT system.
        acc = np.linalg.lstsq(K, np.r_[A.T @ b, np.concatenate(targets)], rcond=1e-10)[0][:m.nv]
        required = full_mass(m, d) @ acc + d.qfrc_bias - d.qfrc_passive
        root_j = contact_j.T[:6]
        weights_inv = np.diag(np.tile([1., 1., 10., .1, .1, .1], len(active)))
        wrench = weights_inv @ root_j.T @ np.linalg.solve(root_j @ weights_inv @ root_j.T, required[:6])
        # Free-root force/moment balance sets contact loads. Gravity/bias alone
        # on actuated DOFs omits this load path through the supporting feet.
        self.last['predicted_min_normal_n'] = float(min(wrench[2::6]))
        self.last['root_dynamics_residual'] = float(np.linalg.norm(root_j @ wrench - required[:6]))
        return (required - contact_j.T @ wrench)[r.vadr]

    def com_task_acceleration(self, jacobian):
        """Standing default; task controllers may add bounded reference feedforward."""
        c, d = self.config, self.r.data
        return c.com_kp * (self.target_com - d.subtree_com[self.pelvis]) - c.com_kd * jacobian @ d.qvel
