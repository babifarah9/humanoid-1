"""Small NumPy contact-wrench QP for slow, flat-floor motion experiments.

The linear friction/CoP/torsion bounds are an approximation to contact-wrench
feasibility, not a hardware guarantee. No computed wrench is applied externally.
"""
import numpy as np
import mujoco
from .whole_body import WholeBodyController, full_mass, rotation_error


class InfeasibleContact(ValueError):
    pass


def allocate_wrenches(hessian, target, root_map, required, inequalities):
    """Minimize weighted wrench error with root balance and C @ wrench >= 0.

    Deterministic active-set solve, with residual checks and a bounded iteration
    count. A failed solve stops motion; it never falls back to an unconstrained
    wrench. Multipliers for active C rows must be nonpositive in this convention.
    """
    H, target, A, b, C = (np.asarray(x, dtype=float) for x in
                          (hessian, target, root_map, required, inequalities))
    n = target.size
    if (target.shape != (n,) or H.shape != (n, n) or A.shape != (6, n)
            or b.shape != (6,) or C.ndim != 2 or C.shape[1] != n or not len(C)
            or not all(np.isfinite(x).all() for x in (H, target, A, b, C))):
        raise InfeasibleContact('invalid wrench allocation data')
    if not np.allclose(H, H.T) or np.min(np.linalg.eigvalsh(H)) <= 0:
        raise InfeasibleContact('wrench objective must be positive definite')
    active = []
    for iteration in range(64):
        E = np.vstack((A, C[active])) if active else A
        rhs = np.r_[b, np.zeros(len(active))]
        K = np.block([[H, E.T], [E, np.zeros((len(rhs), len(rhs)))]])
        solution = np.linalg.lstsq(K, np.r_[H @ target, rhs], rcond=1e-11)[0]
        wrench = solution[:n]
        margins = C @ wrench
        remaining = [i for i in range(len(C)) if i not in active]
        if remaining and min(margins[remaining]) < -1e-6:
            active.append(min(remaining, key=lambda i: margins[i]))
            continue
        if active:
            multipliers = solution[n + 6:]
            if max(multipliers) > 1e-6:
                active.pop(int(np.argmax(multipliers)))
                continue
        residual = float(np.max(np.abs(E @ wrench - rhs)))
        margin = float(np.min(margins))
        if not np.isfinite(wrench).all() or residual > 1e-4 or margin < -1e-5:
            raise InfeasibleContact('contact balance/bounds infeasible')
        return wrench, dict(root_residual=float(np.linalg.norm(A @ wrench - b)),
                            minimum_inequality_margin=margin, iterations=iteration + 1)
    raise InfeasibleContact('contact allocation iteration limit')


def contact_bounds(robot, foot_geoms, active, friction=.6, edge_margin=.01):
    """World wrenches at foot body origins, bounded in each foot's local frame.

    Foot boxes are offset from the ankle. Account for the offset when bounding
    CoP, rather than assuming the ankle is the center of the support rectangle.
    The default friction coefficient is deliberately below the model's 1.2.
    """
    m, d = robot.model, robot.data
    blocks = []
    for column, i in enumerate(active):
        geom = foot_geoms[i]
        if m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise InfeasibleContact('contact bounds require box feet')
        sx, sy, _ = m.geom_size[geom]
        ox, oy, _ = m.geom_pos[geom]
        xmin, xmax = ox - sx + edge_margin, ox + sx - edge_margin
        ymin, ymax = oy - sy + edge_margin, oy + sy - edge_margin
        local = [np.array([0., 0., 1., 0., 0., 0.])]
        for axis in (0, 1):
            for sign in (-1., 1.):
                v = np.zeros(6); v[2] = friction; v[axis] = sign
                local.append(v)
        # Mx = y * Fz, My = -x * Fz for a horizontal supporting sole.
        for axis, sign, coefficient in ((3, 1., -ymin), (3, -1., ymax),
                                         (4, -1., -xmin), (4, 1., xmax),
                                         (5, 1., .03), (5, -1., .03)):
            v = np.zeros(6); v[2] = coefficient; v[axis] = sign
            local.append(v)
        rotation = d.xmat[m.geom_bodyid[geom]].reshape(3, 3).T
        transform = np.zeros((6, 6))
        transform[:3, :3] = rotation; transform[3:, 3:] = rotation
        block = np.zeros((len(local), 6 * len(active)))
        block[:, 6 * column:6 * column + 6] = np.asarray(local) @ transform
        blocks.append(block)
    if not blocks:
        raise InfeasibleContact('no planned support foot')
    return np.vstack(blocks)


class ContactController(WholeBodyController):
    """Separate contact-scheduled controller; standing controller is unchanged.

    Adds swing-foot tasks, COM and torso Jdot terms and bounded wrench allocation.
    The task supplies an explicit support schedule; unexpected loss of a planned
    support remains a stop condition. Only the task may designate a swing foot.
    """
    def __init__(self, robot):
        super().__init__(robot)
        self.home_com = self.target_com.copy()
        self.home_feet = np.asarray(self.foot_positions).copy()
        self.com_velocity = np.zeros(3); self.com_acceleration = np.zeros(3)
        self.foot_velocity = np.zeros((2, 3)); self.foot_acceleration = np.zeros((2, 3))
        self.planned_support = (0, 1)

    def act(self, residual=None):
        r, d = self.r, self.r.data
        if residual is not None:
            r.emergency_stop('stepping does not accept unverified policy residuals')
        if r.stopped:
            return np.zeros(r.nu)
        reason = r.safety_reason()
        if reason:
            r.emergency_stop(reason)
            return np.zeros(r.nu)
        self.last = self.diagnostics()
        normal, _, nonfoot = self.contacts()
        self.last_loaded[normal >= self.config.min_normal_n] = d.time
        active = self.planned_support
        if active not in ((0,), (1,), (0, 1)):
            reason = 'invalid planned support schedule'
        elif nonfoot:
            reason = 'non-foot ground contact'
        elif self.last['torso_tilt_rad'] > .10:
            reason = 'stepping tilt envelope exceeded'
        elif r.limit_events:
            reason = 'stepping nominal joint-limit event'
        elif d.time > self.config.startup_s and any(
                d.time - self.last_loaded[i] > self.config.contact_timeout_s for i in active):
            reason = 'planned support contact lost'
        if reason:
            r.emergency_stop(reason)
            return np.zeros(r.nu)
        try:
            torque = self._motion_torques(active)
        except (np.linalg.LinAlgError, FloatingPointError, InfeasibleContact) as exc:
            r.emergency_stop('stepping contact solve failed: ' + str(exc))
            return np.zeros(r.nu)
        if not np.isfinite(torque).all():
            r.emergency_stop('non-finite stepping torque')
            return np.zeros(r.nu)
        self.last['max_requested_torque_nm'] = float(np.max(np.abs(torque)))
        self.last['n_saturated'] = int(np.sum((torque < r.ctrl_limits[:, 0]) | (torque > r.ctrl_limits[:, 1])))
        return np.clip(torque, r.ctrl_limits[:, 0], r.ctrl_limits[:, 1])

    def _motion_torques(self, active):
        r, m, d = self.r, self.r.model, self.r.data
        rows, targets, swing_rows, swing_targets = [], [], [], []
        for i, body in enumerate(self.feet):
            jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
            jdp = np.zeros_like(jp); jdr = np.zeros_like(jp)
            mujoco.mj_jacBody(m, d, jp, jr, body)
            mujoco.mj_jacDot(m, d, jdp, jdr, d.xpos[body], body)
            jac = np.vstack((jp, jr))
            error = np.r_[self.foot_positions[i] - d.xpos[body],
                rotation_error(d.xmat[body].reshape(3, 3), self.foot_rotations[i])]
            acceleration = (150. * error - 25. * (jac @ d.qvel - np.r_[self.foot_velocity[i], np.zeros(3)])
                + np.r_[self.foot_acceleration[i], np.zeros(3)] - np.vstack((jdp, jdr)) @ d.qvel)
            if i in active:
                rows.append(jac); targets.append(acceleration)
            else:
                swing_rows.append(30. * jac); swing_targets.append(30. * acceleration)
        contact_jac = np.vstack(rows)
        jc = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, jc, self.pelvis)
        jdc = np.zeros_like(jc)
        for body in range(self.pelvis, m.nbody):
            if body > self.pelvis and m.body_parentid[body] < self.pelvis:
                break
            derivative = np.zeros_like(jc)
            mujoco.mj_jacDot(m, d, derivative, None, d.xipos[body], body)
            jdc += derivative * m.body_mass[body] / m.body_subtreemass[self.pelvis]
        jt = np.zeros_like(jc); jdt = np.zeros_like(jc)
        mujoco.mj_jacBody(m, d, None, jt, self.torso)
        mujoco.mj_jacDot(m, d, None, jdt, d.xpos[self.torso], self.torso)
        q = d.qpos[r.qadr]
        posture = 20. * (self.target_q - q) - 8. * d.qvel[r.vadr]
        posture += 100. * (np.maximum(r.limits[:, 0] + .04 - q, 0.) - np.maximum(q - r.limits[:, 1] + .04, 0.))
        com_acc = (60. * (self.target_com - d.subtree_com[self.pelvis])
            - 16. * (jc @ d.qvel - self.com_velocity) + self.com_acceleration - jdc @ d.qvel)
        torso_acc = (80. * rotation_error(d.xmat[self.torso].reshape(3, 3), self.target_torso)
            - 18. * jt @ d.qvel - jdt @ d.qvel)
        A = np.vstack([self.selection, 30. * jc, 10. * jt] + swing_rows)
        b = np.concatenate([posture, 30. * com_acc, 10. * torso_acc] + swing_targets)
        nc = len(contact_jac)
        K = np.block([[A.T @ A + np.eye(m.nv) * 1e-6, contact_jac.T],
                      [contact_jac, np.zeros((nc, nc))]])
        acceleration = np.linalg.lstsq(K, np.r_[A.T @ b, np.concatenate(targets)], rcond=1e-10)[0][:m.nv]
        required = full_mass(m, d) @ acceleration + d.qfrc_bias - d.qfrc_passive
        hessian = np.diag(np.tile([1., 1., .1, 10., 10., 10.], len(active)))
        target = np.zeros(nc)
        target[2::6] = m.body_subtreemass[self.pelvis] * 9.81 / len(active)
        wrench, allocation = allocate_wrenches(hessian, target, contact_jac.T[:6], required[:6],
                                               contact_bounds(r, self.foot_geoms, active))
        self.last.update(allocation)
        self.last['predicted_support_wrenches'] = wrench.reshape(-1, 6).tolist()
        return (required - contact_jac.T @ wrench)[r.vadr]
