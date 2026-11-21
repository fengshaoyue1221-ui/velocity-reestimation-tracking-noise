"""
Add a causal constant-velocity Kalman-filter (KF) velocity estimator to the perception-noise model
(idempotent). New config keys (defaults keep all existing experiments unchanged):

  VEL_ESTIMATOR     'fd' (default) | 'kf'   velocity re-estimated by finite differences or by a causal KF
  KF_POSITIONS      False (default)         True: also replace positions by the KF-filtered positions
  FORCE_FULL_REEST  False (default)         True: alpha = 1 (full re-estimation) also for lambda < 1

The KF runs on exactly the same jittered positions as the finite-difference estimator (it draws no
random numbers), so FD and KF conditions are paired. Per axis the state is [p, v, eps] with white-
acceleration process noise q (2.0 m^2/s^3 vehicles/cyclists, 1.0 pedestrians) and the injected AR(1)
measurement noise model (known sigma_t and rho), i.e. a well-tuned filter. At each track's first valid
frame the velocity is initialized with the labeled velocity, matching the FD fallback.

Run from the repository root (env mtr):  python kf_ctrl/enable_kf.py
"""
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MTR')
PN = os.path.join(ROOT, 'mtr/datasets/waymo/perception_noise.py')

KF_FUNC = '''

def _kf_causal(z, valid, sig, v_init, rho, dt, q, r_jitter=1e-6):
    """Causal Kalman filter, batched over agents and axes.
    z: (N, T, 2) jittered positions, valid: (N, T), sig: (N, T) noise std, v_init: (N, T, 2) labeled velocity,
    q: (N,) process-noise intensity. Returns filtered positions and velocities, (N, T, 2) each."""
    N, T, _ = z.shape
    F = np.array([[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, rho]])
    Qpv = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    Q = np.zeros((N, 3, 3)); Q[:, :2, :2] = q[:, None, None] * Qpv; Q[:, 2, 2] = 1 - rho ** 2
    first = np.where(valid.any(1), np.argmax(valid, axis=1), T)
    out_p = z.astype(np.float64).copy(); out_v = np.zeros((N, T, 2))
    for ax in range(2):
        m = np.zeros((N, 3)); P = np.tile(np.eye(3), (N, 1, 1))
        for t in range(T):
            start = first == t
            if start.any():
                m[start] = np.stack([z[start, t, ax], v_init[start, t, ax], np.zeros(start.sum())], 1)
                P[start] = np.diag([1.0, 1.0, 1.0])
            run = (first < t)
            if run.any():
                m[run] = m[run] @ F.T
                P[run] = F @ P[run] @ F.T + Q[run]
            upd = valid[:, t] & (first <= t)
            if upd.any():
                H = np.zeros((N, 3)); H[:, 0] = 1.0; H[:, 2] = sig[:, t]
                S = np.einsum('ai,aij,aj->a', H, P, H) + r_jitter
                K = np.einsum('aij,aj->ai', P, H) / S[:, None]
                innov = z[:, t, ax] - np.einsum('ai,ai->a', H, m)
                m_new = m + K * innov[:, None]
                P_new = P - np.einsum('ai,aj->aij', K, np.einsum('ai,aij->aj', H, P))
                m = np.where(upd[:, None], m_new, m); P = np.where(upd[:, None, None], P_new, P)
            out_p[:, t, ax] = m[:, 0]; out_v[:, t, ax] = m[:, 1]
    return out_p.astype(np.float32), out_v.astype(np.float32)
'''

s = open(PN).read()
if 'VEL_ESTIMATOR' in s:
    print('already patched', PN)
else:
    a = "        self.pos_for_vel_only = bool(cfg.get('POS_FOR_VEL_ONLY', False))\n"
    assert a in s, 'anchor 1 not found (run mechanism/enable_vel_switches.py first)'
    s = s.replace(a, a + "        self.vel_estimator = str(cfg.get('VEL_ESTIMATOR', 'fd'))\n"
                         "        self.kf_positions = bool(cfg.get('KF_POSITIONS', False))\n"
                         "        self.force_full_reest = bool(cfg.get('FORCE_FULL_REEST', False))\n"
                         "        self.kf_q = {'TYPE_VEHICLE': 2.0, 'TYPE_PEDESTRIAN': 1.0, 'TYPE_CYCLIST': 2.0}\n", 1)
    b = "        alpha = min(1.0, L)  # blend: level 0 -> clean vel, level>=1 -> fully re-estimated\n"
    assert b in s, 'anchor 2 not found'
    s = s.replace(b, "        if self.vel_estimator == 'kf':  # causal Kalman filter on the same jittered positions\n"
                     "            q_agent = np.array([self.kf_q.get(t, 2.0) for t in obj_types], dtype=np.float64)\n"
                     "            kf_p, kf_v = _kf_causal(x[:, :, 0:2], valid, pos_sigma, trajs_past[:, :, 7:9],\n"
                     "                                    self.pos_rho, self.dt, q_agent)\n"
                     "            fd = kf_v\n"
                     "            if self.kf_positions:\n"
                     "                x[:, :, 0:2] = np.where(valid[..., None], kf_p, x[:, :, 0:2])\n"
                     "        alpha = 1.0 if self.force_full_reest else min(1.0, L)\n", 1)
    s = s.rstrip('\n') + '\n' + KF_FUNC
    open(PN, 'w').write(s); print('patched', PN)

for name in ['a2000_clean.yaml', 'a2000_novel.yaml']:
    y = os.path.join(ROOT, 'tools/cfgs/waymo', name)
    if not os.path.exists(y):
        continue
    c = open(y).read()
    if 'VEL_ESTIMATOR' in c:
        print('already patched', y); continue
    a = "        POS_FOR_VEL_ONLY: False\n"
    assert a in c, f'POS_FOR_VEL_ONLY not found in {y}'
    c = c.replace(a, a + "        VEL_ESTIMATOR: fd\n        KF_POSITIONS: False\n        FORCE_FULL_REEST: False\n", 1)
    open(y, 'w').write(c); print('patched', y)
