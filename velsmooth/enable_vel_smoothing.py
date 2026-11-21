"""
Training-free mitigation: at test time, re-estimate every agent's velocity by a least-squares
line fit over its last K observed positions (instead of trusting the per-frame velocity).
Adds smooth_velocity() to perception_noise.py, a hook in waymo_dataset.py, and the config key
VEL_SMOOTH_K (default 0 = off, so all existing results are unchanged). Idempotent.
Run from the kit root:  python velsmooth/enable_vel_smoothing.py
"""
import os
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(KIT, 'MTR/mtr/datasets/waymo')

FUNC = '''

def smooth_velocity(trajs, k, dt=0.1, skip_idx=None):
    """Least-squares velocity over the last k valid frames (per agent, per time step).
    trajs: (N, T, 10) with x,y at [0:2], vx,vy at [7:9], valid at [9]. Frames with fewer than
    2 valid points in the window keep their given velocity. Agent skip_idx (ego) is untouched."""
    out = trajs.copy()
    N, T, _ = trajs.shape
    valid = trajs[:, :, 9] > 0
    t = np.arange(T, dtype=np.float64) * dt
    for e in range(T):
        s = max(0, e - k + 1)
        w = valid[:, s:e + 1].astype(np.float64)
        n = w.sum(1)
        ok = (n >= 2) & valid[:, e]
        if not ok.any():
            continue
        tt = t[s:e + 1][None]
        tm = (w * tt).sum(1) / np.maximum(n, 1)
        dtc = tt - tm[:, None]
        den = (w * dtc ** 2).sum(1)
        ok &= den > 0
        for d in range(2):
            p = trajs[:, s:e + 1, d].astype(np.float64)
            pm = (w * p).sum(1) / np.maximum(n, 1)
            v = (w * dtc * (p - pm[:, None])).sum(1) / np.maximum(den, 1e-12)
            out[ok, e, 7 + d] = v[ok]
    if skip_idx is not None and 0 <= skip_idx < N:
        out[skip_idx] = trajs[skip_idx]
    return out.astype(np.float32)
'''
p = os.path.join(W, 'perception_noise.py'); s = open(p).read()
if 'def smooth_velocity' not in s:
    open(p, 'w').write(s.rstrip('\n') + '\n' + FUNC); print('patched', p)

p = os.path.join(W, 'waymo_dataset.py'); s = open(p).read()
if 'VEL_SMOOTH_K' not in s:
    s = s.replace('from mtr.datasets.waymo.perception_noise import PerceptionNoise\n',
                  'from mtr.datasets.waymo.perception_noise import PerceptionNoise, smooth_velocity\n', 1)
    old = ("                obj_trajs_past, obj_types, sdc_track_index, protect_mask=protect)\n")
    new = old + ("            k_smooth = int(self.noise_cfg.get('VEL_SMOOTH_K', 0))\n"
                 "            if k_smooth > 1:  # test-time least-squares velocity re-estimation\n"
                 "                obj_trajs_past = smooth_velocity(obj_trajs_past, k_smooth,\n"
                 "                                                 float(self.noise_cfg.get('DT', 0.1)), skip_idx=sdc_track_index)\n")
    assert old in s and 'smooth_velocity' in s, 'waymo_dataset.py differs from expected'
    s = s.replace(old, new, 1)
    open(p, 'w').write(s); print('patched', p)

for cfg in ['a2000_clean.yaml', 'a2000_aug.yaml', 'a100_clean.yaml', 'a100_aug.yaml']:
    p = os.path.join(KIT, 'MTR/tools/cfgs/waymo', cfg)
    if os.path.exists(p):
        c = open(p).read()
        if 'VEL_SMOOTH_K' not in c:
            c = c.replace("        DT: 0.1\n", "        DT: 0.1\n        VEL_SMOOTH_K: 0\n", 1)
            assert 'VEL_SMOOTH_K' in c
            open(p, 'w').write(c); print('patched', p)
