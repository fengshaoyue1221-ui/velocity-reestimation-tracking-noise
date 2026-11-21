"""
Adds two switches to the perception-noise module (defaults keep the original behavior,
so every existing result stays valid) and to the clean eval config:
  VEL_FROM_POS     (default True):  velocity is re-estimated from the noisy positions.
                                    False -> keep the clean velocity (plus VEL_STD noise).
  POS_FOR_VEL_ONLY (default False): True -> noisy positions are used ONLY to derive the
                                    velocity; the positions fed to the model are restored to clean.
The random draws are unchanged, so runs with the same seed see identical noise samples.
Idempotent. Run from the kit root:  python mechanism/enable_vel_switches.py
"""
import os
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mod = os.path.join(KIT, 'MTR/mtr/datasets/waymo/perception_noise.py')
s = open(mod).read()
if 'vel_from_pos' not in s:
    s = s.replace("        self.dt = float(cfg.get('DT', 0.1))\n",
                  "        self.dt = float(cfg.get('DT', 0.1))\n"
                  "        self.vel_from_pos = bool(cfg.get('VEL_FROM_POS', True))\n"
                  "        self.pos_for_vel_only = bool(cfg.get('POS_FOR_VEL_ONLY', False))\n", 1)
    old = ("        x[:, :, 7:9] = (1 - alpha) * trajs_past[:, :, 7:9] + alpha * fd\n"
           "        x[:, :, 7:9] += self.rng.standard_normal((N, T, 2)) * self.vel_std * L\n")
    new = ("        if self.vel_from_pos:\n"
           "            x[:, :, 7:9] = (1 - alpha) * trajs_past[:, :, 7:9] + alpha * fd\n"
           "        else:\n"
           "            x[:, :, 7:9] = trajs_past[:, :, 7:9]\n"
           "        x[:, :, 7:9] += self.rng.standard_normal((N, T, 2)) * self.vel_std * L\n"
           "        if self.pos_for_vel_only:\n"
           "            x[:, :, 0:2] = trajs_past[:, :, 0:2]\n")
    assert old in s, 'velocity block not found - perception_noise.py differs from expected'
    s = s.replace(old, new, 1)
    open(mod, 'w').write(s); print('patched', mod)
else:
    print('perception_noise.py already has velocity switches')
for cfg in ['a2000_clean.yaml', 'a2000_aug.yaml', 'a100_clean.yaml', 'a100_aug.yaml']:
    p = os.path.join(KIT, 'MTR/tools/cfgs/waymo', cfg)
    if not os.path.exists(p):
        continue
    c = open(p).read()
    if 'VEL_FROM_POS' in c:
        continue
    c = c.replace("        DT: 0.1\n", "        DT: 0.1\n        VEL_FROM_POS: True\n        POS_FOR_VEL_ONLY: False\n", 1)
    assert 'VEL_FROM_POS' in c, f'patching {cfg} failed'
    open(p, 'w').write(c); print('patched', p)
