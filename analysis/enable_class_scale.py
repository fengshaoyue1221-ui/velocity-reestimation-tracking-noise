"""
Adds an optional CLASS_SCALE override to the perception-noise module and to the
eval config, so the class-dependent noise factors (vehicle 1.0 / pedestrian 1.5 /
cyclist 1.3) can be changed at evaluation time with --set. Defaults are unchanged,
so all existing results stay valid. Idempotent: safe to run more than once.

Run from the kit root AFTER all training has finished:
    python analysis/enable_class_scale.py
"""
import os, re, sys
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mod = os.path.join(KIT, 'MTR/mtr/datasets/waymo/perception_noise.py')
s = open(mod).read()
if 'CLASS_SCALE' not in s:
    s = s.replace("        self.rng = np.random.default_rng(seed)\n",
                  "        self.rng = np.random.default_rng(seed)\n"
                  "        self.type_scale = dict(TYPE_SCALE)\n"
                  "        self.type_scale.update({k: float(v) for k, v in dict(cfg.get('CLASS_SCALE', {}) or {}).items()})\n", 1)
    s = s.replace("type_s = np.array([TYPE_SCALE.get(t, 1.0) for t in obj_types], dtype=np.float32)",
                  "type_s = np.array([self.type_scale.get(t, 1.0) for t in obj_types], dtype=np.float32)", 1)
    assert 'self.type_scale.get' in s, 'patching perception_noise.py failed'
    open(mod, 'w').write(s); print('patched', mod)
else:
    print('perception_noise.py already supports CLASS_SCALE')

for cfg in ['a2000_clean.yaml', 'a2000_aug.yaml', 'a100_clean.yaml', 'a100_aug.yaml']:
    p = os.path.join(KIT, 'MTR/tools/cfgs/waymo', cfg)
    if not os.path.exists(p):
        continue
    c = open(p).read()
    if 'CLASS_SCALE' in c:
        continue
    c = c.replace("        DT: 0.1\n",
                  "        DT: 0.1\n"
                  "        CLASS_SCALE:           # per-class noise multiplier (defaults = paper setting)\n"
                  "            TYPE_VEHICLE: 1.0\n"
                  "            TYPE_PEDESTRIAN: 1.5\n"
                  "            TYPE_CYCLIST: 1.3\n", 1)
    assert 'CLASS_SCALE' in c, f'patching {cfg} failed'
    open(p, 'w').write(c); print('patched', p)
