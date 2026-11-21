"""
Mixed noise-aware training: with probability TRAIN_CLEAN_PROB a training scene is left clean,
otherwise it is corrupted as before. Default 0.0 = previous behavior (every training scene corrupted
when APPLY_TRAIN is True), so existing models/results are unaffected. Evaluation is never affected
(the switch only acts when the dataset is in training mode). Also creates cfgs/waymo/a2000_mix.yaml
(= a2000_aug.yaml + TRAIN_CLEAN_PROB: 0.5). Idempotent.  Run from kit root.
"""
import os
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(KIT, 'MTR/mtr/datasets/waymo/waymo_dataset.py'); s = open(p).read()
if 'TRAIN_CLEAN_PROB' not in s:
    old = ("            obj_trajs_past, _ = PerceptionNoise(self.noise_cfg, seed=seed)(\n"
           "                obj_trajs_past, obj_types, sdc_track_index, protect_mask=protect)\n")
    new = ("            p_clean = float(self.noise_cfg.get('TRAIN_CLEAN_PROB', 0.0))\n"
           "            skip_noise = self.training and p_clean > 0 and np.random.default_rng().random() < p_clean\n"
           "            if not skip_noise:\n"
           "                obj_trajs_past, _ = PerceptionNoise(self.noise_cfg, seed=seed)(\n"
           "                    obj_trajs_past, obj_types, sdc_track_index, protect_mask=protect)\n")
    assert old in s, 'waymo_dataset.py differs from expected'
    s = s.replace(old, new, 1); open(p, 'w').write(s); print('patched', p)
cdir = os.path.join(KIT, 'MTR/tools/cfgs/waymo')
for cfg in ['a2000_clean.yaml', 'a2000_aug.yaml']:
    q = os.path.join(cdir, cfg); c = open(q).read()
    if 'TRAIN_CLEAN_PROB' not in c:
        c = c.replace("        DT: 0.1\n", "        DT: 0.1\n        TRAIN_CLEAN_PROB: 0.0\n", 1)
        assert 'TRAIN_CLEAN_PROB' in c; open(q, 'w').write(c); print('patched', q)
mix = os.path.join(cdir, 'a2000_mix.yaml')
if not os.path.exists(mix):
    c = open(os.path.join(cdir, 'a2000_aug.yaml')).read()
    c = c.replace("        TRAIN_CLEAN_PROB: 0.0\n", "        TRAIN_CLEAN_PROB: 0.5\n", 1)
    assert 'TRAIN_CLEAN_PROB: 0.5' in c and 'APPLY_TRAIN: True' in c
    c = "# Mixed noise-aware training: 50% of training scenes clean, 50% corrupted at LEVEL 1.\n" + c
    open(mix, 'w').write(c); print('created', mix)
