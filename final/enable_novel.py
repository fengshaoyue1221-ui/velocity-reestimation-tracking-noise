"""
No-velocity ablation: DATA_CONFIG.DROP_VELOCITY_INPUT (default False) zeroes the observed
velocities of all agents before feature construction. MTR derives its acceleration feature from
velocity differences, so both motion-state features become zero; positions, headings and sizes are
kept. Applies in training and evaluation alike. Creates cfgs/waymo/a2000_novel.yaml
(= a2000_clean.yaml + DROP_VELOCITY_INPUT: True). Defaults leave all existing results unchanged.
Idempotent.  Run from kit root.
"""
import os
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(KIT, 'MTR/mtr/datasets/waymo/waymo_dataset.py'); s = open(p).read()
if 'DROP_VELOCITY_INPUT' not in s:
    anchor = "        center_objects, track_index_to_predict = self.get_interested_agents(\n"
    assert s.count(anchor) == 1, 'waymo_dataset.py differs from expected'
    s = s.replace(anchor,
        "        if self.dataset_cfg.get('DROP_VELOCITY_INPUT', False):  # no-velocity ablation\n"
        "            obj_trajs_past = obj_trajs_past.copy()\n"
        "            obj_trajs_past[:, :, 7:9] = 0.0\n" + anchor, 1)
    open(p, 'w').write(s); print('patched', p)
cdir = os.path.join(KIT, 'MTR/tools/cfgs/waymo')
for cfg in ['a2000_clean.yaml', 'a2000_aug.yaml', 'a2000_mix.yaml']:
    q = os.path.join(cdir, cfg)
    if not os.path.exists(q): continue
    c = open(q).read()
    if 'DROP_VELOCITY_INPUT' not in c:
        c = c.replace("DATA_CONFIG:\n    DATASET: WaymoDataset\n",
                      "DATA_CONFIG:\n    DATASET: WaymoDataset\n    DROP_VELOCITY_INPUT: False\n", 1)
        assert 'DROP_VELOCITY_INPUT' in c; open(q, 'w').write(c); print('patched', q)
nv = os.path.join(cdir, 'a2000_novel.yaml')
if not os.path.exists(nv):
    c = open(os.path.join(cdir, 'a2000_clean.yaml')).read()
    c = c.replace("    DROP_VELOCITY_INPUT: False\n", "    DROP_VELOCITY_INPUT: True\n", 1)
    assert 'DROP_VELOCITY_INPUT: True' in c and 'APPLY_TRAIN: False' in c
    open(nv, 'w').write("# No-velocity ablation: clean training, observed velocities zeroed (train and eval).\n" + c)
    print('created', nv)
