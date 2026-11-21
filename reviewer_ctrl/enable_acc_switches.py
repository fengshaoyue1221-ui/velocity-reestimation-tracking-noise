"""
Add two evaluation switches to the perception-noise config (idempotent):

  ACC_FROM_CLEAN  (default False): the acceleration feature is computed from the CLEAN labeled velocities,
                                   while the velocity feature keeps the (corrupted / re-estimated) velocity.
  VEL_FEAT_CLEAN  (default False): the velocity feature uses the CLEAN labeled velocities,
                                   while the acceleration feature is computed from the corrupted velocity.

Both are rotated into each target frame exactly like MTR's own velocity channels. With both switches on
and clean positions, the input equals the clean input (used as a correctness check).

Run from the repository root (env mtr):  python reviewer_ctrl/enable_acc_switches.py
"""
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MTR')
DS = os.path.join(ROOT, 'mtr/datasets/waymo/waymo_dataset.py')

s = open(DS).read()
if '_vel_override' in s:
    print('already patched', DS)
else:
    # 1) decide the override once per scene, right before the no-velocity ablation hook
    anchor = "        if self.dataset_cfg.get('DROP_VELOCITY_INPUT', False):  # no-velocity ablation\n"
    assert anchor in s, 'anchor 1 not found'
    block = ("        self._vel_override = None  # reviewer controls: separate velocity and acceleration pathways\n"
             "        if self.noise_cfg is not None and (self.noise_cfg.get('ACC_FROM_CLEAN', False) or self.noise_cfg.get('VEL_FEAT_CLEAN', False)):\n"
             "            clean_v = obj_trajs_full_clean[:, :current_time_index + 1, 7:9].copy()\n"
             "            self._vel_override = {'acc': clean_v if self.noise_cfg.get('ACC_FROM_CLEAN', False) else None,\n"
             "                                  'feat': clean_v if self.noise_cfg.get('VEL_FEAT_CLEAN', False) else None}\n\n")
    s = s.replace(anchor, block + anchor, 1)
    # 2) apply it where MTR builds the velocity and acceleration features
    anchor2 = ("        acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)\n"
               "        acce[:, :, 0, :] = acce[:, :, 1, :]\n")
    assert anchor2 in s, 'anchor 2 not found'
    block2 = ("        ov = getattr(self, '_vel_override', None)\n"
              "        if ov is not None:\n"
              "            hd = torch.as_tensor(np.asarray(center_objects)[:, 6]).float()\n"
              "            cs, sn = torch.cos(-hd)[:, None, None], torch.sin(-hd)[:, None, None]\n"
              "            def _rot(v):  # (num_objects, T, 2) world frame -> (num_center, num_objects, T, 2) target frame\n"
              "                v = torch.as_tensor(v).float()\n"
              "                return torch.stack([cs * v[None, :, :, 0] - sn * v[None, :, :, 1],\n"
              "                                    sn * v[None, :, :, 0] + cs * v[None, :, :, 1]], dim=-1)\n"
              "            if ov.get('acc') is not None:\n"
              "                va = _rot(ov['acc'])\n"
              "                acce = (va - torch.roll(va, shifts=1, dims=2)) / 0.1\n"
              "                acce[:, :, 0, :] = acce[:, :, 1, :]\n"
              "            if ov.get('feat') is not None:\n"
              "                obj_trajs[:, :, :, 7:9] = _rot(ov['feat'])\n")
    s = s.replace(anchor2, anchor2 + block2, 1)
    open(DS, 'w').write(s)
    print('patched', DS)

for name in ['a2000_clean.yaml', 'a2000_novel.yaml']:
    y = os.path.join(ROOT, 'tools/cfgs/waymo', name)
    if not os.path.exists(y):
        continue
    c = open(y).read()
    if 'ACC_FROM_CLEAN' in c:
        print('already patched', y); continue
    a = "        POS_FOR_VEL_ONLY: False\n"
    assert a in c, f'POS_FOR_VEL_ONLY not found in {y} (run mechanism/enable_vel_switches.py first)'
    c = c.replace(a, a + "        ACC_FROM_CLEAN: False\n        VEL_FEAT_CLEAN: False\n", 1)
    open(y, 'w').write(c); print('patched', y)
