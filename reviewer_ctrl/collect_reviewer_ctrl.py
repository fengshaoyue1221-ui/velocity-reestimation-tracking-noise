"""Summarize the reviewer controls. Run from the repository root:  python reviewer_ctrl/collect_reviewer_ctrl.py"""
import glob, json, os, re
KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ROOT = os.path.join(KIT, 'MTR', 'output', 'waymo', 'a2000_clean')

def met(model, tag):
    for f in sorted(glob.glob(os.path.join(ROOT, model, 'eval', 'epoch_30', tag, 'log_eval_*.txt')), reverse=True):
        t = open(f).read()
        d = {k: float(v) for k, v in re.findall(r'^(mAP|minADE|minFDE|MissRate)\s*:\s*([-\d.]+)\s*$', t, re.M)}
        h = {}
        for c, hz, v in re.findall(r'^mAP - TYPE_(VEHICLE|PEDESTRIAN|CYCLIST)_(5|9|15)\s*:\s*([-\d.eE]+)', t, re.M):
            h.setdefault(hz, []).append(float(v))
        if h: d['mAP@3/5/8s'] = [sum(h[k]) / len(h[k]) for k in ('5', '9', '15')]
        if 'mAP' in d: return d
    return None

res = {}
rel = lambda d, b, k: 100 * (d[k] - b[k]) / b[k]
for model in ['clean', 'mix']:
    base = met(model, 'level_0.0'); res[model] = {'clean_input': base}
    print(f'\n================ {model}-trained model  (clean input mAP {base["mAP"]:.4f}) ================')
    chk = met(model, 'rc_check_1.0')
    if chk:
        ok = abs(chk['mAP'] - base['mAP']) < 5e-4
        print(f"[check] both switches on: mAP {chk['mAP']:.4f} vs clean {base['mAP']:.4f} -> {'PASS' if ok else 'FAIL: do not use item-5 results'}")
        res[model]['check'] = {'mAP': chk['mAP'], 'pass': ok}
    fd = met(model, 'rc_fd_clean')
    if fd:
        print(f"[item 1] clean positions + finite-difference velocity: mAP {fd['mAP']:.4f} ({rel(fd, base, 'mAP'):+.1f}%), "
              f"minADE {fd['minADE']:.3f} ({rel(fd, base, 'minADE'):+.1f}%)")
        res[model]['fd_clean'] = fd
    print('[item 5] pathway split (positions clean; velocity re-derived from jittered positions)')
    for L in ['1.0', '0.5']:
        both = met(model, 'mech_vel_derived' if L == '1.0' else f'rc_vel_derived_{L}')
        vf, ac = met(model, f'rc_velfeat_only_{L}'), met(model, f'rc_acc_only_{L}')
        for name, d in [('velocity + acceleration', both), ('velocity feature only', vf), ('acceleration feature only', ac)]:
            if d: print(f"   lambda={L:4s} {name:26s} mAP {d['mAP']:.4f} ({rel(d, base, 'mAP'):+6.1f}%)  minADE {rel(d, base, 'minADE'):+6.1f}%")
        res[model][f'pathway_{L}'] = {'both': both, 'vel_feat_only': vf, 'acc_only': ac}
    print('[item 2] paired interventions by severity (relative to clean input)')
    print(f"   {'lambda':6s} {'offset only':>12s} {'vel only':>10s} {'both':>8s} {'all':>8s} {'all, no re-est':>15s}")
    for L in ['0.35', '0.5', '0.75', '1.0']:
        tags = (['mech_pos_offset', 'mech_vel_derived', 'only_pos', 'level_1.0', 'mech_all_cleanvel'] if L == '1.0' else
                [f'rc_pos_offset_{L}', f'rc_vel_derived_{L}', f'rc_both_{L}', f'rc_all_{L}', f'rc_all_cleanvel_{L}'])
        ds = [met(model, t) for t in tags]
        res[model][f'paired_{L}'] = dict(zip(['offset', 'vel', 'both', 'all', 'all_noreest'], ds))
        cells = [f"{rel(d, base, 'mAP'):+.1f}%" if d else 'n/a' for d in ds]
        print(f"   {L:6s} {cells[0]:>12s} {cells[1]:>10s} {cells[2]:>8s} {cells[3]:>8s} {cells[4]:>15s}")
os.makedirs(os.path.join(KIT, 'reviewer_ctrl_outputs'), exist_ok=True)
json.dump(res, open(os.path.join(KIT, 'reviewer_ctrl_outputs', 'reviewer_ctrl.json'), 'w'), indent=1)
print('\nsaved reviewer_ctrl_outputs/reviewer_ctrl.json')
