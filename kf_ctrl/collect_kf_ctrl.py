"""Summarize the Kalman-filter control. Run from the repository root:  python kf_ctrl/collect_kf_ctrl.py"""
import glob, json, os, re
KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ROOT = os.path.join(KIT, 'MTR', 'output', 'waymo', 'a2000_clean')
def met(model, tag):
    for f in sorted(glob.glob(os.path.join(ROOT, model, 'eval', 'epoch_30', tag, 'log_eval_*.txt')), reverse=True):
        d = {k: float(v) for k, v in re.findall(r'^(mAP|minADE|minFDE|MissRate)\s*:\s*([-\d.]+)\s*$', open(f).read(), re.M)}
        if 'mAP' in d: return d
    return None
res = {}
for model in ['clean', 'mix']:
    base = met(model, 'level_0.0'); res[model] = {'clean_input': base}
    print(f'\n================ {model}-trained model  (clean input mAP {base["mAP"]:.4f}) ================')
    a, b = met(model, 'kc_check_fd_1.0'), met(model, 'only_pos')
    if a and b:
        ok = abs(a['mAP'] - b['mAP']) < 5e-4
        print(f"[check] FD with forced full re-estimation at lambda=1: {a['mAP']:.4f} vs paper's position-jitter run {b['mAP']:.4f} -> {'PASS' if ok else 'FAIL'}")
    print(f"   {'lambda':6s} | {'FD velocity':>22s} | {'KF velocity':>22s} | {'KF position+velocity':>22s}")
    for L in ['1.0', '0.75', '0.5']:
        fd = met(model, 'only_pos' if L == '1.0' else f'kc_fd_{L}')
        kv, kf = met(model, f'kc_kfvel_{L}'), met(model, f'kc_kffull_{L}')
        cell = lambda d: f"{d['mAP']:.4f} ({100 * (d['mAP'] - base['mAP']) / base['mAP']:+6.1f}%)" if d else 'n/a'
        print(f"   {L:6s} | {cell(fd):>22s} | {cell(kv):>22s} | {cell(kf):>22s}")
        res[model][L] = {'fd': fd, 'kf_vel': kv, 'kf_full': kf}
os.makedirs(os.path.join(KIT, 'reviewer_ctrl_outputs'), exist_ok=True)
json.dump(res, open(os.path.join(KIT, 'reviewer_ctrl_outputs', 'kf_ctrl.json'), 'w'), indent=1)
print('\nsaved reviewer_ctrl_outputs/kf_ctrl.json')
