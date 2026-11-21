"""
Residual between the finite-difference velocity and the labeled velocity on CLEAN histories:
    r_i^t = (p_i^t - p_i^{t-1}) / dt - v_i^t,   t = -9..0, both frames valid
This checks the assumption of Proposition 1 that clean labels satisfy (p^t - p^{t-1})/dt ~= v^t.

Usage (from MTR/, env mtr):  python ../missing_tools/residual_stats.py --split validation
Output: missing_outputs/residual_stats_<split>.json
"""
import argparse, glob, json, os, pickle
from multiprocessing import Pool
import numpy as np

DT, CUR = 0.1, 10
CLASSES = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
SPEED_BINS = [(0, 1), (1, 5), (5, 10), (10, 1e9)]


def one(path):
    info = pickle.load(open(path, 'rb'))
    tr = info['track_infos']; x = tr['trajs'][:, :CUR + 1]; types = np.array(tr['object_type'])
    targets = set(int(k) for k in info['tracks_to_predict']['track_index'])
    valid = x[:, :, 9] > 0
    both = valid[:, 1:] & valid[:, :-1]
    fd = (x[:, 1:, 0:2] - x[:, :-1, 0:2]) / DT
    r = fd - x[:, 1:, 7:9]
    speed = np.linalg.norm(x[:, 1:, 7:9], axis=-1)
    out = []
    for i in range(x.shape[0]):
        if types[i] not in CLASSES:
            continue
        m = both[i]
        if not m.any():
            continue
        out.append((types[i], i in targets, r[i][m].astype(np.float32), speed[i][m].astype(np.float32)))
    return out


def summarize(r, s):
    n = np.linalg.norm(r, axis=1)
    d = {'count': int(len(n)), 'mean_xy': r.mean(0).round(5).tolist(), 'std_xy': r.std(0).round(5).tolist(),
         'rms_norm': float(np.sqrt((n ** 2).mean())), 'mean_norm': float(n.mean()),
         'p50_norm': float(np.percentile(n, 50)), 'p90_norm': float(np.percentile(n, 90)), 'p99_norm': float(np.percentile(n, 99))}
    d['by_speed'] = {}
    for lo, hi in SPEED_BINS:
        k = (s >= lo) & (s < hi)
        if k.any():
            d['by_speed'][f'{lo}-{hi if hi < 1e9 else "inf"} m/s'] = {'count': int(k.sum()), 'mean_norm': float(n[k].mean()),
                                                                     'p90_norm': float(np.percentile(n[k], 90))}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='validation', choices=['validation', 'training'])
    ap.add_argument('--max_scenes', type=int, default=0, help='0 = all')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', default='../missing_outputs')
    a = ap.parse_args()
    files = sorted(glob.glob(f'data/waymo/processed_scenarios_{a.split}/sample_*.pkl'))
    if a.max_scenes:
        files = files[:a.max_scenes]
    print(f'{len(files)} scenes')
    with Pool(a.workers) as p:
        rows = [x for part in p.imap_unordered(one, files, chunksize=64) for x in part]
    res = {'split': a.split, 'scenes': len(files), 'definition': '(p^t - p^{t-1})/0.1 - v^t on clean labels, past frames'}
    for name, sel in [('all_agents', lambda t, g: True), ('targets', lambda t, g: g)]:
        res[name] = {}
        for c in CLASSES + ['ALL']:
            rr = [(r, s) for t, g, r, s in rows if sel(t, g) and (c == 'ALL' or t == c)]
            if rr:
                res[name][c] = summarize(np.concatenate([r for r, _ in rr]), np.concatenate([s for _, s in rr]))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'residual_stats_{a.split}.json')
    json.dump(res, open(path, 'w'), indent=1)
    t = res['all_agents']['ALL']
    print(f"ALL agents: n={t['count']}  mean|r|={t['mean_norm']:.4f} m/s  RMS={t['rms_norm']:.4f}  p99={t['p99_norm']:.4f}")
    print('saved', path)


if __name__ == '__main__':
    main()
