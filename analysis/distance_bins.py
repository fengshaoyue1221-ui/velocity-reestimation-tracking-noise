"""
Degradation as a function of the target agent's distance to the ego vehicle.

Reads the result.pkl that MTR's test.py saves for every evaluation, computes simple
per-agent displacement errors (minADE over the 8 s horizon, minFDE at the last valid
future step, 6 modes), groups agents by their distance to the SDC at the current
frame (ground-truth positions), and compares a noisy evaluation with the clean one
of the SAME model.  These are plain displacement errors, not the official Waymo
metrics (which aggregate differently); use them only for relative comparisons.

Usage (from MTR root):
  python ../analysis/distance_bins.py --eval_root output/waymo/a2000_clean/clean/eval/epoch_30 \
      --clean level_0.0 --noisy level_1.0 --out ../paper/results/extra/dist_clean.json
"""
import argparse, glob, json, os, pickle
import numpy as np

BINS = [0, 10, 20, 30, 50, 1e9]
LABELS = ['0-10', '10-20', '20-30', '30-50', '>50']
TYPES = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']


def load_preds(eval_root, tag):
    f = os.path.join(eval_root, tag, 'result.pkl')
    preds = pickle.load(open(f, 'rb'))
    flat = []
    for p in preds:
        flat.extend(p if isinstance(p, list) else [p])
    return {(str(p['scenario_id']), int(p['object_id'])): p for p in flat}


def sdc_positions(data_dir, scenario_ids, cur=10):
    out = {}
    for sid in scenario_ids:
        info = pickle.load(open(os.path.join(data_dir, f'sample_{sid}.pkl'), 'rb'))
        out[sid] = info['track_infos']['trajs'][info['sdc_track_index'], info['current_time_index'], 0:2]
    return out


def errors(p, cur=10):
    gt = p['gt_trajs'][cur + 1:, 0:2]; valid = p['gt_trajs'][cur + 1:, 9] > 0
    if valid.sum() == 0:
        return None
    d = np.linalg.norm(p['pred_trajs'][:, :len(gt), :] - gt[None], axis=-1)     # (modes, T)
    ade = (d[:, valid].mean(axis=1)).min()
    last = np.where(valid)[0][-1]
    fde = d[:, last].min()
    return ade, fde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_root', required=True)
    ap.add_argument('--clean', default='level_0.0')
    ap.add_argument('--noisy', default='level_1.0')
    ap.add_argument('--data_dir', default='data/waymo/processed_scenarios_validation')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    C, N = load_preds(a.eval_root, a.clean), load_preds(a.eval_root, a.noisy)
    keys = sorted(set(C) & set(N))
    sdc = sdc_positions(a.data_dir, sorted({k[0] for k in keys}))
    rows = {}
    for k in keys:
        pc, pn = C[k], N[k]
        ec, en = errors(pc), errors(pn)
        if ec is None or en is None:
            continue
        dist = float(np.linalg.norm(pc['gt_trajs'][10, 0:2] - sdc[k[0]]))
        b = int(np.digitize(dist, BINS) - 1)
        for t in (pc['object_type'], 'ALL'):
            r = rows.setdefault((t, b), [0, 0.0, 0.0, 0.0, 0.0])
            r[0] += 1; r[1] += float(ec[0]); r[2] += float(en[0]); r[3] += float(ec[1]); r[4] += float(en[1])
    res = {}
    for (t, b), (n, ac, an, fc, fn) in sorted(rows.items()):
        res.setdefault(t, {})[LABELS[b]] = {
            'n': n, 'minADE_clean': ac / n, 'minADE_noisy': an / n,
            'minADE_rel_increase_%': 100 * (an - ac) / ac if ac > 0 else None,
            'minFDE_clean': fc / n, 'minFDE_noisy': fn / n,
            'minFDE_rel_increase_%': 100 * (fn - fc) / fc if fc > 0 else None}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({'clean': a.clean, 'noisy': a.noisy, 'bins_m': LABELS, 'results': res}, open(a.out, 'w'), indent=1)
    for t in ['ALL'] + TYPES:
        if t in res:
            print(t, {k: (v['n'], None if v['minADE_rel_increase_%'] is None else round(v['minADE_rel_increase_%'], 1)) for k, v in res[t].items()})
    print('saved', a.out)


if __name__ == '__main__':
    main()
