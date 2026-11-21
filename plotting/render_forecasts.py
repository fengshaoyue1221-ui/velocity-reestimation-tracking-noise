"""
Qualitative forecast comparison for the paper (CPU only, does not touch the GPUs).
For agents whose forecast degrades most under noise, draws three panels:
  (1) clean-trained MTR, clean input   (2) clean-trained MTR, noisy input (lambda=1)
  (3) mixed MTR, noisy input
The noisy history is regenerated with the exact evaluation seed, so it is the input the models saw.

Usage (from kit root, env mtr active):
  python figtools/render_forecasts.py            -> paper/results/forecast_examples/
"""
import argparse, os, pickle, sys, zlib
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MTR = os.path.join(KIT, 'MTR')
sys.path.insert(0, MTR)
from mtr.datasets.waymo.perception_noise import PerceptionNoise  # noqa: E402

EV = os.path.join(MTR, 'output/waymo/a2000_clean/{tag}/eval/epoch_30/{et}/result.pkl')
DATA = os.path.join(MTR, 'data/waymo/processed_scenarios_validation')
CUR = 10


def load_preds(tag, et):
    preds = pickle.load(open(EV.format(tag=tag, et=et), 'rb'))
    flat = [p for scene in preds for p in (scene if isinstance(scene, list) else [scene])]
    return {(str(p['scenario_id']), int(p['object_id'])): p for p in flat}


def errs(p):
    gt = p['gt_trajs'][CUR + 1:, 0:2]; v = p['gt_trajs'][CUR + 1:, 9] > 0
    d = np.linalg.norm(p['pred_trajs'][:, :len(gt)] - gt[None], axis=-1)
    last = np.where(v)[0][-1]
    top = int(np.argmax(p['pred_scores']))
    return float(d[:, v].mean(1).min()), float(d[:, last].min()), float(d[top, last])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(KIT, 'paper/results/forecast_examples'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    noise_cfg = dict(yaml.safe_load(open(os.path.join(MTR, 'tools/cfgs/waymo/a2000_clean.yaml')))['DATA_CONFIG']['PERCEPTION_NOISE'])
    noise_cfg['LEVEL'] = 1.0
    print('loading predictions ...')
    C0, C1, M1 = load_preds('clean', 'level_0.0'), load_preds('clean', 'level_1.0'), load_preds('mix', 'level_1.0')
    keys = sorted(set(C0) & set(C1) & set(M1))
    cand = []
    for k in keys:
        p0 = C0[k]
        if p0['object_type'] != 'TYPE_VEHICLE':
            continue
        g = p0['gt_trajs']
        if g[CUR + 1:, 9].min() <= 0:                   # need a fully observed future
            continue
        speed = np.linalg.norm(g[CUR, 7:9])
        if speed < 4.0:                                  # moving vehicles only
            continue
        ade0, fde0, _ = errs(p0); ade1, fde1, _ = errs(C1[k]); adem, fdem, _ = errs(M1[k])
        if fde0 > 2.0:                                   # clean forecast must be good
            continue
        cand.append((fde1 - fde0, k, (ade0, fde0), (ade1, fde1), (adem, fdem)))
    cand.sort(reverse=True)
    # keep at most one agent per scenario, and only those within 40 m of the ego vehicle
    chosen, seen = [], set()
    for c in cand:
        sid = c[1][0]
        if sid in seen:
            continue
        info = pickle.load(open(os.path.join(DATA, f'sample_{sid}.pkl'), 'rb'))
        tr = info['track_infos']; ids = list(tr['object_id'])
        sdc = info['sdc_track_index']; k_idx = ids.index(c[1][1])
        if np.linalg.norm(tr['trajs'][k_idx, CUR, :2] - tr['trajs'][sdc, CUR, :2]) > 40:
            continue
        chosen.append((c, info, k_idx)); seen.add(sid)
        if len(chosen) >= a.n:
            break
    print(f'{len(cand)} candidates, rendering {len(chosen)}')
    thumbs = []
    for rank, (c, info, k_idx) in enumerate(chosen):
        _, key, e0, e1, em = c
        sid = key[0]
        tr = info['track_infos']; trajs = tr['trajs']; types = np.array(tr['object_type'])
        protect = np.zeros(len(types), bool); protect[info['tracks_to_predict']['track_index']] = True
        seed = zlib.crc32(str(sid).encode()) + int(noise_cfg.get('SEED', 0))
        noisy, _ = PerceptionNoise(noise_cfg, seed=seed)(trajs[:, :CUR + 1], types, info['sdc_track_index'], protect)
        gt = trajs[k_idx, CUR + 1:, :2]
        center = trajs[k_idx, CUR, :2]
        pts = np.concatenate([gt, C0[key]['pred_trajs'].reshape(-1, 2), C1[key]['pred_trajs'].reshape(-1, 2),
                              M1[key]['pred_trajs'].reshape(-1, 2), trajs[k_idx, :CUR + 1, :2]])
        lo, hi = pts.min(0) - 6, pts.max(0) + 6
        span = max(hi - lo); mid = (lo + hi) / 2; lo, hi = mid - span / 2, mid + span / 2
        poly = info['map_infos']['all_polylines']
        inwin = (poly[:, 0] > lo[0]) & (poly[:, 0] < hi[0]) & (poly[:, 1] > lo[1]) & (poly[:, 1] < hi[1])
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
        panels = [('Clean-trained, clean input', trajs, C0[key], e0),
                  (r'Clean-trained, noisy input ($\lambda$=1)', noisy, C1[key], e1),
                  (r'Mixed, noisy input ($\lambda$=1)', noisy, M1[key], em)]
        for ax, (title, hist, pred, (ade, fde)) in zip(axes, panels):
            ax.scatter(poly[inwin, 0], poly[inwin, 1], s=0.15, c='0.8', rasterized=True)
            for j in range(len(types)):                  # other agents: current position
                if j != k_idx and hist[j, CUR, 9] > 0:
                    q = hist[j, CUR, :2]
                    if lo[0] < q[0] < hi[0] and lo[1] < q[1] < hi[1]:
                        ax.plot(*q, 's', ms=2, c='0.55')
            h = hist[k_idx, :CUR + 1]; vh = h[:, 9] > 0
            ax.plot(h[vh, 0], h[vh, 1], '-o', ms=1.6, lw=1.1, c='#1f77b4', label='observed history')
            ax.plot(gt[:, 0], gt[:, 1], '--', lw=1.1, c='k', label='ground-truth future')
            order = np.argsort(-pred['pred_scores'])
            for r_, m in enumerate(order):
                ax.plot(pred['pred_trajs'][m, :, 0], pred['pred_trajs'][m, :, 1], '-', lw=1.6 if r_ == 0 else 0.8,
                        c='#d62728', alpha=1.0 if r_ == 0 else 0.35, label='forecast (top-1 bold)' if r_ == 0 else None)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=7.5, pad=3)
            ax.text(0.03, 0.03, f'minADE {ade:.2f} m\nminFDE {fde:.2f} m', transform=ax.transAxes, fontsize=6.5,
                    va='bottom', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
        axes[0].legend(fontsize=5.8, loc='upper left', frameon=True, framealpha=0.85)
        fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.97))
        name = f'{rank:02d}_{sid}_{key[1]}'
        fig.savefig(os.path.join(a.out, name + '.pdf')); fig.savefig(os.path.join(a.out, name + '.png'), dpi=110)
        plt.close(fig); thumbs.append(os.path.join(a.out, name + '.png'))
        print(f'{name}: FDE clean {e0[1]:.2f} -> noisy {e1[1]:.2f}, mixed {em[1]:.2f}')
    print('saved to', a.out)


if __name__ == '__main__':
    main()
