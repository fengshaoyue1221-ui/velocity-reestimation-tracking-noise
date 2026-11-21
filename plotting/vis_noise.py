"""
Qualitative figure: clean vs. corrupted history for one WOMD scenario.
Usage (from MTR root, after preprocessing):
  python vis_noise.py --pkl data/waymo/processed_scenarios_validation/sample_<id>.pkl --out paper/results/fig_qual.pdf
Tip: try several scenarios / --level 2 and pick one where the corruption is clearly visible.
"""
import argparse, pickle, sys, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
try:
    from mtr.datasets.waymo.perception_noise import PerceptionNoise
except ImportError:
    from perception_noise import PerceptionNoise

ap = argparse.ArgumentParser()
ap.add_argument('--pkl', required=True); ap.add_argument('--out', default='fig_qual.pdf')
ap.add_argument('--level', type=float, default=1.0); ap.add_argument('--radius', type=float, default=50)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
info = pickle.load(open(a.pkl, 'rb'))
cur = info['current_time_index']; sdc = info['sdc_track_index']
tr = info['track_infos']; trajs = tr['trajs']; types = np.array(tr['object_type'])
past = trajs[:, :cur + 1]
protect = np.zeros(len(types), bool); protect[info['tracks_to_predict']['track_index']] = True
noisy, nfo = PerceptionNoise({'LEVEL': a.level, 'IDSWAP_PROB': 0.5}, seed=a.seed)(past, types, sdc, protect)
c = past[sdc, -1, :2]
fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharex=True, sharey=True)
poly = info['map_infos']['all_polylines']
col = {'TYPE_VEHICLE': 'tab:blue', 'TYPE_PEDESTRIAN': 'tab:red', 'TYPE_CYCLIST': 'tab:green'}
for ax, H, title in zip(axes, [past, noisy], ['Clean history (auto-labels)', f'Corrupted history ($\\lambda$={a.level})']):
    if len(poly):
        m = np.linalg.norm(poly[:, :2] - c, axis=1) < a.radius * 1.5
        ax.scatter(poly[m, 0] - c[0], poly[m, 1] - c[1], s=0.1, c='0.75', rasterized=True)
    for i in range(len(types)):
        v = H[i, :, 9] > 0
        if not v.any() or np.linalg.norm(H[i, v][-1, :2] - c) > a.radius: continue
        xy = H[i, v, :2] - c
        ax.plot(xy[:, 0], xy[:, 1], '-', lw=0.8, c='k' if i == sdc else col.get(types[i], 'gray'))
        ax.plot(xy[-1, 0], xy[-1, 1], 'o', ms=2.5 if protect[i] else 1.5, c='k' if i == sdc else col.get(types[i], 'gray'))
    ax.set_title(title, fontsize=8); ax.set_aspect('equal'); ax.tick_params(labelsize=6)
    ax.set_xlim(-a.radius, a.radius); ax.set_ylim(-a.radius, a.radius)
if nfo.get('idswap'): print('ID switch between tracks', nfo['idswap'][:2], 'before step', nfo['idswap'][2])
fig.tight_layout(); fig.savefig(a.out, dpi=300); print('saved', a.out)
