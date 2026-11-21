"""
Constant-velocity (CV) reference under the SAME perception noise as the MTR evaluation
(same config, same per-scenario seeds, same protected agents), scored with MTR's Waymo
evaluation code. CPU only.

The CV forecast extrapolates the (possibly corrupted) current position with the
(possibly corrupted) current velocity. It is unimodal, so only minADE / minFDE /
MissRate are meaningful; mAP is reported but should not be compared.

Usage (from MTR root, env "mtr" active):
  CUDA_VISIBLE_DEVICES="" python ../analysis/cv_baseline.py --cfg tools/cfgs/waymo/a2000_clean.yaml \
      --levels 0 0.5 1 2 3 --out ../paper/results/extra/cv_baseline.json
"""
import argparse, json, os, pickle, sys, zlib
import numpy as np
import yaml

sys.path.insert(0, os.getcwd())
from mtr.datasets.waymo.perception_noise import PerceptionNoise

TYPES = ('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST')


def load_noise_cfg(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    return dict(cfg['DATA_CONFIG']['PERCEPTION_NOISE']), cfg['DATA_CONFIG']


def cv_preds_for_scene(info, noise_cfg, level, n_modes=6):
    cur = info['current_time_index']
    sdc = info['sdc_track_index']
    tr = info['track_infos']
    trajs = tr['trajs']
    types = np.array(tr['object_type'])
    ids = np.array(tr['object_id'])
    tti = np.array(info['tracks_to_predict']['track_index'])
    past = trajs[:, :cur + 1]
    if level > 0:
        cfg = dict(noise_cfg); cfg['LEVEL'] = float(level)
        seed = zlib.crc32(str(info['scenario_id']).encode()) + int(cfg.get('SEED', 0))
        protect = np.zeros(len(types), bool); protect[tti] = True
        past, _ = PerceptionNoise(cfg, seed=seed)(past, types, sdc, protect_mask=protect)
    out = []
    t = (np.arange(1, trajs.shape[1] - cur) * 0.1)[:, None]          # 0.1 .. 8.0 s
    for k in tti:
        if types[k] not in TYPES:
            continue
        p0, v0 = past[k, -1, 0:2], past[k, -1, 7:9]
        traj = p0[None] + t * v0[None]                                  # (80, 2)
        scores = np.zeros(n_modes, np.float32); scores[0] = 1.0
        out.append({'scenario_id': info['scenario_id'], 'object_id': int(ids[k]),
                    'object_type': str(types[k]),
                    'pred_trajs': np.repeat(traj[None], n_modes, 0).astype(np.float32),
                    'pred_scores': scores, 'gt_trajs': trajs[k].astype(np.float32),
                    'track_index_to_predict': int(k)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--levels', nargs='+', type=float, default=[0, 0.5, 1, 2, 3])
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_scenes', type=int, default=-1)
    ap.add_argument('--no_eval', action='store_true', help='testing only: skip Waymo metrics')
    a = ap.parse_args()
    noise_cfg, dcfg = load_noise_cfg(a.cfg)
    root = dcfg['DATA_ROOT']
    infos = pickle.load(open(os.path.join(root, dcfg['INFO_FILE']['test']), 'rb'))
    data_dir = os.path.join(root, dcfg['SPLIT_DIR']['test'])
    if a.max_scenes > 0:
        infos = infos[:a.max_scenes]
    if not a.no_eval:
        from mtr.datasets.waymo.waymo_eval import waymo_evaluation
    results = {}
    for lv in a.levels:
        preds = []
        for inf in infos:
            info = pickle.load(open(os.path.join(data_dir, f"sample_{inf['scenario_id']}.pkl"), 'rb'))
            preds.append(cv_preds_for_scene(info, noise_cfg, lv))
        preds = [p for p in preds if p]
        n = sum(len(p) for p in preds)
        if a.no_eval:
            results[str(lv)] = {'num_agents': n}
        else:
            res, fmt = waymo_evaluation(pred_dicts=preds, num_modes_for_eval=6)
            results[str(lv)] = {k: float(v) for k, v in res.items()
                                if isinstance(v, (int, float)) and not k.startswith('-')}
            print(f'level {lv}:\n{fmt}')
        print(f'level {lv}: {n} agents done')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(results, open(a.out, 'w'), indent=1)
    print('saved', a.out)


if __name__ == '__main__':
    main()
