"""
Measure the errors that the noise model actually injects, with the evaluation seeds
(seed = crc32(scenario_id) + SEED), and compare with the closed-form values of Table 2.

Usage (from MTR/, env mtr):
    python ../missing_tools/injected_error_stats.py --levels 0.25 0.5 1.0 --max_scenes 3000
Output: missing_outputs/injected_error_stats.json
"""
import argparse, glob, json, math, os, pickle, sys, zlib
import numpy as np
import yaml

import importlib.util
_spec = importlib.util.spec_from_file_location('perception_noise', os.path.join(os.getcwd(), 'mtr/datasets/waymo/perception_noise.py'))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
PerceptionNoise = _mod.PerceptionNoise   # loaded directly, so torch is not needed for this script

CUR = 10
CLASSES = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
DIST_BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 1e9)]


def theory(lam, d, kappa=1.0, s0=0.15, beta=0.005, rho=0.8, dt=0.1, sw=0.2):
    sig = lam * kappa * (s0 + beta * d); a = min(1.0, lam)
    sv = math.sqrt((a * sig * math.sqrt(2 * (1 - rho)) / dt) ** 2 + (lam * sw) ** 2)
    k = math.sqrt(math.pi / 2)
    return {'E_pos_err': sig * k, 'E_vel_err': sv * k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    ap.add_argument('--max_scenes', type=int, default=3000)
    ap.add_argument('--cfg', default='tools/cfgs/waymo/a2000_clean.yaml')
    ap.add_argument('--out', default='../missing_outputs')
    a = ap.parse_args()
    base = dict(yaml.safe_load(open(a.cfg))['DATA_CONFIG']['PERCEPTION_NOISE'])
    files = sorted(glob.glob('data/waymo/processed_scenarios_validation/sample_*.pkl'))[:a.max_scenes]
    res = {'scenes': len(files), 'note': 'errors over valid past frames t=-9..0 of non-ego agents; vehicle theory at 20 m'}
    for lam in a.levels:
        cfg = dict(base, LEVEL=lam)
        acc = {}
        for f in files:
            info = pickle.load(open(f, 'rb')); tr = info['track_infos']
            x = tr['trajs'][:, :CUR + 1]; types = np.array(tr['object_type'])
            sdc = info['sdc_track_index']
            protect = np.zeros(len(types), bool); protect[info['tracks_to_predict']['track_index']] = True
            seed = zlib.crc32(str(info['scenario_id']).encode()) + int(cfg.get('SEED', 0))
            y, _ = PerceptionNoise(cfg, seed=seed)(x, types, sdc, protect)
            ok = (x[:, 1:, 9] > 0) & (x[:, :-1, 9] > 0)
            ep = np.linalg.norm(y[:, 1:, 0:2] - x[:, 1:, 0:2], axis=-1)
            ev = np.linalg.norm(y[:, 1:, 7:9] - x[:, 1:, 7:9], axis=-1)
            dist = np.linalg.norm(x[:, 1:, 0:2] - x[sdc, 1:, 0:2][None], axis=-1)
            for i in range(len(types)):
                if i == sdc or types[i] not in CLASSES:
                    continue
                for lo, hi in DIST_BINS:
                    m = ok[i] & (dist[i] >= lo) & (dist[i] < hi)
                    if m.any():
                        k = (types[i], f'{lo}-{hi if hi < 1e9 else "inf"} m')
                        acc.setdefault(k, [[], []]); acc[k][0].append(ep[i][m]); acc[k][1].append(ev[i][m])
        lv = {}
        for (c, b), (p, v) in sorted(acc.items()):
            p = np.concatenate(p); v = np.concatenate(v)
            lv.setdefault(c, {})[b] = {'count': int(len(p)), 'mean_pos_err_m': float(p.mean()), 'mean_vel_err_mps': float(v.mean())}
        lv['theory_vehicle_20m'] = theory(lam, 20.0)
        res[f'lambda_{lam}'] = lv
        veh = lv.get('TYPE_VEHICLE', {}).get('20-30 m')
        if veh:
            print(f"lambda={lam}: vehicles 20-30 m measured E|e_p|={veh['mean_pos_err_m']:.3f} m, E|e_v|={veh['mean_vel_err_mps']:.3f} m/s;"
                  f" theory at 20 m: {lv['theory_vehicle_20m']['E_pos_err']:.3f} m, {lv['theory_vehicle_20m']['E_vel_err']:.3f} m/s")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'injected_error_stats.json'); json.dump(res, open(path, 'w'), indent=1); print('saved', path)


if __name__ == '__main__':
    main()
