"""
Recompute the WOMD metrics at full float precision from every saved result.pkl and check that all
evaluations use the same sample set (scenario_id, object_id, object_type).

Usage (from MTR/, env mtr; CPU only, resumable):
    python ../missing_tools/recompute_metrics.py
Outputs (in missing_outputs/):
    metrics_full_precision.json   all metric keys (averages, per class, per horizon) for every evaluation
    metrics_full_precision.csv    main metrics, one row per evaluation
    sample_set_check.json         number of scenes/agents and a hash of the sample set per evaluation
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''          # TensorFlow metrics on CPU
import argparse, csv, glob, hashlib, json, pickle, sys, time

sys.path.insert(0, os.getcwd())


def sample_signature(pred_dicts):
    keys = sorted((str(d['scenario_id']), int(d['object_id']), str(d['object_type']))
                  for scene in pred_dicts for d in (scene if isinstance(scene, list) else [scene]))
    per_type = {}
    for _, _, t in keys:
        per_type[t] = per_type.get(t, 0) + 1
    h = hashlib.md5('\n'.join(f'{a},{b},{c}' for a, b, c in keys).encode()).hexdigest()
    return {'scenes': len({k[0] for k in keys}), 'agents': len(keys), 'per_type': per_type, 'md5': h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='output/waymo')
    ap.add_argument('--out', default='../missing_outputs')
    a = ap.parse_args()
    from mtr.datasets.waymo.waymo_eval import waymo_evaluation
    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, 'metrics_full_precision.json'); spath = os.path.join(a.out, 'sample_set_check.json')
    done = json.load(open(jpath)) if os.path.exists(jpath) else {}
    sig = json.load(open(spath)) if os.path.exists(spath) else {}
    files = sorted(glob.glob(os.path.join(a.root, '*', '*', 'eval', 'epoch_*', '*', 'result.pkl')))
    print(f'{len(files)} result.pkl files, {len(done)} already done')
    for f in files:
        parts = f.split(os.sep); key = '/'.join(parts[-6:-1])          # cfg/model/eval/epoch/setting
        if key in done:
            continue
        t0 = time.time()
        pred_dicts = pickle.load(open(f, 'rb'))
        res, _ = waymo_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=6)
        done[key] = {k.strip(): float(v) for k, v in res.items() if not k.startswith('---')}
        sig[key] = sample_signature(pred_dicts)
        json.dump(done, open(jpath, 'w'), indent=1); json.dump(sig, open(spath, 'w'), indent=1)
        print(f'{key}: mAP={done[key]["mAP"]:.6f}  agents={sig[key]["agents"]}  ({time.time() - t0:.0f}s)')
    cols = ['mAP', 'minADE', 'minFDE', 'MissRate'] + [f'{m} - {c}' for m in ['mAP', 'minADE'] for c in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']]
    with open(os.path.join(a.out, 'metrics_full_precision.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['evaluation'] + cols)
        for k in sorted(done):
            w.writerow([k] + [repr(done[k].get(c, '')) for c in cols])
    hashes = {}
    for k, s in sig.items():
        hashes.setdefault(s['md5'], []).append(k)
    print(f'\nsample sets: {len(hashes)} distinct across {len(sig)} evaluations')
    for h, ks in hashes.items():
        s = sig[ks[0]]; print(f'  {h[:12]}  {len(ks):3d} evaluations  scenes={s["scenes"]} agents={s["agents"]} {s["per_type"]}')


if __name__ == '__main__':
    main()
