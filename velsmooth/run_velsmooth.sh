#!/usr/bin/env bash
# Training-free mitigation: least-squares velocity re-estimation at test time (VEL_SMOOTH_K).
# 7 evaluations (~45 min), no training. Run AFTER mechanism/run_mechanism.sh.
# Usage (from kit root, env mtr active):  bash velsmooth/run_velsmooth.sh
set -eo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd); MTR=$KIT/MTR; TOOLS=$MTR/tools
NGPU=${NGPU:-3}; EVAL_BS=${EVAL_BS:-12}; CFG=cfgs/waymo/a2000_clean.yaml
export TF_FORCE_GPU_ALLOW_GROWTH=true PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
LOG=$KIT/run_velsmooth.log; exec > >(tee -a "$LOG") 2>&1
echo "=== run_velsmooth $(date)"
python "$KIT/velsmooth/enable_vel_smoothing.py"
P=DATA_CONFIG.PERCEPTION_NOISE
RUNS="clean:5:0.0 clean:11:0.0 clean:5:1.0 clean:11:1.0 clean:11:2.0 aug:11:0.0 aug:11:1.0"
cd "$TOOLS"
for r in $RUNS; do
  IFS=: read tag k lv <<< "$r"; et=vs_k${k}_level_${lv}
  [ $tag = clean ] && CK=$MTR/output/waymo/a2000_clean/clean/ckpt/checkpoint_epoch_30.pth || CK=$MTR/output/waymo/a2000_aug/aug/ckpt/checkpoint_epoch_30.pth
  if ls "$MTR/output/waymo/a2000_clean/$tag/eval/epoch_30/$et"/log_eval_*.txt 2>/dev/null | xargs -r grep -l "^mAP:" >/dev/null 2>&1; then echo "skip $tag/$et (done)"; continue; fi
  echo "--- $tag / $et"
  bash scripts/dist_test.sh "$NGPU" --cfg_file $CFG --ckpt "$CK" --extra_tag $tag --eval_tag $et \
       --batch_size $EVAL_BS --set $P.LEVEL $lv $P.VEL_SMOOTH_K $k
done
python - "$MTR/output/waymo/a2000_clean" "$KIT/paper/results/extra" <<'PY'
import glob, json, os, re, sys
root, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
def met(tag, et):
    for f in sorted(glob.glob(f'{root}/{tag}/eval/epoch_30/{et}/log_eval_*.txt'), reverse=True):
        d = {m: float(v) for m, v in re.findall(r'^(mAP|minADE|minFDE|MissRate)\s*:\s*([-\d.]+)\s*$', open(f).read(), re.M)}
        if 'mAP' in d: return d
    return None
res = {}
for tag, ks, lvs in [('clean', ['raw', '5', '11'], ['0.0', '1.0', '2.0']), ('aug', ['raw', '11'], ['0.0', '1.0'])]:
    base = met(tag, 'level_0.0'); res[tag] = {}
    print(f'\n== {tag} model ==  (mAP, change vs. raw velocity on clean input)')
    print(f"{'velocity':10s}" + ''.join(f'{"lambda="+lv:>26s}' for lv in lvs))
    for k in ks:
        cells = []
        for lv in lvs:
            d = met(tag, f'level_{lv}' if k == 'raw' else f'vs_k{k}_level_{lv}'); res[tag][f'{k}|{lv}'] = d
            cells.append('--' if d is None else f"{d['mAP']:.4f} ({100*(d['mAP']-base['mAP'])/base['mAP']:+5.1f}%) ADE {d['minADE']:.2f}")
        print(f"{('LS K='+k) if k!='raw' else 'raw':10s}" + ''.join(f'{c:>26s}' for c in cells))
json.dump(res, open(f'{out}/velsmooth.json', 'w'), indent=1); print('\nsaved', f'{out}/velsmooth.json')
PY
echo "=== run_velsmooth finished $(date)"
