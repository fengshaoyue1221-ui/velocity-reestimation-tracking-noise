#!/usr/bin/env bash
# Mechanism experiment: is the damage from position jitter caused by the position offset itself,
# or by the velocity that a tracker re-estimates from the jittered positions?
# Evaluation only (no training). 3 settings x 2 models = 6 evaluations (~40 min), then a summary,
# then zoomed qualitative figures. Run AFTER analysis/run_extra.sh has finished.
# Usage (from kit root, env mtr active):  bash mechanism/run_mechanism.sh
set -eo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd); MTR=$KIT/MTR; TOOLS=$MTR/tools
NGPU=${NGPU:-3}; EVAL_BS=${EVAL_BS:-12}; CFG=cfgs/waymo/a2000_clean.yaml
export TF_FORCE_GPU_ALLOW_GROWTH=true PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
LOG=$KIT/run_mechanism.log; exec > >(tee -a "$LOG") 2>&1
echo "=== run_mechanism $(date)"
python "$KIT/mechanism/enable_vel_switches.py"

P=DATA_CONFIG.PERCEPTION_NOISE
ZERO="$P.POS_STD 0.0 $P.POS_RANGE_COEF 0.0 $P.HEAD_STD 0.0 $P.VEL_STD 0.0 $P.SIZE_STD 0.0 $P.DROP_PROB 0.0 $P.TRUNC_PROB 0.0 $P.IDSWAP_PROB 0.0"
POS="$P.POS_STD 0.15 $P.POS_RANGE_COEF 0.005"
declare -A SET
SET[mech_pos_offset]="$P.LEVEL 1.0 $ZERO $POS $P.VEL_FROM_POS False"          # positions jittered, velocity clean
SET[mech_vel_derived]="$P.LEVEL 1.0 $ZERO $POS $P.POS_FOR_VEL_ONLY True"      # positions clean, velocity from jittered positions
SET[mech_all_cleanvel]="$P.LEVEL 1.0 $P.VEL_FROM_POS False"                  # full lambda=1 noise, but velocity not re-derived

cd "$TOOLS"
for tag in clean aug; do
  [ $tag = clean ] && CK=$MTR/output/waymo/a2000_clean/clean/ckpt/checkpoint_epoch_30.pth || CK=$MTR/output/waymo/a2000_aug/aug/ckpt/checkpoint_epoch_30.pth
  for et in mech_pos_offset mech_vel_derived mech_all_cleanvel; do
    if ls "$MTR/output/waymo/a2000_clean/$tag/eval/epoch_30/$et"/log_eval_*.txt 2>/dev/null | xargs -r grep -l "^mAP:" >/dev/null 2>&1; then echo "skip $tag/$et (done)"; continue; fi
    echo "--- $tag / $et"
    bash scripts/dist_test.sh "$NGPU" --cfg_file $CFG --ckpt "$CK" --extra_tag $tag --eval_tag $et --batch_size $EVAL_BS --set ${SET[$et]}
  done
done

# ---- summary ----
python - "$MTR/output/waymo/a2000_clean" "$KIT/paper/results/extra" <<'PY'
import glob, json, os, re, sys
root, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
def met(tag, et):
    for f in sorted(glob.glob(f'{root}/{tag}/eval/epoch_30/{et}/log_eval_*.txt'), reverse=True):
        d = {m: float(v) for m, v in re.findall(r'^(mAP|minADE|minFDE|MissRate)\s*:\s*([-\d.]+)\s*$', open(f).read(), re.M)}
        if 'mAP' in d: return d
    return None
rows = [('level_0.0', 'clean input'), ('only_pos', 'position jitter (offset + re-derived velocity)'),
        ('mech_pos_offset', '  position offset only (velocity clean)'),
        ('mech_vel_derived', '  re-derived velocity only (positions clean)'),
        ('level_1.0', 'all noise, lambda=1'), ('mech_all_cleanvel', '  all noise, velocity NOT re-derived')]
res = {}
for tag in ['clean', 'aug']:
    base = met(tag, 'level_0.0'); res[tag] = {}
    print(f'\n== {tag} model ==   (relative change vs. its own clean input)')
    for et, name in rows:
        d = met(tag, et); res[tag][et] = d
        if d is None: print(f'{name:48s}  MISSING'); continue
        print(f"{name:48s} mAP {d['mAP']:.4f} ({100*(d['mAP']-base['mAP'])/base['mAP']:+6.1f}%)   "
              f"minADE {d['minADE']:.3f} ({100*(d['minADE']-base['minADE'])/base['minADE']:+6.1f}%)   MR {d['MissRate']:.3f}")
json.dump(res, open(f'{out}/mechanism.json', 'w'), indent=1)
print('\nsaved', f'{out}/mechanism.json')
PY

# ---- zoomed qualitative figures (lambda=2, +-25 m) ----
cd "$MTR"; mkdir -p "$KIT/paper/results/qual_zoom"
for s in 10071ee58db4bd92 1008f05c233dd975 10067cf7cc2506c7; do
  python "$KIT/vis_noise.py" --pkl data/waymo/processed_scenarios_validation/sample_$s.pkl \
         --out "$KIT/paper/results/qual_zoom/${s}_l2_r25.pdf" --level 2 --radius 25 || true
done
echo "=== run_mechanism finished $(date)"
