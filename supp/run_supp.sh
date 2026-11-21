#!/usr/bin/env bash
# Supplementary experiments (resumable; finished stages are skipped on re-run):
#   S1  noise-seed variability: lambda=1, seeds 1 and 2, clean + aug models   (4 evals, ~25 min)
#   S2  train MTR (mix): 50% clean / 50% lambda=1 training scenes              (~27 h)
#   S3  evaluate mix: full 11-eval sweep + 3 mechanism evals                    (~1.5 h)
#   S4  summary tables  -> paper/results/extra/supp.json
# Usage (from kit root, env mtr active, inside tmux):  bash supp/run_supp.sh
set -eo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd); MTR=$KIT/MTR; TOOLS=$MTR/tools; ST=$KIT/.stages_supp; mkdir -p "$ST"
NGPU=${NGPU:-3}; TOTAL_BS=${TOTAL_BS:-6}; EVAL_BS=${EVAL_BS:-12}; CFG=cfgs/waymo/a2000_clean.yaml
export TF_FORCE_GPU_ALLOW_GROWTH=true PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
LOG=$KIT/run_supp.log; exec > >(tee -a "$LOG") 2>&1
echo "=== run_supp $(date)"
python "$KIT/supp/enable_mix_training.py"
P=DATA_CONFIG.PERCEPTION_NOISE
CK_CLEAN=$MTR/output/waymo/a2000_clean/clean/ckpt/checkpoint_epoch_30.pth
CK_AUG=$MTR/output/waymo/a2000_aug/aug/ckpt/checkpoint_epoch_30.pth
CK_MIX=$MTR/output/waymo/a2000_mix/mix/ckpt/checkpoint_epoch_30.pth
done_() { touch "$ST/$1"; echo "--- stage $1 done $(date)"; }
evaluate() {  # $1 tag $2 ckpt $3 eval_tag, rest = --set overrides
  local tag=$1 ck=$2 et=$3; shift 3
  if ls "$MTR/output/waymo/a2000_clean/$tag/eval/epoch_30/$et"/log_eval_*.txt 2>/dev/null | xargs -r grep -l "^mAP:" >/dev/null 2>&1; then echo "skip $tag/$et"; return; fi
  echo "--- eval $tag / $et"; cd "$TOOLS"
  bash scripts/dist_test.sh "$NGPU" --cfg_file $CFG --ckpt "$ck" --extra_tag $tag --eval_tag $et --batch_size $EVAL_BS --set "$@"
}

# ---- S1: noise-seed variability ----
if [ ! -f "$ST/seeds" ]; then
  for s in 1 2; do
    evaluate clean "$CK_CLEAN" seed${s}_level_1.0 $P.LEVEL 1.0 $P.SEED $s
    evaluate aug   "$CK_AUG"   seed${s}_level_1.0 $P.LEVEL 1.0 $P.SEED $s
  done
  done_ seeds
fi

# ---- S2: train the mixed model ----
if [ ! -f "$ST/train_mix" ]; then
  cd "$TOOLS"
  bash scripts/dist_train.sh "$NGPU" --cfg_file cfgs/waymo/a2000_mix.yaml --batch_size $TOTAL_BS --extra_tag mix \
       --workers 8 --not_eval_with_train || echo "WARNING: train.py exited non-zero (often only its post-training eval); checking checkpoint"
  [ -f "$CK_MIX" ] || { echo "ERROR: no final checkpoint $CK_MIX - training did not finish. Re-run to resume."; exit 1; }
  done_ train_mix
fi

# ---- S3: evaluate the mixed model ----
if [ ! -f "$ST/eval_mix" ]; then
  cd "$TOOLS"; bash run_noise_sweep.sh "$CK_MIX" mix "$NGPU" "$CFG" "$EVAL_BS"
  ZERO="$P.POS_STD 0.0 $P.POS_RANGE_COEF 0.0 $P.HEAD_STD 0.0 $P.VEL_STD 0.0 $P.SIZE_STD 0.0 $P.DROP_PROB 0.0 $P.TRUNC_PROB 0.0 $P.IDSWAP_PROB 0.0"
  POS="$P.POS_STD 0.15 $P.POS_RANGE_COEF 0.005"
  evaluate mix "$CK_MIX" mech_pos_offset   $P.LEVEL 1.0 $ZERO $POS $P.VEL_FROM_POS False
  evaluate mix "$CK_MIX" mech_vel_derived  $P.LEVEL 1.0 $ZERO $POS $P.POS_FOR_VEL_ONLY True
  evaluate mix "$CK_MIX" mech_all_cleanvel $P.LEVEL 1.0 $P.VEL_FROM_POS False
  done_ eval_mix
fi

# ---- S4: summary ----
python - "$MTR/output/waymo/a2000_clean" "$KIT/paper/results/extra" <<'PY'
import glob, json, os, re, statistics as st, sys
root, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
def met(tag, et):
    for f in sorted(glob.glob(f'{root}/{tag}/eval/epoch_30/{et}/log_eval_*.txt'), reverse=True):
        d = {m: float(v) for m, v in re.findall(r'^(mAP|minADE|minFDE|MissRate)\s*:\s*([-\d.]+)\s*$', open(f).read(), re.M)}
        if 'mAP' in d: return d
    return None
res = {'seeds': {}, 'sweep': {}, 'mech': {}}
print('\n== noise-seed variability at lambda=1 (seeds 0,1,2) ==')
for tag in ['clean', 'aug']:
    runs = [met(tag, 'level_1.0'), met(tag, 'seed1_level_1.0'), met(tag, 'seed2_level_1.0')]
    runs = [r for r in runs if r]; res['seeds'][tag] = runs
    for m in ['mAP', 'minADE']:
        v = [r[m] for r in runs]
        print(f'{tag:6s} {m:7s} ' + ' '.join(f'{x:.4f}' for x in v) + (f'   mean {st.mean(v):.4f}  std {st.stdev(v):.4f}' if len(v) > 1 else ''))
print('\n== mAP vs. severity ==   (rows: model, cols: lambda 0 / 0.5 / 1 / 2 / 3)')
for tag in ['clean', 'aug', 'mix']:
    row = [met(tag, f'level_{l}') for l in ['0.0', '0.5', '1.0', '2.0', '3.0']]; res['sweep'][tag] = row
    print(f'{tag:6s} ' + '  '.join('  --  ' if r is None else f"{r['mAP']:.4f}" for r in row)
          + ('' if row[0] is None else '   | minADE ' + ' '.join('--' if r is None else f"{r['minADE']:.3f}" for r in row)))
print('\n== mix model: decomposition (relative mAP change vs. its clean input) ==')
b = met('mix', 'level_0.0')
for et in ['mech_pos_offset', 'mech_vel_derived', 'only_pos', 'level_1.0', 'mech_all_cleanvel']:
    d = met('mix', et); res['mech'][et] = d
    print(f'{et:20s} ' + ('--' if (d is None or b is None) else f"mAP {d['mAP']:.4f} ({100*(d['mAP']-b['mAP'])/b['mAP']:+.1f}%)"))
json.dump(res, open(f'{out}/supp.json', 'w'), indent=1); print('\nsaved', f'{out}/supp.json')
PY
echo "=== run_supp finished $(date)"
