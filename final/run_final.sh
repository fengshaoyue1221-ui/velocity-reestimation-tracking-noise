#!/usr/bin/env bash
# Final experiments (resumable, with automatic retries):
#   S0  wait until supp/run_supp.sh has finished (GPUs busy until then)
#   S1  sensitivity: rho in {0, 0.5, 0.95} and no distance dependence (beta=0), lambda=1,
#       clean + aug models                                                    (8 evals, ~50 min)
#   S2  train MTR without velocity input (clean training)                     (~27 h)
#   S3  evaluate it: 5 severities + 6 single error types                      (11 evals, ~70 min)
#   S4  summary -> paper/results/extra/final.json
# Usage (from kit root, env mtr active, inside tmux):  bash final/run_final.sh
set -o pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd); MTR=$KIT/MTR; TOOLS=$MTR/tools; ST=$KIT/.stages_final; mkdir -p "$ST"
NGPU=${NGPU:-3}; TOTAL_BS=${TOTAL_BS:-6}; EVAL_BS=${EVAL_BS:-12}
export TF_FORCE_GPU_ALLOW_GROWTH=true PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
LOG=$KIT/run_final.log; exec > >(tee -a "$LOG") 2>&1
echo "=== run_final $(date)"

# ---- S0: wait for the previous script ----
while pgrep -f "supp/run_supp.sh" >/dev/null; do echo "waiting for run_supp.sh to finish... $(date +%H:%M)"; sleep 300; done
python "$KIT/final/enable_novel.py" || { echo "ERROR: patch failed"; exit 1; }

P=DATA_CONFIG.PERCEPTION_NOISE
CK_CLEAN=$MTR/output/waymo/a2000_clean/clean/ckpt/checkpoint_epoch_30.pth
CK_AUG=$MTR/output/waymo/a2000_aug/aug/ckpt/checkpoint_epoch_30.pth
CK_NOVEL=$MTR/output/waymo/a2000_novel/novel/ckpt/checkpoint_epoch_30.pth
done_() { touch "$ST/$1"; echo "--- stage $1 done $(date)"; }
has_result() { ls "$1"/log_eval_*.txt 2>/dev/null | xargs -r grep -l "^mAP:" >/dev/null 2>&1; }
evaluate() {  # $1 cfg  $2 tag  $3 ckpt  $4 eval_tag, rest = --set overrides; skips done evals, retries 3x
  local cfg=$1 tag=$2 ck=$3 et=$4; shift 4
  local out="$MTR/output/waymo/$(basename $cfg .yaml)/$tag/eval/epoch_30/$et"
  if has_result "$out"; then echo "skip $tag/$et"; return 0; fi
  for attempt in 1 2 3; do
    echo "--- eval $tag / $et (attempt $attempt)"; cd "$TOOLS"
    bash scripts/dist_test.sh "$NGPU" --cfg_file $cfg --ckpt "$ck" --extra_tag $tag --eval_tag $et \
         --batch_size $EVAL_BS --set "$@"
    has_result "$out" && return 0
    echo "eval failed, retrying in 60 s"; sleep 60
  done
  echo "ERROR: eval $tag/$et failed 3 times"; exit 1
}

# ---- S1: sensitivity ----
if [ ! -f "$ST/sensitivity" ]; then
  for spec in "clean:$CK_CLEAN" "aug:$CK_AUG"; do
    tag=${spec%%:*}; ck=${spec#*:}
    for rho in 0.0 0.5 0.95; do
      evaluate cfgs/waymo/a2000_clean.yaml $tag "$ck" sens_rho${rho} $P.LEVEL 1.0 $P.POS_RHO $rho
    done
    evaluate cfgs/waymo/a2000_clean.yaml $tag "$ck" sens_beta0 $P.LEVEL 1.0 $P.POS_RANGE_COEF 0.0
  done
  done_ sensitivity
fi

# ---- S2: train the no-velocity model ----
if [ ! -f "$ST/train_novel" ]; then
  for attempt in 1 2 3; do
    cd "$TOOLS"
    bash scripts/dist_train.sh "$NGPU" --cfg_file cfgs/waymo/a2000_novel.yaml --batch_size $TOTAL_BS \
         --extra_tag novel --workers 8 --not_eval_with_train || echo "WARNING: train.py exited non-zero; checking checkpoint"
    [ -f "$CK_NOVEL" ] && break
    echo "no final checkpoint yet (attempt $attempt); resuming in 60 s"; sleep 60
  done
  [ -f "$CK_NOVEL" ] || { echo "ERROR: training did not finish. Re-run to resume."; exit 1; }
  done_ train_novel
fi

# ---- S3: evaluate the no-velocity model (its own config keeps velocities zeroed) ----
if [ ! -f "$ST/eval_novel" ]; then
  NC=cfgs/waymo/a2000_novel.yaml
  for L in 0.0 0.5 1.0 2.0 3.0; do evaluate $NC novel "$CK_NOVEL" level_$L $P.LEVEL $L; done
  ZERO="$P.POS_STD 0.0 $P.POS_RANGE_COEF 0.0 $P.HEAD_STD 0.0 $P.VEL_STD 0.0 $P.SIZE_STD 0.0 $P.DROP_PROB 0.0 $P.TRUNC_PROB 0.0 $P.IDSWAP_PROB 0.0"
  evaluate $NC novel "$CK_NOVEL" only_pos    $P.LEVEL 1.0 $ZERO $P.POS_STD 0.15 $P.POS_RANGE_COEF 0.005
  evaluate $NC novel "$CK_NOVEL" only_head   $P.LEVEL 1.0 $ZERO $P.HEAD_STD 0.05
  evaluate $NC novel "$CK_NOVEL" only_vel    $P.LEVEL 1.0 $ZERO $P.VEL_STD 0.2
  evaluate $NC novel "$CK_NOVEL" only_drop   $P.LEVEL 1.0 $ZERO $P.DROP_PROB 0.05
  evaluate $NC novel "$CK_NOVEL" only_trunc  $P.LEVEL 1.0 $ZERO $P.TRUNC_PROB 0.1
  evaluate $NC novel "$CK_NOVEL" only_idswap $P.LEVEL 1.0 $ZERO $P.IDSWAP_PROB 0.5
  done_ eval_novel
fi

# ---- S4: summary ----
python - "$MTR/output/waymo" "$KIT/paper/results/extra" <<'PY'
import glob, json, os, re, sys
root, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
def met(group, tag, et):
    for f in sorted(glob.glob(f'{root}/{group}/{tag}/eval/epoch_30/{et}/log_eval_*.txt'), reverse=True):
        d = {m: float(v) for m, v in re.findall(r'^(mAP|minADE|minFDE|MissRate)\s*:\s*([-\d.]+)\s*$', open(f).read(), re.M)}
        if 'mAP' in d: return d
    return None
rel = lambda d, b: None if (d is None or b is None) else 100 * (d['mAP'] - b['mAP']) / b['mAP']
fmt = lambda x: '   --' if x is None else f'{x:+6.1f}%'
res = {'sens': {}, 'models': {}, 'novel_types': {}}
print('\n== sensitivity at lambda=1: relative mAP change vs. clean input ==')
print(f"{'model':6s} {'rho=0':>9s} {'rho=0.5':>9s} {'rho=0.8*':>9s} {'rho=0.95':>9s} {'beta=0':>9s}   (* = default)")
for tag in ['clean', 'aug']:
    b = met('a2000_clean', tag, 'level_0.0')
    row = {k: rel(met('a2000_clean', tag, et), b) for k, et in
           [('rho0', 'sens_rho0.0'), ('rho0.5', 'sens_rho0.5'), ('rho0.8', 'level_1.0'), ('rho0.95', 'sens_rho0.95'), ('beta0', 'sens_beta0')]}
    res['sens'][tag] = row
    print(f'{tag:6s} ' + ' '.join(f'{fmt(row[k]):>9s}' for k in ['rho0', 'rho0.5', 'rho0.8', 'rho0.95', 'beta0']))
print('\n== mAP vs. severity (lambda 0 / 0.5 / 1 / 2 / 3) ==')
for name, group, tag in [('clean', 'a2000_clean', 'clean'), ('aug', 'a2000_clean', 'aug'), ('mix', 'a2000_clean', 'mix'), ('no-vel', 'a2000_novel', 'novel')]:
    row = [met(group, tag, f'level_{l}') for l in ['0.0', '0.5', '1.0', '2.0', '3.0']]; res['models'][name] = row
    print(f'{name:7s} ' + '  '.join('  --  ' if r is None else f"{r['mAP']:.4f}" for r in row)
          + ('' if row[0] is None or row[2] is None else f"   | drop at lambda=1: {100*(row[2]['mAP']-row[0]['mAP'])/row[0]['mAP']:+.1f}%"))
print('\n== no-velocity model: single error types at lambda=1 (relative mAP change) ==')
b = met('a2000_novel', 'novel', 'level_0.0')
for et in ['only_pos', 'only_head', 'only_vel', 'only_drop', 'only_trunc', 'only_idswap']:
    r = rel(met('a2000_novel', 'novel', et), b); res['novel_types'][et] = r
    print(f'{et:12s} {fmt(r)}')
json.dump(res, open(f'{out}/final.json', 'w'), indent=1); print('\nsaved', f'{out}/final.json')
PY
echo "=== run_final finished $(date)"
