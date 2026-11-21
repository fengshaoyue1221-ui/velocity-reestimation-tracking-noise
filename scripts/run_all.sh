#!/usr/bin/env bash
# End-to-end, RESUMABLE pipeline. Each finished stage writes a marker in .stages/;
# re-running the same command skips finished stages and resumes training from the
# latest checkpoint (MTR auto-resumes). Run inside tmux.
#
# Usage (from the kit root, env activated):
#   PROFILE=a2000 NGPU=3 DATA_DIR=/path/to/womd_scenario bash scripts/run_all.sh [stage]
#   stage = all (default) | preprocess | check | train_clean | train_aug | sweep | collect
#
# Env vars:
#   PROFILE   a2000 (default; 3x RTX A2000 12GB) | a100     -> picks cfgs/waymo/<PROFILE>_{clean,aug}.yaml
#   NGPU      number of GPUs (default: all visible)
#   TOTAL_BS  total training batch (default: 12 for a2000, 80 for a100 = official)
#   EVAL_BS   total evaluation batch (default: 2 x TOTAL_BS)
#   DATA_DIR  folder with training/ and validation/ shards  WORKERS   dataloader workers per GPU (default 8)
set -euo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd)
MTR=$KIT/MTR; TOOLS=$MTR/tools; ST=$KIT/.stages; mkdir -p "$ST"
NGPU=${NGPU:-$(python -c "import torch;print(torch.cuda.device_count())")}
PROFILE=${PROFILE:-a2000}
if [ "$PROFILE" = a100 ]; then DEF_BS=80; DEF_EBS=80; else DEF_BS=6; DEF_EBS=12; fi
TOTAL_BS=${TOTAL_BS:-$DEF_BS}; EVAL_BS=${EVAL_BS:-$DEF_EBS}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}; WORKERS=${WORKERS:-8}
STAGE=${1:-all}
CLEAN_CFG=cfgs/waymo/${PROFILE}_clean.yaml; AUG_CFG=cfgs/waymo/${PROFILE}_aug.yaml
LOG=$KIT/run_all.log; exec > >(tee -a "$LOG") 2>&1
# TensorFlow (used by preprocessing and the Waymo metrics) must not grab whole GPUs.
export TF_FORCE_GPU_ALLOW_GROWTH=true
echo "=== run_all $STAGE  PROFILE=$PROFILE NGPU=$NGPU TOTAL_BS=$TOTAL_BS EVAL_BS=$EVAL_BS  $(date)"
[ $((TOTAL_BS % NGPU)) -eq 0 ] && [ $((EVAL_BS % NGPU)) -eq 0 ] || { echo "TOTAL_BS and EVAL_BS must be divisible by NGPU"; exit 1; }

done_() { touch "$ST/$1"; echo "--- stage $1 done $(date)"; }
is_done() { [ -f "$ST/$1" ]; }
want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

train() {  # $1 cfg  $2 tag
  cd "$TOOLS"
  if [ "$NGPU" -gt 1 ]; then
    bash scripts/dist_train.sh "$NGPU" --cfg_file "$1" --batch_size "$TOTAL_BS" --extra_tag "$2" --workers "$WORKERS" --not_eval_with_train
  else
    python train.py --cfg_file "$1" --batch_size "$TOTAL_BS" --extra_tag "$2" --workers "$WORKERS" --not_eval_with_train
  fi
}
ckpt() { echo "$MTR/output/waymo/$(basename "$1" .yaml)/$2/ckpt/checkpoint_epoch_30.pth"; }

# ---------------- 1. preprocess ----------------
if want preprocess && ! is_done preprocess; then
  : "${DATA_DIR:?set DATA_DIR to the folder containing training/ and validation/}"
  mkdir -p "$MTR/data/waymo"; ln -sfn "$(realpath "$DATA_DIR")" "$MTR/data/waymo/scenario"
  cd "$MTR/mtr/datasets/waymo"
  # preprocessing is CPU-only; hide GPUs so 16 TensorFlow workers do not each try to grab all GPU memory
  CUDA_VISIBLE_DEVICES="" python data_preprocess.py ../../../data/waymo/scenario/ ../../../data/waymo
  python - <<EOF
import pickle
for s in ['training','val']:
    f = '$MTR/data/waymo/processed_scenarios_%s_infos.pkl' % s
    print(s, 'scenarios:', len(pickle.load(open(f,'rb'))))
EOF
  done_ preprocess
fi

# ---------------- 2. equivalence check (noise hook must be a no-op at LEVEL 0) ----------------
if want check && ! is_done check; then
  cd "$TOOLS"
  # tiny 1-epoch model on a sliver of data, then evaluate twice
  sed -e "s/'train': 1,/'train': 500,/" -e "s/'test': 1/'test': 100/" -e "s/NUM_EPOCHS: 30/NUM_EPOCHS: 1/" "$CLEAN_CFG" > cfgs/waymo/_check.yaml
  python train.py --cfg_file cfgs/waymo/_check.yaml --batch_size 4 --extra_tag check --workers 4
  CK=$(ls "$MTR"/output/waymo/_check/check/ckpt/*.pth | tail -1)
  python test.py --cfg_file cfgs/waymo/_check.yaml --ckpt "$CK" --extra_tag check --eval_tag off  --batch_size 4 --set DATA_CONFIG.PERCEPTION_NOISE.APPLY_TEST False
  python test.py --cfg_file cfgs/waymo/_check.yaml --ckpt "$CK" --extra_tag check --eval_tag lvl0 --batch_size 4 --set DATA_CONFIG.PERCEPTION_NOISE.LEVEL 0.0
  python - <<EOF
import glob, re
def m(tag):
    f = sorted(glob.glob('$MTR/output/waymo/_check/check/eval/*/%s/log_eval_*.txt' % tag))[-1]
    return dict(re.findall(r'\b(mAP|minADE|minFDE|MissRate)\s*:\s*([-\d.]+)\s*$', open(f).read(), re.M))
a, b = m('off'), m('lvl0'); print('noise off :', a); print('level 0.0 :', b)
assert a and a == b, 'EQUIVALENCE CHECK FAILED - stop and investigate'
print('EQUIVALENCE CHECK PASSED')
EOF
  rm -rf "$MTR/output/waymo/_check" cfgs/waymo/_check.yaml
  done_ check
fi

# ---------------- 3/4. training ----------------
# The evaluation that MTR runs right after training can fail sporadically in multi-GPU mode.
# Training itself is judged by the final checkpoint: if it exists, the stage is done.
train_stage() {  # $1 stage name, $2 cfg, $3 tag
  if want "$1" && ! is_done "$1"; then
    train "$2" "$3" || echo "WARNING: train.py exited non-zero (often only its post-training eval); checking checkpoint"
    if [ -f "$(ckpt "$2" "$3")" ]; then done_ "$1"; else echo "ERROR: no final checkpoint $(ckpt "$2" "$3") - training did not finish"; exit 1; fi
  fi
}
train_stage train_clean "$CLEAN_CFG" clean
train_stage train_aug   "$AUG_CFG"   aug

# ---------------- 5. evaluation sweep (both models evaluated with the clean cfg) ----------------
if want sweep && ! is_done sweep; then
  cd "$TOOLS"
  bash run_noise_sweep.sh "$(ckpt $CLEAN_CFG clean)" clean "$NGPU" "$CLEAN_CFG" "$EVAL_BS"
  bash run_noise_sweep.sh "$(ckpt $AUG_CFG aug)"     aug   "$NGPU" "$CLEAN_CFG" "$EVAL_BS"
  done_ sweep
fi

# ---------------- 6. tables / figures for the paper ----------------
if want collect; then
  cd "$MTR"
  python "$KIT/collect_results.py" --root output/waymo/${PROFILE}_clean --tags clean aug \
         --names "MTR (clean-trained)" "MTR (noise-aug)" --out "$KIT/paper/results"
  mkdir -p "$KIT/paper/results/qual_candidates"
  for f in $(ls data/waymo/processed_scenarios_validation/*.pkl | head -5); do
    id=$(basename "$f" .pkl)
    python "$KIT/vis_noise.py" --pkl "$f" --out "$KIT/paper/results/qual_candidates/${id}.pdf" --level 1.0 || true
  done
  echo "Pick the clearest qualitative figure and copy it to paper/results/fig_qual.pdf"
  done_ collect
fi
echo "=== run_all finished $STAGE $(date)"
