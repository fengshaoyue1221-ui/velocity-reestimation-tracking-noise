#!/usr/bin/env bash
# Reviewer-requested controls (evaluation only, no training). Resumable: finished runs are skipped.
#   check   : both switches on -> must reproduce the clean-input mAP exactly (correctness check)
#   item 1  : clean positions, velocity = finite difference of CLEAN positions
#   item 5  : velocity-feature vs acceleration-feature pathways (lambda = 0.5, 1)
#   item 2  : paired interventions at sensor-level severities (lambda = 0.35, 0.5, 0.75)
# Models: clean-trained and mixed-trained. Run inside tmux from the repository root:
#   bash reviewer_ctrl/run_reviewer_ctrl.sh
set -u
KIT=$(cd "$(dirname "$0")/.." && pwd); MTR=$KIT/MTR
cd "$MTR/tools"
export TF_FORCE_GPU_ALLOW_GROWTH=true PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NGPU=${NGPU:-3}; BS=${EVAL_BS:-12}; CFG=cfgs/waymo/a2000_clean.yaml
declare -A CKPT=( [clean]=$MTR/output/waymo/a2000_clean/clean/ckpt/checkpoint_epoch_30.pth
                  [mix]=$MTR/output/waymo/a2000_mix/mix/ckpt/checkpoint_epoch_30.pth )
P=DATA_CONFIG.PERCEPTION_NOISE
ZERO="$P.POS_STD 0.0 $P.POS_RANGE_COEF 0.0 $P.HEAD_STD 0.0 $P.VEL_STD 0.0 $P.SIZE_STD 0.0 $P.DROP_PROB 0.0 $P.TRUNC_PROB 0.0 $P.IDSWAP_PROB 0.0"
POS="$P.POS_STD 0.15 $P.POS_RANGE_COEF 0.005"

JOBS=()                                   # "tag|overrides"
JOBS+=("rc_check_1.0|$P.LEVEL 1.0 $ZERO $POS $P.POS_FOR_VEL_ONLY True $P.ACC_FROM_CLEAN True $P.VEL_FEAT_CLEAN True")
JOBS+=("rc_fd_clean|$P.LEVEL 1.0 $ZERO $P.POS_FOR_VEL_ONLY True")
for L in 1.0 0.5; do
  JOBS+=("rc_velfeat_only_$L|$P.LEVEL $L $ZERO $POS $P.POS_FOR_VEL_ONLY True $P.ACC_FROM_CLEAN True")
  JOBS+=("rc_acc_only_$L|$P.LEVEL $L $ZERO $POS $P.POS_FOR_VEL_ONLY True $P.VEL_FEAT_CLEAN True")
done
for L in 0.5 0.35 0.75; do
  JOBS+=("rc_pos_offset_$L|$P.LEVEL $L $ZERO $POS $P.VEL_FROM_POS False")
  JOBS+=("rc_vel_derived_$L|$P.LEVEL $L $ZERO $POS $P.POS_FOR_VEL_ONLY True")
  JOBS+=("rc_both_$L|$P.LEVEL $L $ZERO $POS")
  JOBS+=("rc_all_$L|$P.LEVEL $L")
  JOBS+=("rc_all_cleanvel_$L|$P.LEVEL $L $P.VEL_FROM_POS False")
done

done_run() { grep -qs "^mAP:" "$MTR/output/waymo/a2000_clean/$1/eval/epoch_30/$2"/log_eval_*.txt; }
n=0; total=$(( ${#JOBS[@]} * 2 ))
for job in "${JOBS[@]}"; do
  tag=${job%%|*}; ovr=${job#*|}
  for model in clean mix; do
    n=$((n+1))
    if done_run $model $tag; then echo "[$n/$total] skip $model $tag"; continue; fi
    echo "[$n/$total] === $model $tag  $(date)"
    for attempt in 1 2 3; do
      bash scripts/dist_test.sh $NGPU --cfg_file $CFG --ckpt "${CKPT[$model]}" --extra_tag $model \
           --eval_tag $tag --batch_size $BS --set $ovr && break
      echo "retry $attempt"; sleep 30
    done
  done
done
echo "=== reviewer controls finished $(date)"
