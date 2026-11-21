#!/usr/bin/env bash
# Kalman-filter control (evaluation only). Same jittered positions as the paper's position-jitter
# condition; all other error types off. Resumable. Run inside tmux from the repository root:
#   bash kf_ctrl/run_kf_ctrl.sh
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
FULL="$P.FORCE_FULL_REEST True"

JOBS=()
JOBS+=("kc_check_fd_1.0|$P.LEVEL 1.0 $ZERO $POS $FULL")                         # must equal only_pos (sanity)
for L in 1.0 0.75 0.5; do
  [ "$L" != "1.0" ] && JOBS+=("kc_fd_$L|$P.LEVEL $L $ZERO $POS $FULL")          # jittered positions + FD velocity
  JOBS+=("kc_kfvel_$L|$P.LEVEL $L $ZERO $POS $FULL $P.VEL_ESTIMATOR kf")        # jittered positions + KF velocity
  JOBS+=("kc_kffull_$L|$P.LEVEL $L $ZERO $POS $FULL $P.VEL_ESTIMATOR kf $P.KF_POSITIONS True")  # KF positions + velocity
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
echo "=== Kalman-filter control finished $(date)"
