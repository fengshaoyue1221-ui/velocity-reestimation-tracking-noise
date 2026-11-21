#!/usr/bin/env bash
# Usage (from MTR/tools):
#   bash run_noise_sweep.sh <ckpt_path> <tag> [num_gpus=1] [cfg=cfgs/waymo/a100_clean.yaml] [total_batch_size=80]
# Evaluates one checkpoint under increasing noise levels, then a per-noise-type
# ablation at LEVEL=1 (each type alone). Results land in
#   ../output/<cfg_dir>/<cfg_name>/<tag>/eval/epoch_*/<eval_tag>/log_eval_*.txt
set -e
CKPT=$1; TAG=$2; NGPU=${3:-1}; CFG=${4:-cfgs/waymo/a100_clean.yaml}; BS=${5:-80}
P=DATA_CONFIG.PERCEPTION_NOISE
run () {  # $1 = eval tag, rest = --set overrides
  local et=$1; shift
  if [ "$NGPU" -eq 1 ]; then
    python test.py --cfg_file $CFG --ckpt $CKPT --extra_tag $TAG --eval_tag $et --batch_size $BS --set "$@"
  else
    bash scripts/dist_test.sh $NGPU --cfg_file $CFG --ckpt $CKPT --extra_tag $TAG --eval_tag $et --batch_size $BS --set "$@"
  fi
}
# 1) strength sweep (LEVEL 0 == clean evaluation)
for L in 0.0 0.5 1.0 2.0 3.0; do run level_$L $P.LEVEL $L; done
# 2) single-type ablation: zero all strengths, then enable one
ZERO="$P.POS_STD 0.0 $P.POS_RANGE_COEF 0.0 $P.HEAD_STD 0.0 $P.VEL_STD 0.0 $P.SIZE_STD 0.0 $P.DROP_PROB 0.0 $P.TRUNC_PROB 0.0 $P.IDSWAP_PROB 0.0"
run only_pos    $ZERO $P.POS_STD 0.15 $P.POS_RANGE_COEF 0.005
run only_head   $ZERO $P.HEAD_STD 0.05
run only_vel    $ZERO $P.VEL_STD 0.2
run only_drop   $ZERO $P.DROP_PROB 0.05
run only_trunc  $ZERO $P.TRUNC_PROB 0.1
run only_idswap $ZERO $P.IDSWAP_PROB 0.5
echo "done: $TAG"
