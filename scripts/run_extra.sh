#!/usr/bin/env bash
# Supplementary analyses for the 8-page paper. Run AFTER `run_all.sh all` has finished.
# No retraining: 2 extra GPU evaluations (uniform class factors) + CPU-only analyses.
# Usage (from kit root, env mtr active):  PROFILE=a2000 NGPU=3 bash analysis/run_extra.sh
set -eo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd); MTR=$KIT/MTR; TOOLS=$MTR/tools
PROFILE=${PROFILE:-a2000}; NGPU=${NGPU:-3}; EVAL_BS=${EVAL_BS:-24}
CFG=cfgs/waymo/${PROFILE}_clean.yaml
EVROOT=$MTR/output/waymo/${PROFILE}_clean
EXTRA=$KIT/paper/results/extra; mkdir -p "$EXTRA"
export TF_FORCE_GPU_ALLOW_GROWTH=true
LOG=$KIT/run_extra.log; exec > >(tee -a "$LOG") 2>&1
echo "=== run_extra $(date)"

# 1) enable optional per-class factors (defaults unchanged)
python "$KIT/analysis/enable_class_scale.py"

# 2) uniform-class-factor evaluation at LEVEL 1 for both models (GPU)
P=DATA_CONFIG.PERCEPTION_NOISE
for spec in "clean:$MTR/output/waymo/${PROFILE}_clean/clean/ckpt/checkpoint_epoch_30.pth" \
            "aug:$MTR/output/waymo/${PROFILE}_aug/aug/ckpt/checkpoint_epoch_30.pth"; do
  tag=${spec%%:*}; ck=${spec#*:}
  if ls "$EVROOT/$tag"/eval/epoch_30/uniform_class/log_eval_*.txt >/dev/null 2>&1; then echo "uniform_class $tag exists, skip"; continue; fi
  cd "$TOOLS"
  args="--cfg_file $CFG --ckpt $ck --extra_tag $tag --eval_tag uniform_class --batch_size $EVAL_BS --set $P.LEVEL 1.0 $P.CLASS_SCALE.TYPE_PEDESTRIAN 1.0 $P.CLASS_SCALE.TYPE_CYCLIST 1.0"
  if [ "$NGPU" -gt 1 ]; then bash scripts/dist_test.sh "$NGPU" $args; else python test.py $args; fi
done

# 3) constant-velocity reference (CPU)
cd "$MTR"
[ -f "$EXTRA/cv_baseline.json" ] || CUDA_VISIBLE_DEVICES="" python "$KIT/analysis/cv_baseline.py" --cfg tools/$CFG --levels 0 0.5 1 2 3 --out "$EXTRA/cv_baseline.json"

# 4) distance-binned degradation (CPU, uses saved result.pkl)
for tag in clean aug; do
  python "$KIT/analysis/distance_bins.py" --eval_root "$EVROOT/$tag/eval/epoch_30" --clean level_0.0 --noisy level_1.0 --out "$EXTRA/dist_${tag}.json"
done

# 5) tables (horizon table comes straight from the sweep logs)
python "$KIT/analysis/extra_tables.py" --eval_base "$EVROOT" --extra "$EXTRA"
echo "=== run_extra finished $(date)"
