#!/usr/bin/env bash
# Download a WOMD subset (scenario format): the first N training shards (of 1000, ~487 scenes each)
# and the first M validation shards (of 150, ~294 scenes each).
#   A2000 workstation plan:  N=50, M=50   (~24k train / ~15k val scenes)
#   A100 plan:               N=200, M=150 (~97k train / full validation)
#
# BEFORE running:
#  1. Register at https://waymo.com/open/ with a Google account and accept the license
#     for the Motion dataset.
#  2. Install the Google Cloud CLI without sudo:
#       curl https://sdk.cloud.google.com | bash ; exec -l $SHELL
#  3. Log in with THAT Google account (headless server):
#       gcloud auth login --no-launch-browser
#  4. Open the Motion dataset download page and confirm the bucket name of the
#     version you want. MTR's published numbers use v1.1 -> default below.
#
# Usage:  bash scripts/download_womd.sh <dest_dir> [bucket_version=v_1_1_0] [n_train_shards=50] [n_val_shards=50]
set -euo pipefail
DEST=${1:?usage: download_womd.sh <dest_dir> [version] [n_train_shards] [n_val_shards]}
VER=${2:-v_1_1_0}
NTRAIN=${3:-50}
NVAL=${4:-50}
BUCKET="gs://waymo_open_dataset_motion_${VER}/uncompressed/scenario"
mkdir -p "$DEST/training" "$DEST/validation"

echo "Checking access to $BUCKET ..."
gsutil ls "$BUCKET/validation/validation.tfrecord-00000-of-00150" || { echo "Cannot list bucket. Check version name and license acceptance."; exit 1; }

TRAIN_LIST=$(mktemp); VAL_LIST=$(mktemp)
for i in $(seq 0 $((NTRAIN-1))); do printf "%s/training/training.tfrecord-%05d-of-01000\n" "$BUCKET" "$i"; done > "$TRAIN_LIST"
for i in $(seq 0 $((NVAL-1)));     do printf "%s/validation/validation.tfrecord-%05d-of-00150\n" "$BUCKET" "$i"; done > "$VAL_LIST"

# -n: skip files that already exist -> safe to re-run after interruption.
# URLs are passed as arguments (reading them from stdin with -I stops early in recent gsutil versions).
gsutil -m cp -n $(cat "$VAL_LIST")   "$DEST/validation/"
gsutil -m cp -n $(cat "$TRAIN_LIST") "$DEST/training/"

nt=$(ls "$DEST/training" | wc -l); nv=$(ls "$DEST/validation" | wc -l)
echo "training shards: $nt / $NTRAIN   validation shards: $nv / $NVAL"
du -sh "$DEST"
[ "$nt" -eq "$NTRAIN" ] && [ "$nv" -eq "$NVAL" ] && echo "DOWNLOAD COMPLETE" || echo "INCOMPLETE - re-run the same command"
