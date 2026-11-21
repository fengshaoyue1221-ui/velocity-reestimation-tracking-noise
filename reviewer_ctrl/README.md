# Reviewer controls for the camera-ready (evaluation only)

From the repository root, env "mtr":
  python reviewer_ctrl/enable_acc_switches.py      # adds ACC_FROM_CLEAN / VEL_FEAT_CLEAN switches (idempotent)
  bash reviewer_ctrl/run_reviewer_ctrl.sh           # 42 evaluations, ~4-4.5 h on 3 GPUs, resumable (run in tmux)
  python reviewer_ctrl/collect_reviewer_ctrl.py     # tables + reviewer_ctrl_outputs/reviewer_ctrl.json

Conditions (clean-trained and mixed-trained models):
  rc_check_1.0          both switches on: must reproduce the clean-input mAP (correctness check)
  rc_fd_clean           clean positions, velocity = finite difference of clean positions      (item 1)
  rc_velfeat_only_L     re-derived velocity in the velocity feature, clean acceleration (L=1, 0.5)   (item 5)
  rc_acc_only_L         clean velocity feature, acceleration from re-derived velocity (L=1, 0.5)     (item 5)
  rc_{pos_offset,vel_derived,both,all,all_cleanvel}_L   paired interventions at L=0.35, 0.5, 0.75 (item 2)
