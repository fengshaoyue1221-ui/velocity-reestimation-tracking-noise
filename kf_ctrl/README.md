# Kalman-filter control (evaluation only, ~2 h)
From the repository root, env "mtr":
  python kf_ctrl/enable_kf.py        # adds VEL_ESTIMATOR / KF_POSITIONS / FORCE_FULL_REEST (idempotent, defaults unchanged)
  bash kf_ctrl/run_kf_ctrl.sh         # 18 evaluations, resumable (run in tmux)
  python kf_ctrl/collect_kf_ctrl.py   # table + reviewer_ctrl_outputs/kf_ctrl.json
