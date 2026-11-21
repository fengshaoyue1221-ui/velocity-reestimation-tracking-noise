# Velocity Re-estimation Amplifies Motion Forecasting Errors under Tracking Noise

Code for the WACV 2027 workshop paper *"Velocity Re-estimation Amplifies Motion Forecasting Errors under Tracking
Noise"*. The repository adds a structured, temporally correlated tracking-noise model to
[Motion Transformer (MTR)](https://github.com/sshaoshuai/MTR) and contains every script used for the experiments,
tables and figures of the paper, including the controls added for the camera-ready version.

## What the code does
- **Tracking-noise model** `C_lambda` (Sec. 2.3): range-dependent AR(1) position error, heading error, velocity
  re-estimation by finite differences, size error, frame dropout / late start, identity switches
  (`mtr_modified_files/mtr/datasets/waymo/perception_noise.py`, unit tests in `mtr_modified_files/tests/`).
- **Paired position-velocity interventions** that separate displacement from velocity re-estimation (Sec. 2.6, 3.3).
- **Training variants**: clean-trained, noise-augmented (q = 1), mixed (q = 0.5), no-velocity model.
- **Camera-ready controls**: clean finite-difference velocity, velocity vs. acceleration features, paired
  interventions at sensor-level severities, and a constant-velocity Kalman-filter baseline.

## Setup
Tested with Python 3.8, PyTorch 1.13.1, TensorFlow 2.6.0, waymo-open-dataset-tf-2-6-0 1.4.8, on 3x 12 GB GPUs.

    git clone https://github.com/<user>/velocity-reestimation-tracking-noise.git && cd velocity-reestimation-tracking-noise
    git clone https://github.com/sshaoshuai/MTR.git && (cd MTR && git checkout a5ba7bdafa09a1a355cc34f8a895499a2b14ddb3 \
        && git apply --whitespace=nowarn ../mtr_full.patch)
    bash scripts/setup_env.sh          # conda env "mtr"
    bash analysis/download_womd.sh     # WOMD v1.1: training and validation shards 0-49 (needs WOMD access)

`mtr_full.patch` contains all changes to MTR (noise model, dataset hooks, evaluation fixes, configs); readable copies
of the modified files are in `mtr_modified_files/`. Data (24,265 training / 14,610 validation scenarios, 63,646
target agents) is preprocessed by `scripts/run_all.sh`.

## Reproducing the paper (run from the repository root, env `mtr`)

    export TF_FORCE_GPU_ALLOW_GROWTH=true PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    PROFILE=a2000 NGPU=3 TOTAL_BS=6 EVAL_BS=12 bash scripts/run_all.sh all   # preprocess, train, main evaluations
    PROFILE=a2000 NGPU=3 EVAL_BS=12 bash scripts/run_extra.sh                # class scale, CV baseline, distance bins
    bash mechanism/run_mechanism.sh                                         # paired position/velocity interventions
    bash velsmooth/run_velsmooth.sh                                         # test-time least-squares velocity
    bash supp/run_supp.sh                                                   # evaluation seeds, mixed training
    bash final/run_final.sh                                                 # rho / beta sensitivity, no-velocity model

Camera-ready controls (evaluation only, after the steps above):

    python reviewer_ctrl/enable_acc_switches.py && bash reviewer_ctrl/run_reviewer_ctrl.sh && python reviewer_ctrl/collect_reviewer_ctrl.py
    python kf_ctrl/enable_kf.py && bash kf_ctrl/run_kf_ctrl.sh && python kf_ctrl/collect_kf_ctrl.py

Verification and figures (from `MTR/`):

    python ../verification/recompute_metrics.py          # full-precision metrics from the evaluation outputs
    python ../verification/residual_stats.py --split validation
    python ../verification/injected_error_stats.py --levels 0.25 0.5 1.0 --max_scenes 3000
    python ../plotting/make_paper_figures.py

| Paper item | Script |
|---|---|
| Main results across severities (Table 3, Fig. 2) | `scripts/run_all.sh`, `supp/run_supp.sh`, `final/run_final.sh` |
| Paired interventions (Table 5, Fig. 3) | `mechanism/run_mechanism.sh` |
| Temporal correlation, distance scaling (Fig. 4b, Sec. 3.6) | `final/run_final.sh` |
| Class and distance breakdown (Table 6) | `scripts/run_extra.sh` |
| Least-squares velocity, constant-velocity baseline (Table 7) | `velsmooth/run_velsmooth.sh`, `scripts/run_extra.sh` |
| Camera-ready controls (severity table, finite-difference control, feature pathways, Kalman filter) | `reviewer_ctrl/`, `kf_ctrl/` |
| Error-magnitude statistics, full-precision metrics | `verification/` |

## Results
`results/` holds the evaluation logs, full-precision metrics (`metrics_full_precision.csv`), error statistics and the
environment record; `results/camera_ready/` holds the camera-ready summaries and the list of every evaluation
condition with its effective settings (`conditions.csv`). Metrics use the waymo-open-dataset 1.4.8 evaluator;
newer versions of the library compute a different mAP.

## Citation

    @inproceedings{velocity_reestimation_2027,
      title     = {Velocity Re-estimation Amplifies Motion Forecasting Errors under Tracking Noise},
      author    = {<authors>},
      booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
      year      = {2027}
    }

## License
Built on MTR (Apache License 2.0); the patch and modified MTR files keep that license. See `NOTICE`.
