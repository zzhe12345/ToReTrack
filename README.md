# ToReTrack: Multi-UAV Multi-Object Tracking

This project provides code for generating cross-view tracking results from images captured by two UAVs.

## Demonstration Videos

The following videos show the final tracking results. Click the play button to watch each demonstration.

The colors in both videos indicate cross-view matching results:

- **Green:** Correct matches.
- **Yellow:** Missed matches.
- **Red:** Incorrect matches.

### Demo 1

https://github.com/user-attachments/assets/409bdeee-6af9-4926-9bd7-33be25e50059

### Demo 2

https://github.com/user-attachments/assets/16277a3b-71b1-44ea-8c3e-be10ad0ddb97

## Environment Setup

The validated environment uses Windows, Python 3.12, PyTorch 2.12.1 with CUDA 12.6, torchvision 0.27.1, MMCV 1.5.0, MMDetection 2.28.2, and MMClassification 0.23.2. Other dependency versions are listed in `requirements.txt`.

After installing Python 3.12 and a compatible NVIDIA driver, run the following commands from the project directory:

```powershell
powershell -ExecutionPolicy Bypass -File setup_environment.ps1
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = 'src;third_party/mia_net_official'
python -B check_runtime.py
```

## Train and Test

After preparing the external MDMT dataset and installing the dependencies, run:

```powershell
python train.py --data-root "E:/MDMT/datafull"
python test.py
```

Alternatively, update the paths in `configs/pipeline.json` and run `python train.py`.

Training automatically generates `<work_root>/trained_pipeline.json`. The `test.py` entry point reads this configuration, loads the newly trained weights, and runs the complete test pipeline and official evaluation. No manual copying of weights or changes to internal paths are required. If you use a different working directory, run `python test.py --config <work_root>/trained_pipeline.json`.

```powershell
# Optional initial weights must match the current model architecture
python train.py --initial-esod "E:/models/esod.pt" --initial-autoassign "E:/models/autoassign.pth"

# Train only the topology branch using existing frozen detector weights
python train.py --stage topology

# After training, check the test pipeline on the first three frames of one sequence
python test.py --scenes 26 --max-frames 3 --visualize
```

The `--smoke` option reduces the number of training epochs and detector image sizes. It is intended only for checking that the pipeline runs and cannot reproduce full evaluation scores. Full training parameters are specified in `detector_training` and `training`. Thresholds are selected using only the validation set. If the validation set provides no reliable evidence for identity repair, the test configuration disables topology-based identity corrections, reports this explicitly, and retains the MIA associations.

## Inference and Evaluation

```powershell
# Run inference on a complete sequence
python run_pipeline.py --phase infer --split test --scenes 26

# Check the first three frames, optionally with visualizations against annotations
python run_pipeline.py --phase infer --split test --scenes 26 --max-frames 3 --visualize

# Evaluate complete results; also add --max-frames 3 for three-frame results
python run_pipeline.py --phase evaluate --split test --scenes 26
```

The final JSON files use keys such as `frame=0` and `frame=1`. Each target is represented as `[id, x1, y1, x2, y2]`. The same ID in both views indicates that the detections are associated with the same target.

The evaluation entry point converts external XML annotations into zero-based MOT annotations and one-based annotations for the official MDA evaluator, then calls `evaluate_mia_metrics.py` and the upstream `mango_eval.py`. Full evaluation scores must be computed on complete test sequences.

The test entry point evaluates only the final results of the complete method:

| Metric | View 1 | View 2 | Overall |
|---|---|---|---|
| MDA | — | — | Cross-view association score |
| MOTA | Computed separately | Computed separately | Computed jointly across both views |
| IDF1 | Computed separately | Computed separately | Computed jointly across both views |
| IDS | Counted separately | Counted separately | Sum of both views |

## Run Individual Stages

Each stage can also be run independently. Use `--help` to view its arguments. The commands `run_pipeline.py --phase prepare --split train` and `--split val` build topology features using frozen detectors; `--phase train` trains the topology branch. The unified `train.py` entry point already connects these steps automatically.

The ESOD architecture and hyperparameters are defined in `configs/uavdt_yolov5m.yaml` and `configs/esod_hyp.yaml`. The AutoAssign configuration and its required base configurations are retained in `third_party/mia_net_official/configs/`. The default detector training schedules use 5 and 60 epochs, respectively. Exported filenames remain fixed for consistent loading; changing the number of training epochs does not change these filenames.

## Code Guide

| Stage | Entry Point / Implementation |
|---|---|
| Complete training / testing | `train.py`, `test.py` |
| Data conversion and detector training | `prepare_detection_data.py`, `train_esod.py`, `train_autoassign.py` |
| Stage-by-stage pipeline | `run_pipeline.py` |
| Detection | `generate_esod_detection_cache.py`, `src/uav_tracking/generate_esod_detection_cache.py` |
| Tracking and geometric association | `run_mia_frontend.py`, `run_mia_official_geometry.py` |
| Topology features | `build_mia_frame_topology_cache.py`, `src/uav_tracking/mia_features.py` |
| Topology model and training | `src/uav_tracking/identity_topology_model.py`, `train_identity_topology.py` |
| Identity correction | `apply_identity_topology_to_mia.py` |
| Causal recovery | `apply_mia_causal_forward_fill.py` |
| Feature protocol checks | `check_feature_protocol.py` |
| Evaluation and annotation conversion | `evaluate_mia_metrics.py`, `build_mot_gt_from_xml.py` |
| Visualization | `visualize_results.py` |

Code comments describe inputs and outputs, temporal constraints, feature pooling, and frame-index conventions. The `tests/` directory contains source-code tests and no experiment results. After installing pytest, run `python -B -m pytest -q`.
