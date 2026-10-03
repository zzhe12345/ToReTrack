# 多无人机多目标跟踪

本项目提供从双视角图像到最终跨视角跟踪结果的代码。

## 环境安装

验证环境为 Windows、Python 3.12、PyTorch 2.12.1 + CUDA 12.6、torchvision 0.27.1、MMCV 1.5.0、MMDetection 2.28.2、MMClassification 0.23.2。其他依赖版本在 `requirements.txt` 中。

安装 Python 3.12 和兼容的 NVIDIA 驱动后，在项目目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File setup_environment.ps1
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = 'src;third_party/mia_net_official'
python -B check_runtime.py
```

## 下载后训练，再直接测试

准备好外部 MDMT 数据并安装依赖后，执行：

```powershell
python train.py --data-root "E:/MDMT/datafull"
python test.py
```

也可以先修改 `configs/pipeline.json` 中的路径，然后直接运行 `python train.py`。

训练结束自动生成 `<work_root>/trained_pipeline.json`。`test.py` 自动读取该配置，加载新训练权重，执行完整测试流程和官方指标评估，不用手动复制权重或修改内部路径。若使用了另一工作目录，则传入 `python test.py --config <work_root>/trained_pipeline.json`。

```powershell
# 可选初始权重，必须与当前模型结构相容
python train.py --initial-esod "E:/models/esod.pt" --initial-autoassign "E:/models/autoassign.pth"

# 已有冻结检测器权重，只训练拓扑分支
python train.py --stage topology

# 训练完成后，先用一个序列前三帧检查测试流程
python test.py --scenes 26 --max-frames 3 --visualize
```

`--smoke` 会减少训练轮数和检测器图像尺寸，仅用于工程运行检查，不能用于复现完整指标。完整训练参数在 `detector_training` 和 `training` 中。阈值只从验证集选择；若验证集没有可靠的身份修复证据，测试配置会禁用拓扑身份修改并明确提示，保留 MIA 关联。

## 推理与评估

```powershell
# 完整单序列推理
python run_pipeline.py --phase infer --split test --scenes 26

# 前三帧运行检查，可选带标注可视化
python run_pipeline.py --phase infer --split test --scenes 26 --max-frames 3 --visualize

# 对完整结果评估；三帧结果则同样添加 --max-frames 3
python run_pipeline.py --phase evaluate --split test --scenes 26
```

最终 JSON 使用 `frame=0`、`frame=1` 等键，每个目标为 `[id, x1, y1, x2, y2]`。双视角相同 ID 表示关联到同一目标。

评估入口从外部 XML 生成零基帧 MOT 标注和一基帧官方 MDA 标注，再调用 `evaluate_mia_metrics.py` 与上游 `mango_eval.py`。完整指标需在完整测试序列上计算。

测试入口固定评估完整方法的最终结果：

| 指标   | 视角 1 | 视角 2 | 总体      |
| ---- | ---- | ---- | ------- |
| MDA  | —    | —    | 双视角关联得分 |
| MOTA | 分别计算 | 分别计算 | 两视角联合计算 |
| IDF1 | 分别计算 | 分别计算 | 两视角联合计算 |
| IDS  | 分别计数 | 分别计数 | 两视角次数相加 |

## 单独运行各环节

各阶段也可独立运行，参数用 `--help` 查看。`run_pipeline.py --phase prepare --split train` 和 `--split val` 构建冻结检测器的拓扑特征，`--phase train` 训练拓扑分支。统一的 `train.py` 已自动串联这些步骤。

ESOD 结构和超参数在 `configs/uavdt_yolov5m.yaml`、`configs/esod_hyp.yaml`；AutoAssign 配置及实际使用的基础配置保留在 `third_party/mia_net_official/configs/`。默认检测器训练轮数分别为 5 和 60，文件名用于稳定加载；修改训练轮数不会改变导出文件名。

## 代码导航

| 环节         | 入口 / 实现                                                                                |
| ---------- | -------------------------------------------------------------------------------------- |
| 完整训练 / 测试  | `train.py`、`test.py`                                                                   |
| 数据转换与检测器训练 | `prepare_detection_data.py`、`train_esod.py`、`train_autoassign.py`                      |
| 分阶段流程      | `run_pipeline.py`                                                                      |
| 检测         | `generate_esod_detection_cache.py`、`src/uav_tracking/generate_esod_detection_cache.py` |
| 跟踪与几何关联    | `run_mia_frontend.py`、`run_mia_official_geometry.py`                                   |
| 拓扑特征       | `build_mia_frame_topology_cache.py`、`src/uav_tracking/mia_features.py`                 |
| 拓扑模型和训练    | `src/uav_tracking/identity_topology_model.py`、`train_identity_topology.py`             |
| 身份修正       | `apply_identity_topology_to_mia.py`                                                    |
| 因果恢复       | `apply_mia_causal_forward_fill.py`                                                     |
| 特征协议检查     | `check_feature_protocol.py`                                                            |
| 评估和标注转换    | `evaluate_mia_metrics.py`、`build_mot_gt_from_xml.py`                                   |
| 可视化        | `visualize_results.py`                                                                 |

代码注释说明输入输出、时序约束、特征池化和帧编号约定。`tests/` 是源代码测试，不包含实验结果；安装 pytest 后执行 `python -B -m pytest -q`。
