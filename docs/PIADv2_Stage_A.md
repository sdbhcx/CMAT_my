# PIADv2 多 affordance Stage A

## 最新：LAS 分割损失与 union 对照已完成

两组均完成 10+20 轮训练，三组最佳模型的统一验证及可视化已完成。FBD union=0 的固定验证 aIoU 为 0.15082，高于 union=0.2 的 0.14467，仍低于 LAS 的 0.16528。详细结果和限制见 [统一诊断](PIADv2_Stage_A_Union_Comparison.md)。下方“未启动”表述为早期实现/预检记录。


## LAS 分割损失对照已实现（2026-09-20）

新增 FBD 的 LAS 同款分割损失选项，保留 union=0.2，旧配置默认不变。55 项测试及真实 batch 前后向预检通过；尚未启动完整训练。配置与复现说明见 [LAS 分割损失对照](PIADv2_Stage_A_LAS_Segmentation.md)。


## 当前：第 11～30 轮后台续训（2026-09-20）

两组已从各自第 10 轮的最新 checkpoint 恢复模型与 AdamW 状态，继续训练 20 轮，总计 30 轮。数据划分、骨干冻结、object batch size 2 和 FBD 固定温度 0.2 不变。新的后续 20 轮余弦计划从 5e-5 衰减到 1e-6；原 10 轮调度已完成，不直接沿用其低学习率继续。

- FBD：GPU 2，启动 PID 239481，配置 `configs/piadv2_fbd_afford_holdout_30ep.yaml`。
- LAS：GPU 3，启动 PID 239482，配置 `configs/piadv2_las_holdout_30ep.yaml`。
- 日志/状态：`work/stage_a/controlled/continue30_20260920_114200/` 下的 `fbd.log`、`las.log`、`fbd_status.json`、`las_status.json`。
- 启动记录仍可通过 `work/stage_a/controlled/active_run.json` 查询。原 10 轮作业与 checkpoint 保留，最佳 checkpoint 会复制至新目录并继续按验证 aIoU 更新。

`train.py` 已修正从恢复后的 epoch 开始循环；已完成第 10 轮时，内部零基 epoch 从 10 开始，对应用户第 11 轮。续训入口参数为 `--resume <checkpoint.pth> --restart-lr 5e-5`，拒绝模型/数据配置不一致，并保存 `resume.json`。恢复后的真实最大功能数 batch 前后向检查均通过。

旧 checkpoint 没有随机数状态，本次使用 restart seed=52；不宣称逐位复现不中断运行。状态历史保留前 10 轮，后续每轮记录实际学习率和验证指标。

## 2026-09-20：已启动 10 轮受控训练

服务器项目目录 `/home/junbo/wyn/codes/CMAT_my`，环境 `cmat`。按类别将 5,115 个多标注训练对象划为 4,603 个训练对象与 512 个验证对象（seed 42）。提示图像池也独立划分，训练 5,526 张、验证 616 张，路径不重叠。正式测试集不参与训练或模型选择。两组从预训练骨干重新初始化，不加载此前看过整个训练集的诊断 checkpoint。

| 作业 | GPU | PID | 配置 |
|---|---:|---:|---|
| FBD，K=8，固定温度 0.2 | 2 | 110833 | `configs/piadv2_fbd_afford_holdout_10ep.yaml` |
| LAS，同划分对照 | 3 | 110834 | `configs/piadv2_las_holdout_10ep.yaml` |

均训练 10 个 epoch，object batch size 2、学习率 1e-4、冻结完整骨干，每轮 train 有 2301 个 batch（drop_last 丢弃最后一个对象），val 有 256 个 batch。LAS 展平有效提示，因此实际 image-point 样本数随 batch 变化。LAS 使用原有 focal/dice 实现（权重均为 1），FBD 另有 union 项；两者损失数值不能直接当作同一指标比较。

运行目录：`work/stage_a/controlled/run_20260920_101730/`，包含 `fbd.log`、`las.log`、各自的 `*_status.json` 和启动信息。PID 是启动时记录，判断进程状态需结合命令行和日志。状态文件每轮记录验证指标，FBD 另记录选择器分布、熵与温度；异常写入 failed，结束写入 completed。最新启动记录为 `work/stage_a/controlled/active_run.json`。

`tools/prepare_stage_a_controlled.py` 生成独立对象/提示划分及配置；`tools/run_stage_a_controlled.py` 调用正式 UnifiedTrainer，检查固定温度不在优化器内、完整骨干冻结、loss/梯度有限。正式入口现支持 `data.validation_split: val` 和 `model.selector_temperature_fixed: true`；未指定时保留原行为。后台作业已脱离 SSH，会话结束后继续运行。

以下是此前一轮验收的记录。

当前实验只使用 PIADv2 的 `Unseen_obj`，在 195 服务器的 `/home/junbo/wyn/codes/CMAT_my` 中运行，环境为 `cmat`。

## 数据规则与索引

HOI 图像按 **object category + affordance** 匹配，是类别功能提示，不要求与点云来自同一物理实例。同一对象的多个 mask 必须拥有完全相同的 xyz 和点序；对象标识表示已核查的采样几何，不宣称已恢复原始 mesh ID。不把文件名数字后缀作为实例标识。

v2 索引分别记录 `image_matching_policy: category_affordance` 和 `point_identity_policy: exact_sampled_geometry`。图像池按 setting、split、category、affordance 隔离，索引保存 mask 来源、几何指纹、共享点索引和 NPZ 校验值。构建时重新检查实际坐标、重复标注一致性以及图像解码，默认遇到 train/test 几何重复即拒绝。

`Unseen_obj` 实测 train 有 28,003 个几何对象，其中 5,115 个具有至少两个 affordance；test 有 1,365 个对象，其中 53 个具有两个 affordance。train/test 无精确几何交集，两份索引均为 `GO`。其他 setting 不混入本实验；Seen 和 Unseen_aff 的重复项需要单独处理。

在已有全量原始审计后，复现核查和构建：

```bash
conda run --no-capture-output -n cmat python -m tools.verify_piadv2_object_groups --data-root /home/datasets/PIAD2 --workers 8
conda run --no-capture-output -n cmat python -m tools.build_piadv2_geometry_index --data-root /home/datasets/PIAD2 --setting Unseen_obj --output-dir work/index/piadv2/Unseen_obj --workers 8 --min-multi-objects 20
```

核查输入默认为 `work/data_audit/piadv2/all/point_inventory.json`，可通过 `tools.audit_multi_affordance_data --dataset piadv2 --data-root /home/datasets/PIAD2` 生成。构建报告为 `work/index/piadv2/Unseen_obj/build_report.json`。

## 训练与验收

独立配置：`configs/piadv2_fbd_unseen_obj_stage_a.yaml`。配置使用 `model.name: fbd_afford`、`loss.stage: A`，冻结 Point-MAE 和 DINOv3；只训练投影、功能基底和选择器等新增模块。loss 为 segmentation + 0.2 × union，关系与其他 Stage B–D 损失保持关闭。

`data.min_train_affordances: 2` 选择 5,115 个多标注训练对象，`max_affordances_per_object: 4`，对象 batch size 为 8。每个对象共享点采样与几何增强；点云只编码一次，只编码 valid 的图像提示。测试保留全部对象和全部 affordance。

服务器默认 CUDA 库搜索路径包含旧版库；以下命令为当前进程选择 CUDA 11.8，避免与 cmat 中的 PyTorch 2.0.0+cu118 冲突。GPU 编号应根据运行时空闲情况选择。

```bash
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 conda run --no-capture-output -n cmat python -m tools.validate_piadv2_stage_a --config configs/piadv2_fbd_unseen_obj_stage_a.yaml --short --output-dir work/stage_a/short
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 conda run --no-capture-output -n cmat python -m tools.validate_piadv2_stage_a --config configs/piadv2_fbd_unseen_obj_stage_a.yaml --output-dir work/stage_a/full_epoch
```

短运行使用 32 个训练 batch 和 8 个测试 batch，不能代替完整测试指标。整轮运行从预训练骨干重新初始化训练，使用 639 个训练 batch；现有训练 loader 的 `drop_last=True` 使本轮访问 5,112 个对象，剩余三个不进入该轮。完整测试不丢弃末尾 batch。

验证工具调用真实 `UnifiedTrainer`，检查实际 A=2/3/4 样本的确定性、padding、有限 loss 和梯度，并严格重载 checkpoint 比较预测，恢复 AdamW 状态后执行一次更新。每次输出：

- `report.json`：实际批次数、训练/验证 loss、AUC/aIoU/SIM/MAE、梯度与 checkpoint 检查结果；
- `loss_trace.json`：逐 batch 的 loss、对象数和有效 affordance 数；
- `real_object_gt_predictions.png`：同一点云的四个功能，展示 HOI 提示、GT 和预测；
- `work/stage_a/checkpoints/<experiment>/`：最新与最佳 checkpoint，具体路径见报告。

一轮验收确认真实多 affordance 训练链路可运行，不代表最终性能已收敛。这里验证的是权重/优化器状态重载并继续一步；没有把 CLI 的跨轮断点续训作为本次验收项。

2026-09-19 实际验收结果：42 项测试通过。32 batch 短训练和整轮训练的梯度、checkpoint 预测一致性、优化器恢复更新均通过。整轮训练 loss 为 1.13317，完整测试 loss 为 1.19432，AUC 为 0.51693，aIoU 为 0.05099，SIM 为 0.35811，MAE 为 0.21902。可视化中的四种功能预测仍较相似，尚不能据此认定已经学到充分的功能区分。

```bash
conda run --no-capture-output -n cmat python -m unittest discover -s tests -v
```

服务器现有 `.gitignore` 忽略 `configs/`、`tests/` 和 `work/`；运行配置、测试和产物均保留在服务器，交付或提交时需显式纳入所需源码与配置。
