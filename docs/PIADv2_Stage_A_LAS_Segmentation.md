# FBD 使用 LAS 同款分割损失

## 实现

以服务器提交 `f2a7429` 为基线，增加 `loss.segmentation_loss: las`。FBD 先通过 valid 取出有效提示，将 logits/GT 转为 `[有效提示数,N,1]`，直接调用已有 `models.las_model.LASLoss`，使用相同软标签 focal、整批前景/背景 Dice 及权重。union 仍使用原有 noisy-OR BCE+Dice，不受分割损失切换影响。

默认 `segmentation_loss: fbd` 保留旧行为：分割项仍是原有 focal+逐提示前景 Dice，历史配置中的 focal_weight 不会突然改变旧实验结果。LAS 模式才使用 focal_weight/dice_weight；本次对照配置二者均为 1，与原 LAS holdout 对照一致。日志仍返回 segmentation、focal、dice、union 等原有字段。

LAS Dice 定义包含常数 1.5，可能出现负值，不应把总 loss 的绝对大小与旧 FBD 实验比较。主要比较验证 aIoU/AUC/SIM/MAE。

## 对照配置及训练方法

新增配置：
- `configs/piadv2_fbd_las_seg_holdout_10ep.yaml`
- `configs/piadv2_fbd_las_seg_holdout_30ep.yaml`

与原 FBD 对照保持相同数据划分、seed=42、K=8、固定温度 0.2、冻结骨干、object batch size=2、优化器和学习率阶段。分割项切换到 LAS，union=0.2 保留。

先从预训练骨干重新初始化训练 10 轮，不加载旧 FBD 最佳模型。以下命令仅为运行说明，本次未启动完整训练：

```bash
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 conda run --no-capture-output -n cmat python -m tools.run_stage_a_controlled --config configs/piadv2_fbd_las_seg_holdout_10ep.yaml --status work/stage_a/controlled/fbd_las_seg_10ep_status.json
```

10 轮完成后，从该新分支 status 中 experiment_dir 对应的最新 `checkpoint.pth` 续训，使用 30ep 配置、`--resume <新分支checkpoint.pth> --restart-lr 5e-5`，status 使用独立的 `work/stage_a/controlled/fbd_las_seg_30ep_status.json`。30ep 配置的 previous_status 已指向上述 10ep status。此流程匹配原有 10+20 轮预算，不能改成直接运行连续 30 轮后声称调度完全相同。

受控恢复入口现在比较 checkpoint 与当前 loss 配置；不同损失会拒绝恢复，防止改变目标后仍把旧最佳指标作为新分支基线。

## 验证（2026-09-20）

- 服务器 cmat 全量 unittest：55 项通过。
- 新增测试验证软/硬 GT 下与 LAS 损失及梯度完全一致、padding 不参与分割、union 不变且梯度有限、默认 FBD 兼容、非法模式拒绝及损失不同的续训拒绝。
- 新配置与原 FBD 的 model/data/training/seed/paths 字段一致性检查通过。
- GPU 2 上通过真实最大功能数 batch 前向、反向、AdamW 单步预检，完整骨干冻结和固定温度检查通过；loss=1.509148836，峰值 PyTorch CUDA 已分配内存约 616.6 MiB。这不是完整训练的性能或总显存基准。
- 预检状态为 `work/stage_a/controlled/fbd_las_seg_preflight_status.json`，history 为空。预检权重未用于新实验初始化，正式训练会重新建模。
