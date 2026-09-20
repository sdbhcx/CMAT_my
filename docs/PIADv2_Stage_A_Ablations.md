# Stage A 温度与基底数量对照

## 范围和控制条件

在 195 服务器 `/home/junbo/wyn/codes/CMAT_my` 的 `cmat` 环境运行。沿用诊断实验的 16 个真实 PIADv2 Unseen_obj 训练对象及 38 个有效提示。所有分支固定相同对象、图像、点云、GT、seed=42、每批 4 对象、2000 次更新、AdamW lr=0.001、无 weight decay、无随机增强和 dropout，骨干冻结。没有使用正式测试集调参。

本次新运行五个分支，并引用前次相同预算的原温度实验：

| 分支 | 功能基底头 | 温度 |
|---|---|---|
| learned_original | 一轮 checkpoint 的 K=8 头 | 延续原可学习温度 |
| fixed007 | 同一 checkpoint 的 K=8 头 | 固定 0.07 |
| fixed020 | 同一 checkpoint 的 K=8 头 | 固定 0.20 |
| learnable020 | 同一 checkpoint 的 K=8 头 | 重置为 0.20 后学习 |
| fresh_k1 | 新初始化 K=1 头 | 固定 0.20 |
| fresh_k8 | 新初始化 K=8 头 | 固定 0.20 |

K=1/K=8 分支恢复相同骨干和投影权重，只重建各自的头；不从已训练 K=8 中选择一个基底。头的第一层等共享形状参数采用相同随机种子，但不同 K 的结构与参数量自然不同。共享的投影权重已在此前 K=8 一轮训练中更新，因此该对照是同一现有特征下的诊断，不是完全独立预训练的最终基准比较。

K=1 的 softmax 恒等于 1，在当前架构下不能利用查询区分功能；它是功能区分的退化对照，不等价于 LAS。不同 K 还会改变 union 的组成，因此同时比较 segmentation loss、GT MAE 和逐对象功能差异，不只看总 loss。

`tools/summarize_stage_a_ablations.py` 校验所有分支的子集清单、图像张量指纹、点坐标、mask、valid、初始 checkpoint、学习率及训练步数一致后才生成汇总。

## 复现入口

```bash
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 conda run --no-capture-output -n cmat python -m tools.diagnose_piadv2_stage_a \
  --config configs/piadv2_fbd_unseen_obj_stage_a.yaml \
  --checkpoint work/stage_a/checkpoints/fbd_piadv2_unseen_obj_stage_a_piadv2_unseen_obj_20260919_233923/checkpoint.pth \
  --steps 2000 --temperature 0.2 --fixed-temperature \
  --output-dir work/stage_a/ablations/fixed020
```

- fixed007：温度改为 0.07。
- learnable020：去掉 `--fixed-temperature`。
- fresh_k1 / fresh_k8：在固定 0.2 的命令上分别添加 `--fresh-basis-count 1` / `--fresh-basis-count 8`。
- 每组使用独立输出目录。原温度参考位于 `work/stage_a/diagnosis_2000/`。

```bash
conda run --no-capture-output -n cmat python -m tools.summarize_stage_a_ablations
```

产物包括每组的报告、梯度轨迹、预测、图像和 checkpoint，以及 `work/stage_a/ablations/summary.json`、`comparison.png`。GPU 应依据当时空闲状态选择。固定 seed 不保证 CUDA 运算逐比特相同。

这些都是单 seed、固定训练子集的机制对照，不能用于宣称测试集提升、优于 LAS 或创新已经成立。保守经验门槛 `ready_for_long_training` 也不是软标签损失的理论判据；应结合对照指标与实际预测分析。

## 2026-09-20 实测结果

| 分支 | 总 loss | 分割 loss | GT MAE | 功能预测几乎相同的对象 |
|---|---:|---:|---:|---:|
| learned_original | 0.89570 | 0.69932 | 0.09879 | 13/16 |
| fixed007 | 0.80520 | 0.60655 | 0.08719 | 1/16 |
| fixed020 | 0.77191 | 0.56743 | 0.07762 | 0/16 |
| learnable020 | 0.80255 | 0.60323 | 0.08327 | 3/16 |
| fresh_k1 | 0.96298 | 0.75747 | 0.11884 | 16/16 |
| fresh_k8 | 0.76909 | 0.56639 | 0.07680 | 0/16 |

“几乎相同”定义为同一对象功能预测两两平均绝对差 < 1e-6；避免该状态并不等于已准确预测全部功能。

固定 0.2 相比原实验的 GT MAE 降低约 21.4%。其选择器最后 100 步梯度范数中位数约 1.55e-3，原实验约 4.89e-7。将温度初始化为 0.2 但继续学习，会再次降到下限 0.02，并留下 3 个退化对象；这支持在后续验证中将固定 0.2 作为候选设置。没有自动修改正式模型的默认参数。

同一公共特征初始化下，K=8 相比 K=1 的 GT MAE 降低约 35.4%，分割 loss 从 0.75747 降到 0.56639。K=8 打乱提示使 loss 增加 0.52327，而 K=1 不变。这提供了多基底可以利用提示进行功能区分的正面证据。但 K=1 是结构上不具备查询区分能力的弱对照，尚不能据此证明功能基底优于其他合理的查询条件分割头。

固定 0.2 的 K=8 仍有约 58%～61% 的提示选择同一 argmax 基底，并非所有基底均被充分利用。下一阶段需要多个 seed、训练集内划分的验证集，以及同骨干/数据/训练预算的 LAS 对照；正式测试集继续留作最终评估。

新增单基底诊断回归测试后共 47 项测试通过；全部新实验均完成 2000 次有限梯度更新。汇总同时验证实际点云、GT、valid 数组完全相同。
