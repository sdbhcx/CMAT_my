# Stage A 提示敏感性与固定对象过拟合

在 195 服务器 `/home/junbo/wyn/codes/CMAT_my`、`cmat` 环境完成。只用 PIADv2 Unseen_obj 训练对象；没有读取测试集调参，没有修改正式模型或启动长训练/Stage B。

## 实现

`tools/diagnose_piadv2_stage_a.py` 从已验收的一轮 checkpoint 开始，按类别轮流选择 16 个真实对象，要求功能 mask 的平均绝对差至少 0.02。此子集用于诊断，不代表数据总体。固定 seed、图像、点云、功能集合和采样索引，关闭增强和 dropout，保留新增模块的梯度，骨干冻结。每批 4 个对象，AdamW 学习率 0.001、weight decay 0、梯度裁剪 1。

对原提示分别进行对象内循环置换、重复第一张有效提示、替换为归一化均值图像。GT 和 padding 保持原位。固定前向随机种子，检查置换后输出是否对应置换、功能基底是否保持不变。记录选择器熵、有效温度、基底选择分布、基底描述相似度、逐对象功能预测差异以及各模块裁剪前梯度。CUDA 计算不承诺逐比特复现。

## 复现

在服务器仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 conda run --no-capture-output -n cmat python -m tools.diagnose_piadv2_stage_a \
  --config configs/piadv2_fbd_unseen_obj_stage_a.yaml \
  --checkpoint work/stage_a/checkpoints/fbd_piadv2_unseen_obj_stage_a_piadv2_unseen_obj_20260919_233923/checkpoint.pth \
  --steps 2000 --output-dir work/stage_a/diagnosis_2000
```

默认 400 步。两次运行均从上述一轮 checkpoint 开始，2000 步实验不是接着 400 步的优化器状态继续。产物在 `work/stage_a/diagnosis/` 和 `work/stage_a/diagnosis_2000/`：报告、固定子集与图像张量指纹、逐步梯度/loss、预测 NPZ、训练后权重。2000 步目录还保存 loss 曲线和 GT/预测图。

## 2026-09-20 实测

| 指标 | 初始 checkpoint | 400 步后 | 2000 步后 |
|---|---:|---:|---:|
| 固定子集 loss | 1.15654 | 0.90175 | 0.89570 |
| 相对 loss 降幅 | — | 22.03% | 22.55% |
| 对象内功能预测平均差异 | 0.00608 | 0.02341 | 0.02699 |
| 选择器归一化熵 | 0.19294 | 0.00318 | 0.0000543 |
| 打乱提示后的 loss | 1.15677 | 1.00491 | 1.02609 |

真实 mask 的对象内功能平均差异为 0.13429。2000 步后，38 个有效提示的基底 argmax 分布为 `[1,35,2,0,0,0,0,0]`，有效温度触及 0.02 下限；13/16 个对象的功能预测差异小于 1e-6。提示置换等变检查通过（最大误差约 9.54e-7），更换提示时基底最大变化为 0，所有训练步梯度有限。提示分支并未断开，但出现明显的选择器集中与功能区分不足。

GT 是软标签。即使直接预测 GT，当前 focal/dice/union 组合也不为零；该参考 loss 为 0.78941，且它不是理论最小值。脚本中的 loss 降幅 50%、预测差异达到 GT 差异的 25%、打乱提示使 loss 增加 0.01 是保守经验门槛，不是收敛定理。本次暂缓扩大训练还依据逐对象输出及选择分布，并非只依据 loss 门槛。

2000 步预测的 GT MAE 为 0.09879；使用 GT 计算的每点跨功能中位数，其共享预测 MAE 为 0.07411。后者是利用 GT 的诊断参照，不是可部署基线，也不是泛化指标。

下一步应在同一子集做选择器对照：固定较高温度与现有可学习温度比较，观察有效基底数量、逐对象功能差异和梯度是否恢复，再决定是否修改正式模型。当前证据指向选择器饱和，但尚未通过对照实验证明它是唯一原因。
