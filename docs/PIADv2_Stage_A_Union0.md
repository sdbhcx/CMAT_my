# FBD LAS 分割损失：union=0 对照（2026-09-20）

以 195 服务器现有 union=0.2 分支为对照，仅将 loss.union 改为 0；分割仍复用 LAS focal/Dice，权重均为 1。K=8、固定温度 0.2、完整骨干冻结、seed=42、batch size=2、数据划分与优化器保持不变。配置经逐字段比较，除名称和 previous_status 路径外仅 union 权重变化。

GPU 0，从预训练骨干重新初始化；先 10 轮（1e-4 余弦调度），再恢复本分支最新 checkpoint 和 AdamW 状态，重启 lr=5e-5 续训 20 轮，总计 30 轮。后段使用既有 restart seed=52 规则。未加载 union=0.2 已训练权重。

运行目录：`/home/junbo/wyn/codes/CMAT_my/work/stage_a/controlled/union0_20260920_212334/`。

- 配置快照：config_10ep.yaml、config_30ep.yaml。
- 状态：pipeline_status.json，以及 10ep_status.json、30ep_status.json（后段启动后生成）。
- 日志：10ep.log、30ep.log、launcher.log。
- 后台调度 PID：1306906；最初 10 轮训练 PID：1308028。PID 仅是启动记录，应结合命令行和状态判断。
- 调度脚本：上级目录 run_union0_union0_20260920_212334.py。脱离 SSH 运行，预检通过后自动顺序执行两段训练；失败会记录错误并停止。

真实最大功能数 batch 的前向、反向和 AdamW 单步预检已通过，状态为 preflight_passed。预检模型丢弃，正式训练重新初始化。本文记录时已进入正式 10 轮训练阶段，尚无完整实验结果。

union 分支仍被计算和记录用于诊断，但乘以 0，不参与总目标或参数更新。比较重点为验证 aIoU/AUC/SIM/MAE、逐对象功能区分及选择器分布；不要把总 loss 下降直接视为效果提升。

## 完成记录

两段训练均 completed。已将实际 config_10ep.yaml、config_30ep.yaml 原样保存为 configs/piadv2_fbd_las_seg_union0_holdout_10ep.yaml 与 configs/piadv2_fbd_las_seg_union0_holdout_30ep.yaml，供版本记录。30ep 的 previous_status 保留原实验路径；复现新实验需改为新分支的 10ep status 路径，并使用该分支最新 checkpoint。完整结论见 [统一诊断](PIADv2_Stage_A_Union_Comparison.md)。
