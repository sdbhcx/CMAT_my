# CMAT/LAS 多 Affordance 功能基底改动方案

## 1. 目标与范围

本方案将《基于 CMAT 的多 Affordance 功能基底建模实现方案》映射到当前仓库的真实代码结构，目标是在不破坏现有 LAS 基线训练的前提下，增加对象级多 affordance 联合训练、查询无关的功能基底、关系监督与对应评测。

当前仓库根目录实现的是 Stage-2 LAS，而不是文档中假设的独立 CAST decoder：

- `models/las_model.py` 使用 Point-MAE、DINO/RoBERTa、统一序列 co-attention 和逐点分割头；
- `data/piad_dataset.py` 与 `data/piadv2_dataset.py` 都按单个 image-affordance 样本返回数据；
- `train.py` 假设输出为 `[B, N, 1]`，损失只接收 logits 与单张 mask；
- `utils/metrics.py` 将整个验证集展平后计算 aIoU、AUC、SIM 和 MAE。

因此实现应保留 `model.name: las` 作为可复现基线，并新增 `model.name: fbd_afford`。第一版不修改 Stage-0/Stage-1 预训练代码，不实现 HAMMER 迁移，不加入不受约束的 segmentation residual。

## 2. 已确认的实现约束

### 2.1 数据约束

1. PIAD 的单个点云文件包含多列 affordance 标注，具备构建对象级 mask 矩阵的潜力；但当前训练只按图像中的 affordance 名选择一列，并按对象类别随机抽取点云，不保证 image 与 point cloud 属于同一实例。
2. PIADv2 当前训练按“对象类别 + affordance”随机配对，仍不保证同一对象实例；其 `.npy` 文件直接将 `data[:, 3:]` 当成 GT，必须先审计列数和路径中的实例标识。
3. 两个 loader 都在采样后才返回点云。对象内关系监督要求同一对象的全部 mask 共享一组采样索引，因此不能复用当前逐样本独立随机采样方式。
4. 当前路径解析大量使用 `path.split('/')`。对象级索引代码应统一使用 `pathlib.Path`/规范化相对路径，以兼容 Windows 与 Linux。
5. 数据未包含在仓库中，任何对象标识、点序一致性和多 affordance 覆盖率都不能仅凭代码假定，必须由审计报告确认。

### 2.2 模型约束

1. `LASModel` 的 `fused_point_features` 已经经过 point/prompt 联合 co-attention，是 cue-conditioned 特征，不能用于生成“查询无关”功能基底。
2. 可用于功能基底的接口是 Point-MAE group feature 经过 `point_projection` 和 `feature_propagation` 后的逐点特征，即加入 prompt 和 co-attention 之前的 `[B, N, C]` 张量。
3. 多 affordance batch 中同一对象的点云只能编码一次；图像提示可展平为 `[B*A, 3, H, W]` 编码，再恢复为 `[B, A, T, C]`。
4. 第一版 query 使用 projected prompt tokens 的 masked mean pooling。现有 co-attention 保留给 LAS 基线；FBD MVP 不让 prompt 改写 basis maps。

## 3. 目标数据与模型契约

对象级 dataset/collate 统一返回：

```python
{
    "points": Tensor[B, N, 3],
    "images": Tensor[B, A, 3, H, W],
    "masks": Tensor[B, A, N],
    "affordance_ids": LongTensor[B, A],
    "valid": BoolTensor[B, A],
    "object_ids": list[str],
    "category_ids": LongTensor[B],
}
```

约定：

- `A` 是 batch 内 padding 后的 `Amax`，首版最大为 4；
- padding 的 `affordance_ids` 使用 `-1`，其 image/mask 清零，所有 loss 和 metric 必须显式使用 `valid`；
- 每个对象先确定一组 canonical/sample indices，再一次性应用于所有 mask；
- 几何增强对同一对象的所有 affordance 共享，图像增强可独立；
- 验证与测试阶段 affordance 顺序稳定，按 affordance ID 排序，不随机采样。

FBD 模型输出：

```python
{
    "segmentation_logits": Tensor[B, A, N],
    "basis_logits": Tensor[B, N, K],
    "basis_maps": Tensor[B, N, K],
    "basis_descriptors": Tensor[B, K, C],
    "alpha": Tensor[B, A, K],
    "query_tokens": Tensor[B, A, C],
    "point_features": Tensor[B, N, C],
}
```

## 4. 分阶段实施计划

### 阶段 0：数据审计与 Go/No-Go 门槛

新增：

- `tools/audit_multi_affordance_data.py`
- `tools/build_object_affordance_index.py`
- `tools/align_affordance_masks.py`

产物默认写到未版本化的 `work/data_audit/<dataset>/<split>/`，包含 summary JSON、对象-affordance 共现、mask overlap、点序对齐和异常样本清单。仓库只提交 schema 示例和说明，不提交包含本地绝对路径的产物。

审计必须回答：

- 能否从路径或文件内容得到稳定的原始 object/mesh ID；
- 每个对象有多少 affordance、每个 affordance 有多少 HOI 图像；
- 同一对象跨 affordance 的坐标是否相同、是否仅点序不同、最近邻对齐误差是多少；
- PIADv2 `.npy` 的 mask 列数及其语义；
- train/test 与 Seen/Unseen split 是否存在 object ID 泄漏；
- 可组成 2/3/4-affordance set 的对象数量。

Go/No-Go：只有 object ID 可可靠恢复，且至少存在足够的多标注对象时，才启用对象内关系损失。否则保留单 affordance segmentation 样本，并将 `relation`、`coefficient_relation` 设为 0；禁止用类别均值伪造实例关系。

### 阶段 1：对象级索引、canonical 点云与 set-wise loader

新增：

- `data/object_affordance_index.py`：索引 schema、读写和一致性校验；
- `data/multi_affordance_dataset.py`：基于审计后的 index 加载对象；
- `data/multi_affordance_collate.py`：按 affordance 维 padding；
- `tests/data/test_object_affordance_index.py`；
- `tests/data/test_multi_affordance_dataset.py`。

实现顺序：

1. 先为 PIAD 实现 adapter，因为同一文件已包含多 affordance 列；再根据审计结果实现 PIADv2 adapter。
2. 索引中保存 `object_id`、`category`、canonical point source、每个 affordance 的 mask source、image 列表、对齐状态与误差。
3. 相同坐标不同点序时，保存确定性的重排索引；分别采样时，构建 2048 点 canonical cloud，并保存 mask 插值映射和距离质量。
4. loader 每个 epoch 最多选 4 个 affordance；有 overlap 统计时优先一组高重叠和一组低重叠 pair，其余随机选择。
5. 增加兼容模式，将对象级 batch 展平为单 affordance batch，送入原 `LASModel`，用于证明数据重构没有改变基线语义。

阶段验收：

- 合成 fixture 上所有输出形状、padding 和 `valid` 正确；
- 同一对象所有 mask 与 points 使用完全相同的采样索引；
- 固定 seed 可复现，测试阶段完全确定；
- 兼容模式与旧 loader 在同一固定样本上的 mask/point 对齐一致；
- 在真实数据 smoke run 中不出现跨实例或跨 affordance 错配。

### 阶段 2：Functional Basis MVP

新增：

- `models/functional_basis_generator.py`；
- `models/basis_selector.py`；
- `models/relational_affordance_model.py`；
- `tests/models/test_functional_basis.py`；
- `configs/piad_fbd.yaml`；
- `configs/piadv2_fbd.yaml`。

修改：

- `models/las_model.py`：抽取可复用的 `encode_points()`、`encode_visual_prompts()`，保持原 `forward()` 的行为与 checkpoint key 尽量稳定；
- `models/model_factory.py`、`models/__init__.py`：注册 `fbd_afford`；
- `train.py`：识别对象级 batch 和新输出。

核心计算：

1. `encode_points(points)` 只执行一次 Point-MAE、projection 与 feature propagation，得到 query-independent `[B, N, C]`；
2. `FunctionalBasisGenerator(C, hidden=256, K=8)` 输出 `[B, N, K]` basis logits/maps；
3. 使用归一化 basis maps 对 point features 加权池化，得到 `[B, K, C]` descriptors；
4. 仅对 `valid` 图像编码，提示 token 经过 projection 和 masked mean pooling 得到 `[B, A, C]` query；
5. `BasisSelector` 对 query 与 descriptors 做归一化点积和温度 softmax，得到 `[B, A, K]`；
6. `einsum("bnk,bak->ban")` 组合 basis logits，生成 `[B, A, N]` logits。

必须加入的断言/测试：

- 同一对象替换或重排 cue 时，`basis_logits` 不变；
- cue 重排只导致 `alpha` 与预测的 A 维同步重排；
- padding cue 不影响有效预测；
- basis descriptor 在全零权重的极端输入下不产生 NaN；
- CPU 小张量 forward/backward 可运行；
- `K=4/8/12/16` 均满足接口。

阶段验收：只启用 `Lseg + Lunion`，冻结 Point-MAE 和 prompt encoder 训练 10–20 epoch；Seen aIoU 相对冻结条件下的参数量匹配 baseline 下降不超过 2–3 个点，并能可视化出覆盖主要功能区域的 basis。

### 阶段 3：统一损失接口与对象内关系监督

新增：

- `losses/__init__.py`；
- `losses/functional_basis_loss.py`；
- `losses/relation_losses.py`；
- `losses/basis_regularization.py`；
- `tests/losses/test_functional_basis_losses.py`。

修改 `models/model_factory.py` 与 `train.py`，将 criterion 统一为：

```python
total_loss, loss_dict = criterion(outputs, batch)
```

为旧 LAS 增加薄 adapter，避免训练循环继续硬编码 `focal_loss`/`dice_loss` 两个字段。新的 loss dict 固定包含 `total`、`segmentation`，其余按配置启用，并由通用 logger 遍历记录。

损失启用顺序：

1. A：`segmentation=1.0`、`union=0.2`；
2. B：增加 `relation=0.5`、`coefficient_relation=0.1`；
3. C：增加 `basis_diversity=0.01`、`coefficient_sparsity=0.01`；
4. D：增加 `cross_object_correspondence=0.1`。

实现规则：

- focal/dice 只聚合 `valid` affordance；
- relation 只取上三角、有效且非自身 pair，单 affordance 对象返回可微的 0；
- union 的 GT 只聚合有效 mask；
- coefficient relation 使用 alpha cosine similarity；
- diversity 不强制正交，只使用小权重；
- correspondence 正样本必须是相同 affordance、不同 object ID，负样本按 mask overlap 降权；第一版只做 batch 内对比，不做 memory queue。

阶段验收：所有 loss 在 padding、单 affordance、空正样本和混合精度输入下有限；关系矩阵对称、对角线不计入 loss；关系监督启用后验证集 Relation MAE 相比阶段 A 下降。

### 阶段 4：训练器、优化器和 checkpoint 集成

修改 `train.py`：

- 将 dataloader 构建统一到一个 dataset factory，避免当前单机/分布式两套路径漂移；
- 支持对象级 batch、通用 loss dict 和逐 loss TensorBoard 日志；
- optimizer 分为 `point_encoder`、`prompt_encoder`、`basis_generator/selector`、其他 projection 四组；
- 阶段 A–C 冻结 backbone，阶段 D 只解冻 Point-MAE 最后 2 或 4 层；
- 使用 `head_lr=1e-4`、`backbone_lr=1e-5`，并确保 `requires_grad=False` 的参数不进入 optimizer；
- 将 loss stage、index 版本、affordance vocabulary、K 和 alignment policy 写入 checkpoint config；
- 若启用 AMP，则补齐 `autocast` 与 `GradScaler`，不能只保留未使用的配置字段；
- DDP 下 sampler 每个 epoch 调用 `set_epoch`，对象级随机采样由 `seed + epoch + rank` 控制。

恢复 checkpoint 时对 model/loss stage 不兼容给出明确错误；允许从 LAS checkpoint 以 `strict=False` 加载共享编码器，但必须打印 missing/unexpected keys 摘要。

### 阶段 5：评测、可视化和组合泛化

新增：

- `evaluation/evaluate_standard_metrics.py`；
- `evaluation/evaluate_affordance_relations.py`；
- `evaluation/evaluate_functional_retrieval.py`；
- `evaluation/visualize_functional_basis.py`；
- `tools/build_compositional_split.py`；
- 对应的 metric/split 单元测试。

修改 `utils/metrics.py` 或迁移其公共函数：标准指标必须按 `valid` 去除 padding，并同时报告 overall、per-affordance、per-object-category 和 split 指标。新增：

- Relation MAE；
- Recall@1、Recall@5、mAP；
- basis 两两 IoU、空 basis 比例、alpha entropy、每个 affordance 有效 basis 数；
- Seen、Unseen Object、Unseen Affordance 和 compositional split。

组合 split 生成器必须保证：目标 category-affordance 组合从训练集完全删除、测试 affordance 在其他类别出现、测试类别的其他 affordance 在训练中出现、object ID 无泄漏，并输出 split manifest 与校验报告。

可视化每个对象输出 point cloud、GT/pred maps、K 个 basis、alpha 柱状图以及 GT/pred relation matrix。生成图属于运行产物，不默认提交仓库。

### 阶段 6：扩展项（MVP 验收后）

按以下顺序处理表达能力不足：增加 K；softmax 改 sigmoid multi-selection；增加 query-to-basis cross-attention；最后才评估 `gamma <= 0.1` 的受限 residual。memory queue、HAMMER `[CONT]` query 适配和其他 3D 数据预训练均独立成实验，不混入首个可复现版本。

## 5. 配置设计

建议新增而不覆盖现有 LAS 配置：

```yaml
model:
  name: fbd_afford
  num_basis: 8
  basis_hidden_dim: 256
  selector_hidden_dim: 256
  selector_temperature: 0.07
  prompt_pooling: mean
  residual_weight: 0.0

data:
  index_path: work/index/piadv2_seen_train.json
  num_points: 2048
  max_affordances_per_object: 4
  alignment_max_distance: null  # 由审计结果确定，禁止先硬编码

loss:
  stage: A
  focal_alpha: 0.25
  focal_gamma: 2.0
  segmentation: 1.0
  relation: 0.0
  coefficient_relation: 0.0
  union: 0.2
  basis_diversity: 0.0
  coefficient_sparsity: 0.0
  cross_object_correspondence: 0.0

training:
  batch_size_objects: 4
  head_lr: 1.0e-4
  backbone_lr: 1.0e-5
  freeze_point_encoder: true
  freeze_prompt_encoder: true
```

配置加载时校验：stage 与非零 loss 权重一致；`num_basis > 0`；`max_affordances_per_object >= 1`；关系损失启用时 index 必须包含 alignment 质量字段。

## 6. 文件级变更清单

| 文件 | 计划变更 |
|---|---|
| `data/piad_dataset.py` | 保留旧 loader；复用解析函数时去除 `/` 字符串切分 |
| `data/piadv2_dataset.py` | 保留旧 loader；提取共享的路径、点云读取和 mask schema 校验 |
| `data/object_affordance_index.py` | 新增 index schema、校验和版本号 |
| `data/multi_affordance_dataset.py` | 新增 PIAD/PIADv2 对象级 adapter 与共享采样 |
| `data/multi_affordance_collate.py` | 新增 A 维 padding 和 valid mask |
| `models/las_model.py` | 抽取 point/prompt 编码方法，保持 LAS forward 兼容 |
| `models/functional_basis_generator.py` | basis logits/maps 与 descriptor pooling |
| `models/basis_selector.py` | object-conditioned selector |
| `models/relational_affordance_model.py` | 组装对象编码、cue 编码与 basis combination |
| `models/model_factory.py` | 注册模型与统一 criterion adapter |
| `losses/*` | 新增分阶段复合损失及 valid/pair masking |
| `train.py` | dataset factory、对象 batch、loss logging、参数组和 checkpoint 元数据 |
| `utils/metrics.py` | valid-aware/per-class 指标，或迁移后保留兼容导出 |
| `tools/*` | 审计、索引、对齐和组合 split 工具 |
| `evaluation/*` | 标准、关系、检索与可视化评测 |
| `configs/piad_fbd.yaml` | PIAD FBD 实验配置 |
| `configs/piadv2_fbd.yaml` | PIADv2 FBD 实验配置 |
| `tests/*` | 数据、模型、损失、指标与 split 测试 |
| `README.md` | 审计、索引构建、阶段训练与评测命令 |

## 7. 推荐提交序列

1. `data: add multi-affordance audit and index schema`
2. `data: add canonical object-level dataset and collate`
3. `model: expose query-independent point features`
4. `model: add functional basis generator and selector`
5. `loss: add staged functional basis objectives`
6. `train: support object-level batches and criterion outputs`
7. `eval: add relation retrieval and basis diagnostics`
8. `data: add compositional split builder`
9. `docs: document FBD experiments and ablations`

每个提交都应保持原 `model.name: las` 配置可加载；涉及张量契约的提交必须同时包含单元测试。

## 8. 最终验收矩阵

| 阶段 | 必须通过的验收 |
|---|---|
| 数据审计 | object ID、点序、mask schema、多标注覆盖率和 split 泄漏有可复查报告 |
| Set-wise Dataset | 同对象共享 points/采样索引；旧 LAS 兼容模式指标基本一致 |
| Basis MVP | basis 与 cue 无关；`Lseg+Lunion` 可训练；Seen aIoU 降幅不超过约 2–3 点 |
| 关系监督 | Relation MAE 下降；Unseen Object/Affordance 不出现系统性退化 |
| 基底正则 | collapse/empty-basis 指标改善，且 grasp/lift 等允许共享 basis |
| 跨对象对应 | 跨类别 retrieval 优于无 correspondence loss 的版本 |
| 组合泛化 | 无 object 泄漏，未见 category-affordance 组合稳定优于原 LAS/CMAT baseline |

所有主结果至少运行 3 个随机种子，报告均值和标准差；A0–A7 使用相同 checkpoint、点数、图像分辨率、增强和 epoch。若数据审计无法支持实例级多 affordance 关系，则应在进入模型开发前缩减目标，而不是用不可靠匹配继续训练。
