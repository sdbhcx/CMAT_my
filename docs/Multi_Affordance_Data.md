# 阶段 0 / 1：数据审计与对象索引

> 当前 PIADv2 实验采用类别 + affordance 图像提示，以及经过核查的相同采样几何上的多 mask。请先阅读 [PIADv2 Stage A 实施与验收](PIADv2_Stage_A.md)。下文的实例图像 manifest 是 v1 严格模式；其图像实例身份要求及原始 NO_GO 不能用来否定当前 v2 类别提示协议。

所有命令在仓库根目录执行。工具依赖 NumPy、Pillow；loader/测试另需 PyTorch。服务器使用 `conda run --no-capture-output -n cmat ...`。运行产物写到 `work/`，不提交源数据、绝对路径或生成图。

## 1. 无映射时先审计原始数据

```bash
python -m tools.audit_multi_affordance_data --dataset piad --data-root /home/datasets/PIAD_root
python -m tools.audit_multi_affordance_data --dataset piadv2 --data-root /home/datasets/PIAD2
```

根目录应包含 `<setting>/Point_<split>.txt` 和 `Img_<split>.txt`，同时兼容 PIAD 的首字母大写 split。清单中 `Data/` 前缀会移除；其余路径相对于 `--data-root`。Windows 反斜杠会规范化，禁止绝对路径、`..` 及越界符号链接。

默认扫描全部清单。`--workers 8` 可增加有界文件读取并发，输出顺序保持确定。`--max-files 50` 仅抽取每个 partition 最前面的 50 个点文件，报告会明确标为 sampled；这不替代全量审计。`--output-dir` 可覆盖默认 `work/data_audit/<dataset>/all/`。

报告包含文件列数及语义、PIAD 文本原始 UUID、每类/affordance 图像数、非空 mask 覆盖与 overlap、相同坐标/点序不同的候选组，以及跨 split/setting 的 ID/坐标重复。跨 setting 的重复可能来自实验协议复用，需要与同一 setting 内的 train/test 泄漏分别解释。

`point_object_candidates.json` 按原始 UUID（未知时仅按坐标候选）汇总点云来源；`cooccurrence.json` 按 partition 统计非空 mask 共现。`object_summary.json` 单独列出同一 setting 内跨 split 的重复组。它们是审计材料，缺少图像实例对应时不能直接传给训练 loader。

**原始清单本身不足以证明 HOI 图像与点云是同一个实例。** PIAD 的 UUID 能识别点云，但图像文件名不提供这个 UUID；PIADv2 的 `3D_Aff`、`MVPNet` 等目录是数据来源。坐标相同仅作为候选匹配，不作为新的实例身份。无外部映射时报告 `NO_GO`，不会伪造可训练对象索引，也不会自动开启关系损失。

## 2. 提供可复查的实例映射 manifest

参考 [object_affordance_manifest.example.json](object_affordance_manifest.example.json)。示例故意设置 `identity_verified: false`，不能直接当作已审计数据。

- `object_id` 必须是跨 affordance、split 和 setting 稳定的原始 ID，不能附加 split 来隐藏泄漏。PIAD 文本源要求 ID/category 与文件每行前两个字段一致。
- `identity_evidence` 记录图像与点云属于同一实例的外部标注来源或已确认的规则。确认后才设 `identity_verified: true`。
- `point_source.format`: `piad_txt` 跳过两列字符串元数据，读取 xyz 和后续 mask；`piadv2_npy` 读取前三列 xyz。
- `mask_columns` 从 xyz 后的第一个 mask 开始计数，必须完整覆盖文件所有 mask 列，不允许猜测 PIADv2 多列语义。PIAD 的既有 17 类顺序见 `data/object_affordance_index.py`，也必须与实际文件列数匹配。
- `images` 仅列出有可靠实例映射、对应 mask 已知的 affordance；一个 affordance 可以有多张图。没有图像映射的 mask 不进入训练 set。
- 路径始终相对于 data root。根路径只在运行参数或本地配置中设置，索引可迁移。
- 一个对象可有多行，分别指向不同 affordance 源文件；重复 affordance 只有在对齐后的 mask 相同时才能合并图像。
- **同一个 manifest 应包含全部待比较的 train/val/test 与 setting。** 工具只能检查提供的数据。若将不同实验协议拆成独立 manifest，应另外查看全量原始审计的交叉重复报告。

## 3. 审计并构建索引

```bash
python -m tools.audit_multi_affordance_data --manifest work/mapping.json --data-root /path/to/data --split train --setting Seen --min-multi-objects 100
python -m tools.build_object_affordance_index --manifest work/mapping.json --data-root /path/to/data --split train --setting Seen --min-multi-objects 100 --index-path work/index/train.json
python -m tools.build_object_affordance_index --manifest work/mapping.json --data-root /path/to/data --split test --setting Seen --min-multi-objects 100 --index-path work/index/test.json
```

`--min-multi-objects` 默认 1 仅用于技术 smoke test；真实实验请依据规模确定阈值。只有身份已验证、图像可解码、所有 mask 语义明确、点序对齐通过且多标注对象达到阈值，并且 manifest 无 partition 泄漏时，`relation_ready` 才为 true。单标注对象仍可进入索引用于分割；`NO_GO` 索引禁止启用关系损失。一个源出错会拒绝整个对象，原因写入 `anomalies.json`。

默认只接受坐标完全相同或确定性点序重排。独立采样必须显式指定 `--max-distance`，单位为**原始坐标单位**，不自动缩放、ICP 或设置通用阈值。检查双向最近邻最大误差，使用 3NN 逆距离插值；精确重合点使用原值。独立采样时构建默认 2048 点 canonical cloud（可用 `--canonical-count` 调整），保存可重放的 canonical indices、源索引和权重。全零坐标或重复坐标不产生除零；重排歧义且标签冲突时拒绝。

独立检查某一对源文件：

```bash
python -m tools.align_affordance_masks --data-root /path/to/data --canonical cloud-a.npy --source cloud-b.npy --format piadv2_npy --max-distance 0.002 --output work/aligned.npz
```

上例距离仅演示参数格式，不能直接作为真实数据阈值。索引 JSON 保存版本、词表、类别表、审计门槛、alignment policy、图像清单、mask 来源和质量。`objects/*.npz` 为内容寻址的规范化产物，读取时校验 SHA256；修改源数据后应重新审计和构建。

## 4. 接入训练和 LAS 兼容模式

在独立实验配置中添加以下设置，保留原 LAS / A=1 配置：

```yaml
dataset_type: piadv2  # PIAD 使用 piad
seed: 42
data:
  index_paths:
    train: work/index/train.json
    test: work/index/test.json
  max_affordances_per_object: 4
  num_points: 2048
  image_size: [224, 224]
  use_augmentation: true
  object_las_compat: false
paths:
  data_root: /path/to/data
```

`model.name: fbd_afford` 会使用对象 loader。只配置 `index_path` 会报错，要求显式指定训练/验证索引，避免误用同一 split；两者词表须相同，object ID 不得重叠。未配置 index_paths 时继续使用原 A=1 loader。

训练时每对象最多选择 4 个 affordance，有 overlap 时优先选高、低 overlap pair，其余随机补齐。对象内所有 mask 使用完全相同的点索引和几何增强；图像增强独立。采样由 seed/epoch/rank/object ID 控制，不依赖 Python hash 或 worker 调度。训练器每轮调用 dataset.set_epoch；当前 loader 不启用 persistent workers。

验证时按 affordance ID 排序，并将超过 Amax 的对象拆成连续 chunks，**覆盖全部 affordance**；同一对象各 chunk 的点与采样完全相同。这些 chunk 不能当成不同物体进行关系评测。

使用 `model.name: las`、`model.prompt_type: visual`、`data.object_las_compat: true` 可将有效 cue 展平为旧 LAS batch。points/mask 直接复制，不重新采样，用于同一固定样本的对齐比较。此模式不是对旧 loader 随机类别配对结果的数值复现，也不支持文字 prompt。

本次不实现 Stage B–D 损失：即便审计为 GO，训练仍只用已有 segmentation + union；未实现的损失继续由 loss factory 拒绝。

## 5. 验证

```bash
conda run --no-capture-output -n cmat python -m unittest discover -s tests -v
```

新增测试涵盖索引读写与篡改、身份/语义缺失拒绝、泄漏、精确重排与插值映射重放、距离门槛、共享采样、确定性、不同 worker 数、eval chunk、overlap 选择及 LAS 展平。真实多 affordance loader 验收必须使用相应协议下经过核查的索引；v1 要求实例图像映射，v2 要求相同点载体上的 mask 与有效类别功能图像池，均不能用合成映射冒充真实验收。
