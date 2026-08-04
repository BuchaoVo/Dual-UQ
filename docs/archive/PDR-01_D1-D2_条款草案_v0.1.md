# PDR-01：D1 序列不等价处置 / D2 长度与 fragment 入选标准（条款草案 v0.1）

| 项 | 值 |
| --- | --- |
| 文档性质 | 协议决策记录（Protocol Decision Record），冻结后为规范性文件 |
| 状态 | **草案，未冻结**。§0.3 的待验证项全部关闭前不得冻结 |
| 覆盖决策 | D1、D2（实验设计方案 v0.1 §3） |
| 证据基础 | Round-1 eligibility census，`reports/dataset_a_census/round1_report.json` |
| 冻结后果 | manifest schema 扩展、`p1.py` 定向修改、`pipeline_version` 升版至 `dataset-a.v2` |
| 起草日期 | 2026-08-02 |

---

## 0. 证据基础与待验证项

### 0.1 Round-1 census 结果

| 指标 | 值 |
| --- | --- |
| 本地 pair 目录 | 28 |
| 跳过（`no_identity_source`） | 2（早期 pilot 遗留，不属任一候选池） |
| 可评估 | 26 |
| P0 阶段失败 | 7 |
| P1 阶段失败 | 11 |
| 完整通过（`p1_pairing_ok`） | 8 |
| 通过率 | 30.8%（Wilson 95% CI 约 14%–50%） |
| `pdb_amino_acid_mismatch` 命中 | 6 |
| mismatch 计数分布 | [6, 1, 1, 1, 1, 1] |

三个交叉验证案例与 A8 handoff §9 的既有只读结论完全一致（index 9 `unsupported_afdb_fragment`；index 36 `missing_mapping_column`；index 103 `pdb_amino_acid_mismatch` 且 `mismatch_count=1`），据此认定 harness 结果可用于协议决策。

**取样偏差声明**：这 28 个 pair 是已落盘的子集，不是 213 候选的随机样本，很可能富集早期筛选中表现较好的蛋白。因此上表的通过率应视为**乐观上界**。

**截尾声明**：pipeline 分阶段依赖，各 failure_code 的计数是"首次失败"分布，不是患病率。任何一个码的真实发生率 ≥ 报告值。

### 0.2 对规模目标的直接影响

按 30.8% 外推 213 候选，最终可用蛋白约 **30–107**（CI 区间）。对照实验设计方案 §4.2：

- **A1 = 64**：仅在 CI 上半区可达；
- **A2 = 128–256**：即使按 CI 上界亦不可达。

**结论性建议（需 T0 单独立项处理，不属本 PDR）**：A2 规模目标须下调，或须扩大候选来源。本 PDR 的 D1 取值会显著改变这一测算（见 §1.7）。

### 0.3 冻结前必须关闭的待验证项

| # | 待验证项 | 责任 | 阻断 |
| --- | --- | --- | --- |
| V1 | 7 个 P0 失败的码分布，其中 `unsupported_afdb_fragment` 的计数 | T1 | D2 §2.3 阈值 |
| V2 | 11 个 P1 失败中，除 6 个 mismatch 外另 5 个的码 | T1 | 规模测算 |
| V3 | 6 个 mismatch 蛋白的完整位点列表（`output_position` / `uniprot_position` / `mapping_aa` / `pdb_aa`）、`mapped_length`、`sequence_identity`、`max_consecutive_run` | T1 | D1 §1.2 阈值校准、§1.4 允许表条目 |
| V4 | 对 5 个孤立 mismatch 蛋白，带假定允许表定向重跑 `resolve_p1_inputs`，确认其确实能到达 `p1_pairing_ok` 而非仅仅推迟到下一个失败 | T1 | D1 §1.7 收益估计 |
| V5 | AFDB 分片规则的实测确认（片长、步长、触发分片的长度阈值） | T1 | D2 §2.3 表述 |

**V4 尤其关键**：P1 的 mismatch 检查是 fail-fast，这 5 个蛋白在 mismatch 之后是否还有其他 P1 失败（缺 N/CA/C/O、AFDB model 位置异常等）目前未知。因此"恢复 5 个"是上界而非估计值。

---

## 1. D1 — 序列不等价的处置

### 1.1 问题陈述

P1 要求每个 output position 上 mapping / PDB / AFDB 三方 canonical AA 完全一致。真实数据中该检查会因两类**成因完全不同**的原因失败：

- **真实序列变异**：工程化构建体、点突变体、isoform 差异、SNP、克隆痕迹。结构本身是正确的，只是与 UniProt canonical 序列在个别位点不同。
- **映射错误**：frame shift、错误 entity、错误链、错误 UniProt 归属。此时整段对应关系是错的。

**严格等价检查的科学目的是 mapping sanity check，不是要求两侧必须是同一变体。** paired backbone 仅含 N/CA/C/O 四个原子，残基身份在 backbone 中是元数据而非几何。以零容忍处理会系统性剔除被充分研究的工程化结构——恰是解析质量最高的一批。

### 1.2 分类判据（规范性）

对每个命中 `pdb_amino_acid_mismatch` 的蛋白，计算：

```text
mismatch_count            不匹配位点总数（非 fail-fast 全量扫描）
mapped_length             P1 配对残基总数
sequence_identity         1 − mismatch_count / mapped_length
max_consecutive_run       在 output_position 上最长连续不匹配段长度
min_pairwise_spacing      任意两个不匹配位点间的最小 output_position 间距
```

判为 **variant-like**（可进入允许表）当且仅当全部满足：

```text
mismatch_count       ≤ 3
max_consecutive_run  = 1
min_pairwise_spacing ≥ 5
sequence_identity    ≥ 0.99
```

否则判为 **mapping-error-like**，一律剔除，不得进入允许表。

> **阈值校准状态**：上述四个阈值为草案值，须以 V3 的真实分布校准后定稿。已知锚点：index 103 为 `mismatch_count=1`、`mapped_length=240`、`sequence_identity≈0.9958`，应落在 variant-like 侧；census 中的 6-mismatch 蛋白应落在 mapping-error-like 侧。`min_pairwise_spacing = 5` 是纯草案值，V3 之前无任何证据支持。

### 1.3 决策

**采纳"声明式变体允许表"（方案 C），附加全局 gate。**

否决方案 A（数据集层面一律剔除）：在 26 个可评估蛋白中命中 6 个，代价过大，且系统性偏向剔除工程化结构。
否决方案 B（更换 PDB entry 或 isoform）：仅对少数情况可行，且引入新的 provenance 分支，不具备通用性。保留为个案手段。

**核心原则：容忍度作为被 hash 的数据存在，代码中不出现任何静默容忍路径。**

### 1.4 manifest schema 扩展（规范性）

新增可选字段 `authorized_sequence_variants`，为数组，缺省为空数组：

```json
{
  "authorized_sequence_variants": [
    {
      "uniprot_position": 92,
      "mapping_aa": "D",
      "pdb_aa": "N",
      "evidence_class": "engineered_construct",
      "evidence_note": "<人工审核依据，须可追溯>",
      "reviewer": "<审核人标识>",
      "review_date": "<ISO 8601>"
    }
  ]
}
```

规范要求：

1. 该字段进入 `input_digest` 的 canonical 化范围。任何增删改触发 `blocked_input_drift`。
2. `evidence_class` 取值受限于枚举：`engineered_construct` / `point_mutant` / `isoform_difference` / `natural_variant` / `expression_artifact`。不得新增 `unknown` 一类。
3. `evidence_note` 与 `reviewer` 为必填非空。**没有人工审核依据的变体不得声明。**
4. 数组按 `uniprot_position` 升序，重复 position 为 schema 错误。
5. 不使用 `output_position` 作为键——`output_position` 由 P1 派生，manifest 不应依赖下游产物。

### 1.5 P1 契约修改（需显式授权）

**这是对冻结模块 `p1.py` 的定向修改，本 PDR 即为 A8 handoff §13 所要求的"显式协议决策"。**

现行行为（`p1.py:684-691`）：遇到第一个不匹配位点即 raise，不继续扫描。

修改后行为：

1. 全量扫描所有 output position，收集**观测 mismatch 集合** `{(uniprot_position, mapping_aa, pdb_aa)}`；
2. 与 manifest 的 `authorized_sequence_variants` 构成的**声明集合**做**精确集合相等**比较；
3. 相等 → 通过，并把变体位点写入 provenance（见 §1.6）；
4. 不相等 → 失败，`failure_code` 按 §1.8 区分三种情形，`details` 中同时给出观测集合与声明集合的对称差。

**不变量**：

- 声明集合为空且观测集合非空时仍然失败——即默认行为不放松，只是失败信息从"首个位点"变为"全部位点"，严格更 informative。
- AFDB 侧的 canonical AA 必须与 mapping 一致，**不受允许表豁免**。允许表只豁免 mapping↔PDB 的差异（AFDB 由 UniProt canonical 序列预测而来，与 mapping 不一致意味着 mapping 或 fragment 定位有误，属真错误）。
- 不得新增 force flag、不得新增环境变量开关、不得在代码中内置任何默认允许位点。

**连带影响**：`tests/dataset_a_scale/test_p1.py` 中断言首位点错误载荷的用例需同步更新；本次修改须与测试更新在同一次提交内完成。`pipeline_version` 升版至 `dataset-a.v2`，使 `config_digest` 与 v1 时代产物自然区隔。

### 1.6 下游义务（规范性）

1. **provenance 标记**：`backbone_residue_provenance.tsv` 增列 `authorized_variant`（布尔），标记该 output position 是否属声明变体。
2. **P7 排除**：声明变体位点**不得**进入 P7 候选位点池。理由：该位点上两个结构条件对应的是**不同的化学实体**，其序列决策差异混杂了变体效应与结构条件效应，不能作为 H4 的因果证据。`authorized_variant` 属 mapping cleanliness 范畴，P7 可读，不违反盲法。
3. **P6/P9 分层**：报告含变体蛋白与不含变体蛋白的效应是否存在系统差异，作为敏感性分析。
4. **论文披露**：dataset construction 章节须报告声明变体的蛋白数、位点数、`evidence_class` 分布，并说明 §1.7 的局限。

### 1.7 已知局限（须在论文中披露）

尽管 backbone 只含 N/CA/C/O，声明变体位点上两个结构条件仍对应不同的化学实体：侧链不同可能局部影响主链几何。因此：

- 变体位点本身**不是**干净的配对反事实单位（已由 §1.6.2 排除出 P7）；
- 变体对邻近残基几何的影响未被量化。若 V3 显示变体位点邻域（±3 残基）在效应量上系统偏离，须追加邻域排除规则。

### 1.8 失败码（规范性）

| 码 | 触发条件 |
| --- | --- |
| `pdb_amino_acid_mismatch` | 观测集合 ⊄ 声明集合（存在未声明的不匹配位点） |
| `unused_authorized_variant` | 声明集合 ⊄ 观测集合（声明了实际不存在的变体，提示 manifest 与输入不同步） |
| `sequence_variant_budget_exceeded` | 观测集合满足 §1.2 的形状判据但 `mismatch_count > 3` |
| `sequence_identity_below_threshold` | `sequence_identity < 0.99` |
| `mapping_error_like_mismatch` | `max_consecutive_run > 1` 或 `min_pairwise_spacing < 5` |

后三者为独立码，**不得**折叠进 `pdb_amino_acid_mismatch`——D2 之后的 attrition 表需要区分"变体预算超限"与"映射疑似错误"。

### 1.9 验收测试（规范性）

1. 声明集合与观测集合精确相等 → P1 通过，provenance 中 `authorized_variant` 标记正确。
2. 声明集合为空、观测集合含 1 个位点 → 失败于 `pdb_amino_acid_mismatch`，`details` 含完整观测集合（回归：确认已非 fail-fast）。
3. 声明 2 个、观测 1 个（其一不存在）→ 失败于 `unused_authorized_variant`。
4. 声明 1 个、观测 2 个 → 失败于 `pdb_amino_acid_mismatch`，对称差正确。
5. AFDB 侧不匹配且已声明该位点 → 仍然失败（允许表不豁免 AFDB）。
6. 合成 6 连簇 mismatch → 失败于 `mapping_error_like_mismatch`，不因数量超限而先落到 budget 码（码优先级须确定且可测）。
7. 置换 `authorized_sequence_variants` 内容 → `input_digest` 改变 → `blocked_input_drift`。
8. 声明变体位点不出现在 P7 候选池（待 P7 实现后补，本 PDR 登记为延迟验收项）。

---

## 2. D2 — 长度与 fragment 入选标准

### 2.1 问题陈述

AFDB 对超长蛋白分片存储。A6/A8 契约禁止：固定 F1 选择、最近 fragment 选择、fragment 拼接、推断 offset。因此若映射区间无法由**单一** fragment 完整覆盖，该蛋白无法进入 Dataset A。Index 9（`7kr0_A__P0DTD1`，复制酶多聚蛋白）据此正确失败。

### 2.2 决策

**维持 A6/A8 的禁止清单不变，不引入任何拼接或跨片机制。** 入选标准显式表述为覆盖条件，而非隐式副产品。

理由：拼接会引入片间坐标系与置信度语义的不一致，而 PAE 是**片内**定义的，跨片的残基对 PAE 根本不存在。若为纳入长蛋白而拼接，high-PAE long-range 这一机制的定义本身就会崩塌——恰恰是最需要它的那类样本上最不可靠。

### 2.3 入选条件（规范性）

蛋白入选当且仅当存在**恰好一个** AFDB fragment 满足：

```text
mapped_uniprot_interval ⊆ [fragment_start, fragment_end]
```

其中 `mapped_uniprot_interval` 取自 P0 冻结映射，`fragment_start/end` 取自冻结的 fragment 身份，不得由矩阵长度反推。

**长度上限的表述**：AFDB 仅对超过某一长度阈值的蛋白分片，因此本条件在实践中等价于一个隐含长度上限。**该阈值须由 V5 实测确认后填入**，草案不写入具体数字，以免把未经验证的记忆值固化进规范文件。填入格式：

```text
AFDB 分片触发阈值：【待填，V5】残基
分片片长 / 步长：【待填，V5】
由此得到的 Dataset A 实际长度上限：【待填，V5】
```

### 2.4 选择偏差与补偿（规范性）

**high-PAE long-range 机制在大型多域蛋白中最富集，而 §2.3 的覆盖条件恰好从长度上端截断。** 这是本决策已知的、无法通过工程手段消除的选择偏差。

补偿要求：

1. **筛选期主动过采样**：候选筛选阶段对 high-PAE long-range 类别设置高于目标配额的采样比例，具体倍率由 V1 给出的 `unsupported_afdb_fragment` 实际占比确定。
2. **不得延后**：不得等到 A1 抽样阶段才发现配额凑不满——那时补采需要重跑 P0–P1，成本量级完全不同。
3. **分布报告**：必须报告入选蛋白的长度分布 vs 原始 eligible pool 的长度分布，量化截断程度。
4. **结论限定**：论文中关于 high-PAE long-range 机制的结论须显式限定在长度上限以内，不得外推至超长多域蛋白。

### 2.5 失败码细化（规范性）

保留 `unsupported_afdb_fragment` 为主码，新增子原因字段 `fragment_failure_reason`：

| 子原因 | 含义 |
| --- | --- |
| `interval_crosses_fragment_boundary` | 映射区间跨越两个及以上 fragment |
| `no_fragment_covers_interval` | 无任何单一 fragment 完整覆盖 |
| `multi_fragment_protein_out_of_scope` | 蛋白本身被分片，且映射区间无法落入单片 |
| `fragment_metadata_incomplete` | fragment 身份或区间信息缺失 |

区分这四者的意义：前三者是科学入选问题，第四者是数据完整性问题，处置路径不同（后者可能通过补数据恢复）。

### 2.6 验收测试（规范性）

1. 映射区间完全落入单一 fragment → 通过。
2. 映射区间跨两片 → `unsupported_afdb_fragment` + `interval_crosses_fragment_boundary`。
3. 映射区间与所选 fragment 部分重叠 → 失败，**不得**回退到最近片或截断区间（回归 index 9 的既有结论）。
4. 单片蛋白但 metadata 缺 fragment 区间 → `fragment_metadata_incomplete`，与前三者可区分。
5. 置换 fragment 身份 → `input_digest` 改变 → drift 拦截。

---

## 3. 冻结程序

### 3.1 冻结前置

```text
[ ] V1–V5 全部关闭
[ ] §1.2 四个阈值以 V3 真实分布校准并定稿
[ ] §2.3 长度阈值以 V5 实测填入
[ ] 6 个 mismatch 蛋白逐一完成人工审核，evidence_class / evidence_note / reviewer 填写完毕
[ ] V4 确认 5 个孤立 mismatch 蛋白恢复后确实到达 p1_pairing_ok
[ ] 规模影响重新测算并同步 T0（A2 目标是否下调）
```

### 3.2 冻结后的实施顺序

```text
1. manifest schema 扩展（§1.4）+ 单测
2. p1.py 定向修改（§1.5）+ test_p1.py 更新，同一提交
3. pipeline_version → dataset-a.v2
4. round-2 census（带允许表，需下载授权，属另一任务）
5. 依 round-2 结果最终确认 A1/A2 规模
```

### 3.3 修订政策

本 PDR 冻结后，任何阈值调整均须：新增修订记录、说明触发原因、重新评估已产出数据是否失效、`pipeline_version` 升版。**不得就地修改已冻结条款。**

特别地：**不得为提高通过率而事后放宽 §1.2 的阈值。** 若冻结后发现通过率不足，正确处置是扩大候选来源或下调规模目标，不是移动判据——后者会使入选标准依赖于结果，破坏 pre-registration 的意义。

---

## 附录：条款与证据对应

| 条款 | 直接证据 | 状态 |
| --- | --- | --- |
| §1.2 分类判据 | census mismatch 分布 [6,1,1,1,1,1] 的双峰形状 | 形状已确认，阈值待 V3 校准 |
| §1.3 采纳允许表 | 6/26 命中率；方案 A 代价过大 | 已确认 |
| §1.5 P1 改为全量扫描 | 精确集合相等语义的实现前提 | 逻辑必然，需授权 |
| §1.6.2 变体位点排除出 P7 | 变体位点两侧为不同化学实体 | 逻辑必然 |
| §2.2 维持禁止拼接 | 跨片残基对 PAE 不存在 | 已确认 |
| §2.4 过采样补偿 | 长度截断与 high-PAE 机制富集方向相反 | 倍率待 V1 |
