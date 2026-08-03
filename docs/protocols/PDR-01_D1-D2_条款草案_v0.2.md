# PDR-01：D1 序列不等价处置 / D2 长度与 fragment 入选标准（条款草案 v0.2）

| 项 | 值 |
| --- | --- |
| 文档性质 | V1–V5/C-R 证据填充后的协议决策记录草稿 |
| 状态 | **DRAFT / NOT FROZEN** |
| 覆盖决策 | D1、D2（实验设计方案 v0.1 §3） |
| 证据基础 | Round-1 V1–V3、V4 recovery probe、V5 fragment empirical probe、C-R canonical length audit |
| 本版边界 | 呈现实测证据并同步已确认的 H1；不替代 H2–H4 人工决策 |
| 起草日期 | 2026-08-03 |

> 本版是 evidence-population draft，不是 protocol-decision。A1–A8 的冻结契约、
> pipeline version 和任何实现均未因本文改变。

---

## 0. 证据基础、规模含义与人工门禁

### 0.1 V1/V2：Round-1 census

Round-1 共发现 28 个本地 pair；其中 2 个早期 pilot pair 因
`no_identity_source` 跳过，26 个具有候选池身份并进入评估。

| 结果 | 数量 |
| --- | ---: |
| P0 failure | 7 |
| P1 failure | 11 |
| `p1_pairing_ok` | 8 |
| 可评估合计 | 26 |
| 严格 admission rate | 8/26 = 30.8% |

V1 的 P0 首次失败分布：

| P0 failure code | 数量 |
| --- | ---: |
| `missing_mapping_column` | 4 |
| `pair_qc_not_acceptable` | 1 |
| `unsupported_afdb_fragment` | 1 |
| `ambiguous_auth_residue_mapping` | 1 |
| 合计 | 7 |

V2 的 P1 首次失败分布：

| P1 failure code | 数量 |
| --- | ---: |
| `pdb_amino_acid_mismatch` | 6 |
| `residue_count_mismatch` | 2 |
| `invalid_atom_record` | 1 |
| `invalid_atom_residue_identity` | 1 |
| `missing_backbone_atom` | 1 |
| 合计 | 11 |

实测 `stage_reached` 分布：

| 最深到达阶段 | 数量 |
| --- | ---: |
| `manifest_built` | 7 |
| `p0_frozen` | 11 |
| `p1_resolved` | 8 |

在 26 个可评估蛋白中，未到达各阶段的数量为：

| 阶段 | 未被检查到该阶段的蛋白数 |
| --- | ---: |
| `manifest_built` | 0 |
| `p0_resolved` | 7 |
| `p0_frozen` | 7 |
| `p1_resolved` | 18 |

**Censoring statement**：failure counts are first-failure lower bounds;
earlier failures can mask independent downstream failures。已确认案例
`1i1w_A__P23360`（Index 111）同时存在 PDB AA mismatch 和一个
missing-PDB coverage gap；fail-fast 顺序使报告主码停在
`pdb_amino_acid_mismatch`，而后续 `residue_count_mismatch` 被遮蔽。

### 0.2 Evidence-based scale estimate

当前严格 admission 为 8/26。V4 在假定 mismatch 声明集合精确匹配观测集合的
探针中实测恢复 4 个，因此在 H2 尚未完成人工审核时，可报告的证据范围为：

| 情景 | 26 蛋白中的数量 | 比例 | 对 213 候选的描述性外推 |
| --- | ---: | ---: | ---: |
| 当前严格契约 | 8 | 30.8% | 约 66 |
| 加上 V4 实测可恢复项（条件性上界） | 12 | 46.2% | 约 98 |

该 **66–98** 范围只是 evidence-based descriptive estimate。样本是已落盘、可能
富集高质量候选的子集，不是 213 候选的随机抽样；不得把该范围解释为总体置信区间。
4 个 V4 recovered 也只有在 H2 人工证据审核完成后才可能转化为 admission。

**H4 — HUMAN SCALE DECISION**：A1 是否保持约 64、A2 是否调整，不由本文决定。

### 0.3 V1–V5/C-R 完成状态与剩余人工门禁

| 项 | 当前证据状态 | 本文处置 |
| --- | --- | --- |
| V1 | P0 分布已实测 | 已填入 §0.1 |
| V2 | P1 分布已实测 | 已填入 §0.1 |
| V3 | 6 个 mismatch 蛋白完整扫描已实测 | 已填入 §1.2–§1.3 |
| V4 | 四桶 recovery probe 已实测 | 已填入 §1.4、§1.7 |
| V5 | 26 蛋白 fragment probe 已实测 | 已填入 §2.3 |
| C-R | canonical full-length provenance 已修正并审计 | 已填入 §2.3、§2.7 |

H1 已同步，剩余人工门禁为：

- **H1 — DECIDED**：`mismatch_count ≤ 3`、`sequence_identity_paired ≥ 0.99`；
  run/spacing 仅作诊断；
- **H2 — HUMAN REVIEW REQUIRED**：Index 28、103、22、25的人工变体证据字段；
- **H3 — HUMAN SIGN-OFF REQUIRED**：PDR-01 的签署与状态变更；
- **H4 — HUMAN SCALE DECISION**：A1/A2 的规模。

---

## 1. D1 — 序列不等价的处置

### 1.1 问题陈述

P1 当前要求每个 output position 上 mapping/PDB/AFDB 的 canonical AA 一致。
`pdb_amino_acid_mismatch` 既可能来自真实结构中的序列差异，也可能来自错误映射；
仅凭 mutation 形态不能替代人工来源审核。AFDB 与 mapping 的差异属于另一类一致性
问题，不可由 PDB authorized variant 声明豁免。

### 1.2 V3 证据与候选判据

判据讨论优先使用 `sequence_identity_paired`，因为它以实际具有 PDB 配对残基的数量
为分母。`sequence_identity_mapped` 同时保留用于审计 missing-PDB 的影响。

| Index / pair | mapped | UniProt full | mismatch | missing PDB | missing AFDB | identity mapped | identity paired | max run | min spacing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 111 / `1i1w_A__P23360` | 303 | 329 | 6 | 1 | 0 | 0.980198 | 0.980132 | 1 | 14 |
| 28 / `1muw_A__P15587` | 386 | 387 | 1 | 0 | 0 | 0.997409 | 0.997409 | 1 | null |
| 103 / `5avd_A__P00772` | 240 | 266 | 1 | 0 | 0 | 0.995833 | 0.995833 | 1 | null |
| 22 / `7af2_AAA__P29768` | 379 | 382 | 1 | 0 | 0 | 0.997361 | 0.997361 | 1 | null |
| 25 / `7bbx_A__Q5F6E9` | 351 | 351 | 1 | 0 | 0 | 0.997151 | 0.997151 | 1 | null |
| 101 / `8pb5_A__P0DPA9` | 309 | 309 | 1 | 0 | 0 | 0.996764 | 0.996764 | 1 | null |

`null` spacing 表示只有一个 mismatch，因而不存在成对间距；它不是零。

完整 mismatch positions（`output / UniProt: mapping→PDB`）：

- Index 111：`39/65: D→N`、`193/219: Q→K`、`217/243: G→S`、
  `245/271: P→S`、`259/285: Q→S`、`300/326: D→N`；
- Index 28：`175/176: T→I`；
- Index 103：`66/92: D→N`；
- Index 22：`59/62: D→G`；
- Index 25：`8/8: K→A`；
- Index 101：`102/102: R→K`。

6-mismatch 蛋白 Index 111 的位点不是连续段，也不是紧密簇：`max_consecutive_run=1`，
`min_pairwise_spacing=14`，并分散在 output positions 39–300。该真实形态削弱了仅用
run/spacing 区分 variant-like 与 mapping-error-like 的能力；数量与 paired identity
在当前六例中提供了更强的分离信息。

#### Automatic review-eligibility screen（H1）

进入人工 variant review 必须同时满足以下两个 hard gates：

```text
mismatch_count ≤ 3
sequence_identity_paired ≥ 0.99
```

- `mismatch_count ≤ 3` 是位于当前观测 burden 1 与 6 之间的保守 variant budget，
  不是统计估计得到的 optimal cutoff；
- `sequence_identity_paired ≥ 0.99` 是按实际配对长度归一化的 mapping-sanity
  safeguard，不是经过验证的 classifier；
- 通过两个 gates 只表示 **eligible for human variant review**，不等于
  `authorized variant`，也不等于 Dataset A admission；
- `max_consecutive_run` 与 `min_pairwise_spacing` 继续记录，但仅用于 audit 和
  morphology diagnostics，不得决定 review eligibility 或 admission。

五个 singleton 的 paired identity 为 0.995833–0.997409，均通过两个hard gates。
Index 111 的 `mismatch_count=6`、`sequence_identity_paired=0.9801324503311258`，
同时违反两个hard gates；尽管其 `max_consecutive_run=1`、
`min_pairwise_spacing=14`，仍必须被拒绝。这也直接说明run/spacing对当前
high-burden negative anchor没有判别力。

**H1 — DECIDED**：以上两个hard gates及run/spacing的diagnostic-only地位已由人工确认。

### 1.4 V4 recovery probe

在不修改冻结 P1 的独立探针中，将每个蛋白的已观测 mapping↔PDB mismatch 集合作为
假定声明集合，得到：

| V4 bucket | 数量 | 蛋白 |
| --- | ---: | --- |
| `recovered` | 4 | Index 28、103、22、25 |
| `blocked_downstream_missing_pdb` | 1 | Index 111 |
| `blocked_downstream_other` | 1 | Index 101 |
| `still_mismatching` | 0 | — |

V4的实测结论为：

> **4 empirically recovered under the probe's assumed declaration set.**

Index 101 `8pb5_A__P0DPA9` 在 UniProt/output position 102 上为 mapping R、PDB K，
同时 AFDB 也是 K。mapping R vs AFDB K 是独立 downstream consistency failure，
不属于可由 PDB `authorized_sequence_variants` 恢复的案例。

### 1.5 声明式允许表：待人工完成的候选机制

若 H3 后续授权实施声明式机制，观测 mismatch 集合与人工声明集合必须精确相等；
AFDB↔mapping mismatch 不得豁免。声明内容应进入 canonical input digest，任何变化均应
触发 input drift。本文不修改 manifest schema、P1 或 pipeline version。

全部6个 mismatch proteins 均保留审计记录，但只有通过H1且不存在AFDB-side
mismatch的4个蛋白进入authorization review：

| Index | pair | PDB mismatch positions | V4 technical outcome | authorization review |
| ---: | --- | --- | --- | --- |
| 111 | `1i1w_A__P23360` | 65, 219, 243, 271, 285, 326 | downstream missing PDB | 排除：count和identity gate均失败 |
| 28 | `1muw_A__P15587` | 176 | recovered | **H2 review candidate** |
| 103 | `5avd_A__P00772` | 92 | recovered | **H2 review candidate** |
| 22 | `7af2_AAA__P29768` | 62 | recovered | **H2 review candidate** |
| 25 | `7bbx_A__Q5F6E9` | 8 | recovered | **H2 review candidate** |
| 101 | `8pb5_A__P0DPA9` | 102 | downstream AFDB mismatch | 排除：不具备authorized PDB variant资格 |

**H2 — HUMAN REVIEW REQUIRED**：只对Index 28、103、22、25审核；其
`evidence_class`、`evidence_note`、`reviewer`、`review_date` 均保持未填写。
Index 111和101不得填写这些字段。不能依据位点数量、间距或recovery bucket推断
engineered construct、point mutant、isoform difference或其他来源类别。

### 1.6 候选实现契约及下游义务

以下内容只在 H2/H3 完成后才可能进入实施任务：

1. P1 全量收集 mapping↔PDB mismatch 集合；
2. 观测集合与人工声明集合精确比较，多一个或少一个都失败；
3. AFDB 必须继续与 mapping 一致；
4. provenance 明确标记 authorized variant；
5. authorized variant 位点不得进入 P7 候选位点池；
6. P6/P9 对含变体与不含变体蛋白进行敏感性分层；
7. 不新增 force、offset 推断或内置允许位点。

### 1.7 Evidence-based admission impact and limitations

V4 证明 4 个单 mismatch 案例在假定声明集合下可越过 mismatch 检查并到达
`p1_pairing_ok`；Index 111 和101分别被 downstream missing-PDB 与 AFDB mismatch
阻断。因此规模收益应使用 **4**，而不是 v0.1 的 5 个上界。

在当前 26 蛋白上，严格 admission 8 个；条件性加入4个为12个，对应 §0.2 的
30.8%–46.2% 描述范围。该范围不决定 A1/A2，且不消除以下局限：

- 声明变体位点两侧可能是不同化学实体；
- zero-mismatch admission 可能排除合法的engineered/experimental constructs，
  从而产生选择偏差；
- authorized variants可改善样本代表性，但会引入construct-related confounding；
- 变体邻域对局部主链几何的影响尚未量化；它只登记为未来sensitivity question，
  本文不新增 `±3` 邻域排除规则；
- 26 蛋白不是随机样本；
- 4 个 recovered 尚无 H2 人工来源证据。

**H4 — HUMAN SCALE DECISION** 保持开放。

### 1.8 H1 failure-code semantics

Automatic review-eligibility screen至少保留：

| failure code | 触发条件 |
| --- | --- |
| `sequence_variant_budget_exceeded` | `mismatch_count > 3` |
| `sequence_identity_below_threshold` | `sequence_identity_paired < 0.99` |

`max_consecutive_run`和`min_pairwise_spacing`不得触发
`mapping_error_like_mismatch`或任何其他hard failure；两者只写入diagnostic details。

若同一蛋白同时违反多个predicate，确定性主码优先级为：

```text
1. sequence_variant_budget_exceeded
2. sequence_identity_below_threshold
```

主`failure_code`使用最先命中的码，`details.failed_predicates`保存全部失败predicate及
实测值。AFDB↔mapping mismatch和downstream consistency failure仍是不可豁免的独立
失败，不得被上述review-eligibility主码吞掉。

### 1.9 验收测试条款草稿

若 H3 后进入实现，验收范围至少包括：

1. singleton且`sequence_identity_paired > 0.99`时，结果为eligible for human
   review，但不得自动成为authorized variant；
2. 6个分散mismatch、`max_consecutive_run=1`、`min_pairwise_spacing=14`、
   `sequence_identity_paired≈0.98013`时，必须被count/identity gates拒绝；
   run/spacing不得使其通过；
3. 同时违反count和identity时，主码为`sequence_variant_budget_exceeded`，并在
   `details.failed_predicates`记录两个失败predicate；
4. 声明集合与观测集合精确相等时，只有在全部downstream checks也通过后P1才通过，
   provenance标记正确；
5. 空声明、非空观测时失败，并返回完整观测集合；
6. 声明多于或少于观测时失败并报告集合差；
7. AFDB↔mapping mismatch即使位点有authorized PDB variant也必须失败；
8. 变更声明集合必须改变input digest并触发input drift；
9. authorized variant位点不得进入P7候选池；
10. **当 authorized mismatch 的声明集合与观测集合精确相等，但循环结束后存在
   `residue_count_mismatch` 时，P1 必须失败于该 downstream consistency check，
   不得错误通过。**

第10项由Index 111的V3/V4联合证据直接支持；本文只登记条款，不修改P1或测试代码。

---

## 2. D2 — 长度与 AFDB fragment 入选标准

### 2.1 问题陈述

AFDB 对部分超长蛋白提供多个 fragment。跨 fragment 拼接会破坏单一坐标与片内 PAE
语义，因此 Dataset A 只接受单个 fragment 完整覆盖冻结 mapped UniProt interval 的情况。

### 2.2 正式覆盖原则

维持 A6/A8 已有原则：

```text
exactly one AFDB fragment must fully cover mapped interval
```

并维持禁止清单：

```text
no F1 fallback
no nearest fragment
no stitching
no offset inference
```

覆盖判断必须使用显式 UniProt interval 和显式 fragment metadata；不得由矩阵长度、
model residue count、mapped length 或 P1 paired length反推。

### 2.3 V5 fragment empirical evidence

| 指标 | 实测值 |
| --- | ---: |
| cohort | 26 |
| single / multi / unknown | 25 / 1 / 0 |
| max single full length | 546 |
| min multi full length | 7095 |
| empirical transition bracket | `(546, 7095]` |
| `threshold_identifiable` | `false` |
| observed fragment length | `null` |
| observed step | `null` |

`(546, 7095]` 只是当前非随机26蛋白样本中的 **empirical transition bracket**，
不等于 AFDB fragmentation threshold。只有一个 multi-fragment 蛋白，边界附近证据
不足；其 fragment intervals 的长度和起点步长也不统一，因此不能从当前数据识别统一
fragment length 或 step。

以下三项保持 **unresolved**：

```text
exact AFDB fragmentation threshold
fragment length
fragment step
```

不得把546、7095或区间宽度写成其中任一参数。

### 2.4 Index 9 regression and C-R correction

Index 9 `7kr0_A__P0DTD1`：

| 字段 | 值 |
| --- | --- |
| canonical `uniprot_full_length` | **7095** |
| canonical source | `screening_pool_preflight.canonical_uniprot_length` |
| C-R 纠正前 pair-QC值 | 126（selected fragment/model length，不是 full length） |
| mapped interval | 1024–1192 |
| selected fragment interval | 1368–1493 |
| coverage result | `unsupported_afdb_fragment` |

C-R 对26蛋白的审计为 matched 25、corrected 1、missing 0、conflict 0；只有 Index 9
被修正。126仍可作为 selected fragment length 出现在 fragment evidence 中，但不得再被
称为该蛋白的 UniProt full length。

### 2.5 选择偏差与补偿候选

单-fragment完整覆盖原则会从长度上端排除一部分大型、多域蛋白，而 high-PAE
long-range 机制可能在这些蛋白中富集。候选筛选应保留完整 attrition 和长度分布，
并在人工规模/抽样决策中评估是否需要对该机制过采样。V1 中仅观测到1个
`unsupported_afdb_fragment` 首次失败，且属于截尾下界；当前证据不足以确定过采样倍率。

### 2.6 候选失败语义与验收

保持主码 `unsupported_afdb_fragment`。未来细分原因可区分 interval 跨界、无覆盖片、
multi-fragment out-of-scope 和 metadata 不完整，但任何细分都不得产生选择 fallback。

验收条款草稿：

1. mapped interval 完全落入恰好一个 fragment时通过；
2. 跨两个或更多 fragment时失败；
3. 与某片仅部分重叠时失败，不截断、不选最近片；
4. metadata 缺失时结构化失败；
5. fragment身份变化必须触发 input drift；
6. Index 9 保持 `unsupported_afdb_fragment`。

### 2.7 D2 unresolved boundary

当前证据支持覆盖原则及 Index 9 回归，但不支持一个具体的全局长度 threshold、统一
fragment length或step。§2.3 的三项 unresolved 状态必须由新的、预先定义的证据任务
或人工协议判断处理；TASK-D 不补数字。

---

## 3. 人工决策与后续实施边界

### 3.1 H1–H4

| 门禁 | 尚需人工完成 | TASK-D 状态 |
| --- | --- | --- |
| H1 | count/paired-identity hard gates；run/spacing diagnostic only | DECIDED |
| H2 | Index 28、103、22、25的 `evidence_class`、`evidence_note`、`reviewer`、`review_date` | OPEN |
| H3 | PDR-01 sign-off和状态变更 | OPEN |
| H4 | A1/A2规模 | OPEN |

### 3.2 本草稿不授权的操作

- 不修改 manifest schema、P1、pipeline version或A1–A8模块；
- 不启动 Round-2 census；
- 不启动 P2；
- 不启动 TASK-E/F/G/H；
- 不将 V4 technical recovery 等同于 H2 人工证据；
- 不把 V5 bracket 当作 AFDB threshold。

---

## 附录 A：证据追溯

| 证据 | 文件 | 本文使用位置 |
| --- | --- | --- |
| V1/V2/V3/C-R | `reports/dataset_a_census/round1_report.json` | §0.1、§1.2–§1.3、§2.4 |
| V4 | `reports/dataset_a_census/v4_recovery_probe.json` | §1.4、§1.7、§1.9 |
| V5 | `reports/dataset_a_census/v5_fragment_empirical_probe.json` | §2.3–§2.7 |
| 上游设计边界 | `docs/design/Dual-UQ_Dataset-A_实验设计方案_v0.1.md` | 全文 |
| 上一版条款 | `docs/archive/PDR-01_D1-D2_条款草案_v0.1.md` | 结构与待验证项基线 |

## 附录 B：v0.2 evidence-population 摘要

- V1/V2 首次失败分布和 stage censoring 已填入；
- V3 六蛋白完整 mismatch evidence 已填入；
- V4 实测恢复4个，另有两个downstream blockers；
- V5 仅支持 `(546, 7095]` empirical bracket，threshold/length/step未识别；
- C-R 将 Index 9 canonical full length 从误用的126修正为7095；
- H1已同步；H2/H3/H4保持开放。

**H1 decision has been synchronized into the draft. PDR-01 remains DRAFT and NOT FROZEN.**
**H2 human evidence review is the next gate; H3 still requires human sign-off.**
