# Dataset A 后续任务规格包（供 agent 顺序执行）v0.1

| 项 | 值 |
| --- | --- |
| 用途 | 每个 TASK 单独交给执行 agent，一次一个，不得合并 |
| 基线 | commit `a3999ec`（round-1 census + missing_pdb 回归测试） |
| 上游规范 | A8 handoff（工程契约）、实验设计方案 v0.1（科学设计）、PDR-01 v0.1（D1/D2 草案） |
| 起草日期 | 2026-08-02 |

---

## 0. 对执行 agent 的常设规则（每个 TASK 均适用）

以下规则承自 A8 handoff §12–§13，**任何 TASK 不得覆盖**：

1. 开始前记录 `git status --short`，保留全部既有脏条目，不清理、不修复、不格式化无关文件。
2. 先写聚焦测试并确认 RED，再写实现。
3. 只实现 TASK 明确规定的范围，到边界即停止，不"顺手"优化。
4. 禁用 `git add .` / `git add -A`；只 stage TASK 的文件白名单；提交前核对 `git diff --cached --name-only`。
5. 单一目的提交。**未经明确指示不得提交**——完成后报告状态并等待。
6. 不修改 A1–A8 冻结模块（`manifest.py` / `hashing.py` / `schema.py` / `lifecycle.py` / `structures.py` / `proteinmpnn.py` / `pae.py` / `p0.py` / `p1.py`），除非该 TASK 显式授权并注明依据。
7. 不下载任何 PDB/AFDB 数据，不启动任何训练，不实现 P2 及之后阶段。
8. 基线不得退化：Dataset A 362 tests、全库 682 tests、Ruff clean。
9. 若 TASK 与已冻结契约冲突，**停止并报告冲突**，不得静默修改早期模块。

---

## 1. 人工决策点（agent 不得代劳）

以下事项必须由人完成，agent 遇到时应停止并报告：

| # | 事项 | 阻断 |
| --- | --- | --- |
| H1 | PDR-01 §1.2 四个阈值的最终取值 | TASK-D 之后 |
| H2 | 六个变体的人工审核（`evidence_class` / `evidence_note` / `reviewer`） | PDR 冻结 |
| H3 | PDR-01 冻结签署 | TASK-E 及之后全部 |
| H4 | A1/A2 规模目标是否下调 | 实验设计方案修订 |
| H5 | 评估器集合与 held-out oracle 选定 | D4 / P2 |
| H6 | 下载授权（round-2 census 需要） | TASK-H |
| H7 | Gate 0 / Gate 1 / Gate 2 放行 | 各自下游 |

**agent 可以起草上述内容的候选方案与依据，但不得代为定稿。**

---

## TASK-A — V1/V2/V3 明细扩展与重跑

### 目标
扩展 round-1 census 报告字段，产出 PDR-01 §0.3 中 V1/V2/V3 所需的全部明细，用新 `run_id` 重跑并替换交付物。

### 范围
```text
src/dual_uq/dataset_a_scale/census.py      # 加字段与统计量
tests/dataset_a_scale/test_census.py       # 新断言
reports/dataset_a_census/round1_report.json  # 重跑后替换
```

### 字段要求

**每蛋白（全部 26 个可评估蛋白）**
```text
stage_reached                # 枚举：manifest_built / p0_resolved / p0_frozen / p1_resolved
mapped_length
uniprot_full_length
```

**仅 pdb_amino_acid_mismatch 命中蛋白（6 个）追加**
```text
mismatch_count
missing_pdb_count
missing_afdb_count           # 预期恒为 0；非 0 须在报告中显著标出
sequence_identity_mapped     # 分母 = mapped_length
sequence_identity_paired     # 分母 = mapped_length − missing_pdb_count
max_consecutive_run
min_pairwise_spacing         # mismatch_count ≤ 1 时为 null，不用哨兵值
mismatches: [{output_position, uniprot_position, mapping_aa, pdb_aa}]
```

**聚合区追加**
```text
p0_failure_code 直方图                    # V1
p1_failure_code 直方图（mismatch 单列）    # V2
stage_reached 直方图
每个码的「未被检查到该项的蛋白数」
censoring_note
```

`censoring_note` 必须包含：各码计数为首次失败下界；同一蛋白可能同时存在多个独立失败原因，先命中者掩盖后续原因；已确认案例 `1i1w_A__P23360` 同时存在 AA mismatch 与 `missing_pdb` 覆盖缺口（`p1.py:729-734` 的 `residue_count_mismatch` 在循环后才 raise，故被前者掩盖）。

### 实现约束
- `min_pairwise_spacing` 为 `null` 时，下游比较代码必须显式处理，不得依赖数值比较的隐式行为。
- 两个 `sequence_identity` 都记录，PDR 使用 `_paired` 版本，`_mapped` 保留备查。
- 不新增解析逻辑，继续复用 `p1._load_atom_records` / `structures` 原语。
- scratch 目录使用新 `run_id`，不覆盖旧 run。

### 验收
```bash
pytest tests/dataset_a_scale/test_census.py -q     # 先 RED
pytest tests/dataset_a_scale -q                     # 362 不退化
pytest -q                                           # 682 不退化
ruff check src/dual_uq/dataset_a_scale/census.py tests/dataset_a_scale/test_census.py
python scripts/dataset_a/census/round1_census.py     # 新 run_id 实跑
```

交叉验证（全部必须成立）：
```text
index 9   → unsupported_afdb_fragment
index 36  → missing_mapping_column
index 103 → pdb_amino_acid_mismatch, mismatch_count = 1
1i1w      → missing_pdb_count ≥ 1
```

### 交付
新版 `round1_report.json`，以及六个 mismatch 蛋白的位点表摘要（贴给人工审阅）。

### 停止点
报告完成状态，**等待人工审阅位点表后再决定是否提交**。

---

## TASK-B — V4 只读诊断

### 前置
TASK-A 完成且位点表已产出。

### 目标
在**不修改 `p1.py`**的前提下，回答：把已知 mismatch 位点从 AA 一致性检查中摘除后，这些蛋白是否真能到达 `p1_pairing_ok`，还是只是把失败推迟到下一关。

### 范围
```text
scripts/dataset_a/census/v4_variant_recovery_probe.py   # 新增，只读
tests/dataset_a_scale/test_v4_probe.py                 # 新增
```
**不得修改 `census.py` 与任何冻结模块。**

### 输入
TASK-A 产出的位点表，取全部 6 个 mismatch 蛋白（含 `1i1w`，作为阳性对照）。

### 方法
复用 `p1._load_atom_records` / `_load_mapping` / `_index_records` / `select_backbone_atoms` 等原语，逐位点走完真实规则：
- 已知 mismatch 位点：跳过 AA 一致性检查；
- 其余位点：AA 一致性、backbone 原子完整性（N/CA/C/O 各恰一个）、AFDB 侧配对（model-local 唯一、无 insertion、`uniprot_position − fragment_start + 1` 成立）、label identity——**全部按真实规则检查**；
- 循环后检查：`residue_count_mismatch` 等循环后才触发的项也必须执行。

### 输出分类（四桶）
```text
recovered                       # 其余检查全过
blocked_downstream_missing_pdb  # 因覆盖缺口失败
blocked_downstream_other        # 其他码，须记录具体码
still_mismatching               # 出现位点表未记录的新 mismatch
```

### 硬性断言
- `still_mismatching` 桶**必须为空**。非空即表示诊断脚本与 `count_all_sequence_mismatches` 对残基选择的处理存在分歧，此时**停止并报告**，不得继续得出 V4 结论。
- `1i1w` 预期落入 `blocked_downstream_missing_pdb`。若未落入，说明覆盖度检查未接通，须先修复。

### 交付
`reports/dataset_a_census/v4_recovery_probe.json` + 四桶计数摘要。

---

## TASK-C — V5 分片规则实证

### 前置
TASK-A 完成（`uniprot_full_length` 已入库）。

### 目标
从真实数据观察 AFDB 分片触发的长度区间，为 PDR-01 §2.3 提供实证依据。

### 范围
只读分析，产出报告；可作为 TASK-A 报告的一个附加小节，或独立脚本。

### 方法
对 26 个蛋白，比对 `uniprot_full_length`、所选 fragment 的区间、fragment 编号，观察：
- 单片蛋白的最大全长；
- 多片蛋白的最小全长；
- 若两者之间存在明确间隙，即为分片触发阈值的实证区间。

### 输出约束
**若 26 个蛋白未覆盖阈值附近区间，报告必须如实写明"无法从本批数据确定阈值"，不得填入任何未经实证的数字。** PDR §2.3 的留白优于错误的具体值。

---

## TASK-D — PDR-01 数值填充（草稿）

### 前置
TASK-A / B / C 全部完成。

### 目标
把 V1–V5 的实测结果填入 PDR-01，产出 v0.2 草稿。

### 允许 agent 做的
- 填入 V1/V2 的码分布、V3 的位点表与统计量、V4 的四桶结果、V5 的实证区间；
- 按 §0.1 更新证据基础表与截尾声明；
- 按 V4 结果更新 §1.7 的收益估计与 §0.2 的规模测算；
- 补 §1.9 第 9 项验收测试：声明集合与观测集合相等、但循环后存在 `residue_count_mismatch` 的蛋白，必须失败于后者而非通过；
- 在 §1.2 处**列出**阈值的候选取值及各自依据。

### 禁止 agent 做的
- **不得为 §1.2 的四个阈值定稿**（H1）；
- 不得填写 `evidence_class` / `evidence_note` / `reviewer`（H2）；
- 不得宣告 PDR 冻结（H3）；
- 不得修改 A1/A2 规模目标（H4）。

### §1.2 的分支说明（必须在草稿中显式呈现）
依 `1i1w` 六个位点的实际形态，判据设计分三种走向：
```text
6 个连续              → max_consecutive_run = 1 足够，判据保持简单
分散成两三簇          → 形状判据仍可用，阈值需重新斟酌
完全分散且两两间距 ≥5 → 两条形状判据全部失效，只剩 mismatch_count ≤ 3
                        此时"孤立 vs 成簇"的双峰论证不成立，
                        须改以 sequence_identity 为主判据并重写 §1.2 叙事
```

---

## TASK-E — manifest schema 扩展

### 前置
**PDR-01 已冻结（H3）。未冻结不得开始。**

### 范围
```text
src/dual_uq/dataset_a_scale/manifest.py    # 授权修改，依据：PDR-01 §1.4
tests/dataset_a_scale/test_manifest_schema.py
```

### 内容
新增可选字段 `authorized_sequence_variants`（数组，缺省空），schema 规则见 PDR-01 §1.4：进入 `input_digest`、`evidence_class` 枚举受限、`evidence_note` 与 `reviewer` 必填、按 `uniprot_position` 升序、重复 position 为 schema 错误、不使用 `output_position` 作键。

### 验收
含"置换该字段 → `input_digest` 改变 → `blocked_input_drift`"的测试。

---

## TASK-F — `p1.py` 定向修改

### 前置
TASK-E 完成。**授权依据：PDR-01 §1.5（经收窄口径修订）。**

### 精确范围
`_pair_residues` 循环中，**仅** mapping↔PDB 的 canonical AA 一致性判断点由 raise 改为记入观测集合并 continue。

**明确不改动的**（判定时机与严格度分毫不变，仍 fail-fast）：
```text
backbone 原子完整性（N/CA/C/O 各恰一个选定原子）
AFDB 侧配对（model-local 唯一、无 insertion、坐标变换成立）
label identity 一致性
missing_pdb（映射位点上 PDB 侧无残基）
```

循环结束后：观测集合与 manifest 声明集合做**精确集合相等**比较，按 PDR-01 §1.8 的五个码分派失败。

### 不变量
- 声明集合为空且观测集合非空 → 仍失败，只是 details 从首位点变为全部位点；
- AFDB 侧 AA 不一致**不受允许表豁免**；
- 不新增 force flag、环境变量开关或代码内置默认允许位点。

### 连带
`tests/dataset_a_scale/test_p1.py` 中断言首位点错误载荷的用例同步更新，**必须同一提交**。新增 PDR §1.9 第 9 项用例（循环后 `residue_count_mismatch` 优先）。

### 注意
AA 检查改为 continue 后，**原先被 fail-fast 挡住的循环后段检查将第一次真正执行**（如 `p1.py:729-734`）。实现时需预期部分蛋白的失败码发生迁移，这是正确行为，不是回归。

---

## TASK-G — `pipeline_version` 升版

### 前置
TASK-F 完成。

### 内容
`dataset-a.v1` → `dataset-a.v2`，使 `config_digest` 与 v1 时代产物自然区隔。检查全仓库对版本字符串的引用，确保无硬编码遗漏。

---

## TASK-H — Round-2 census

### 前置
TASK-G 完成 + **下载授权（H6）**。

### 内容
带允许表对全候选池（213）跑 P0→P1，产出最终入选表与 attrition 表。后者直接进论文 dataset construction 章节。

### 交付
入选表、attrition 表、按机制分层的入选计数、长度分布 vs 原始 eligible pool 的对比（PDR-01 §2.4.3 要求）。

### 停止点
产出后停止。**A1/A2 规模的最终确认属 H4，不由本 TASK 决定。**

---

## 2. 执行顺序与并行

```text
串行主链：
TASK-A → TASK-B → TASK-C → TASK-D → [H1 H2 H3] → TASK-E → TASK-F → TASK-G → [H6] → TASK-H → [H4 H7]

可与主链并行（纯设计，不碰代码）：
· D3/D4/D5 条款草案（PDR-02）
· 评估器 registry 起草（H5 的输入）
```

**评估器 registry 应立刻开始**，不要等 census 收尾：它是 H3 假设的唯一承载，且被 D4 卡着时间点（候选池冻结前必须定，因为评估器数量对打分成本线性放大）。

---

## 3. 每个 TASK 的完成报告格式

```text
TASK 编号与名称
基线是否退化（三个测试数字 + Ruff）
交付物路径
关键数字/发现
git status --short 的变化（新增/修改路径清单）
是否触及人工决策点（若是，停止并说明）
建议的下一步
```

**不得在报告中夹带未经授权的改动。**
