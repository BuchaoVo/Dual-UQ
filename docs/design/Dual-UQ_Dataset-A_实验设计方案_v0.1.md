# Dual-UQ Dataset A 实验设计方案（协作草案 v0.1）

| 项 | 值 |
| --- | --- |
| 文档性质 | 实验设计规格（pre-registration 草案），面向合作者 |
| 状态 | 草案，§3 的协议决策未冻结前不得启动 P2 及之后任何阶段 |
| 上游依据 | Dual-UQ Dataset A Pipeline Handoff after Task A8（frozen HEAD `8251100`） |
| 起草日期 | 2026-08-01 |
| 待补 | §3 全部决策项、§9 分工人选与时间点 |

> **阅读顺序建议**：新合作者先读 §1–§2 建立科学目标，再读 §3 了解哪些还没定，再按自己承担的 track 读 §5–§7。工程契约（identity、hashing、immutability、structured failure）以 A8 handoff 为准，本文不重复。

---

## 1. 研究问题与假设

### 1.1 核心命题

一个在单一、确定骨架上得分最高的蛋白序列，是否仍然是在结构与评估不确定性下最可靠的设计？

传统 inverse-folding 隐含假设 $B=B^*$，即输入骨架就是唯一、无误差的真实结构状态。实际设计场景中输入骨架来自预测模型、低分辨率实验结构、生成式骨架或不同构象状态，此时 $\widehat B\neq B^*$。本研究把研究对象从

$$\text{backbone}\rightarrow\text{sequence}$$

扩展为

$$(\widehat B, C_B, \mathcal M, w, \rho)\rightarrow\text{sequence distribution and decision}$$

其中 $C_B$ 为结构可信度证据（pLDDT / PAE / 实验质量），$\mathcal M$ 为属性评估器集合，$w$ 为多目标偏好，$\rho$ 为风险容忍度。

### 1.2 四个可检验假设

| 编号 | 假设 | 承载阶段 | 主要终点 |
| --- | --- | --- | --- |
| **H1** | 结构条件改变会改变序列设计决策，且效应超过同等突变量的中性扰动 | P2–P4 | protein 级 source-conditioned effect 减 matched-control effect |
| **H2** | 不同结构不确定性机制产生**不同**效应，而非统一的"结构质量差" | P6 分层 | 机制因子在混合模型中的固定效应 |
| **H3** | 结构不确定性与评估器不确定性存在**非可加交互** $\gamma^{B\times M}\neq 0$ | P3′–P3″ | 交互项的似然比检验 + 置换检验 |
| **H4** | 局部序列决策翻转具有因果性，可通过局部结构干预恢复 | P7–P8 | out-of-selection 的 protein 级 $I_{\text{DID}}$ |

**H3 是本研究区别于"分别加两个 confidence feature"的唯一贡献点。** 现有 P0–P9 阶段图不含评估器维度，必须按 §6 新增 track，否则 H3 无法被回答。

### 1.3 明确不属于本研究的范围

- Dataset A 不是 inverse-folding 训练集；大规模训练仍使用 CATH 等标准数据集。Dataset A 只用于机制验证、风险评估、方法消融、局部解释、受控对比与可靠性测试。
- 训练 GearNet / ProteinMPNN / Joint-UQ 模型不在本文档范围内，需另行立项。
- PDB 与 AFDB 的不一致**不等价于 AlphaFold 错误**（见 §4.4）。

---

## 2. 实验设计骨架

### 2.1 设计类型

**同一蛋白身份内的配对反事实设计。** 通过同一 UniProt 的 PDB–AFDB 配对，保持：

```text
相同蛋白身份 / 相同天然序列 / 相同映射残基 / 相同候选序列池
```

只改变：

```text
结构条件 s ∈ {PDB, AFDB}
```

从而把蛋白身份、家族、长度、序列组成造成的混杂从效应估计中移除。

### 2.2 因子与推断单位

| 因子 | 水平 | 说明 |
| --- | --- | --- |
| $s$ 结构条件 | PDB / AFDB | 蛋白内配对 |
| $m$ 评估器 | observed 评估器集合 + held-out oracle | 见 §6 |
| 机制 | easy control / low-confidence local / high-PAE long-range / high-confidence state disagreement | 蛋白间分层 |
| tier | Tier-1（P0–P4）/ Tier-2（P0–P9） | 成本分层 |

**最终统计推断单位是 protein，不是 site，也不是 candidate。** 30 个蛋白各 9 个位点**不得**表述为 $n=270$ 独立样本。site 级与 candidate 级量一律先汇总到 protein 级再做检验。

### 2.3 效应分解模型

对属性 $k$，候选序列在结构条件 $s$ 与评估器 $m$ 下的评分：

$$r_k^{(m,s)} = \mu_k + \alpha_s^{B} + \beta_m^{M} + \gamma_{m,s}^{B\times M} + \epsilon$$

以 protein 为随机效应的混合模型拟合，H3 检验 $\gamma_{m,s}^{B\times M}\neq 0$。

---

## 3. 待冻结的协议决策（阻塞项）

**以下五项未定稿前不得启动 P2。** 每项需在协议决策记录（PDR）中写明取值、判据与验收测试，并进入 manifest schema，使其被 `input_digest` 覆盖。

### D1 序列不等价的处置

**背景**：P1 要求每个 output position 上 mapping / PDB / AFDB 三方 canonical AA 完全一致。Index 103（`5avd_A__P00772`）在 UniProt 92 处 mapping 为 D、PDB 为 N，整体一致性约 0.9958，属于典型工程化构建体或点突变，不是 mapping 错误。

**判据**：mapping 错误的签名是 mismatch 数量大、位置成簇、整体一致性显著偏低；真实序列变异的签名是孤立 1–2 个位点、一致性 ≥ 0.99。

**候选方案**：

| 方案 | 说明 | 风险 |
| --- | --- | --- |
| A 数据集层面剔除 | 直接排除命中蛋白 | 系统性剔除被充分研究的工程化结构；若命中率高则不可行 |
| B 修改输入 | 换 PDB entry 或 UniProt isoform | 仅少数情况可行，引入新 provenance 分支 |
| C 声明式变体允许表 | manifest 增加 `authorized_sequence_variants`，P1 要求**观测 mismatch 集合与声明集合精确相等**，多一个少一个均 fail | 需人工审核依据；需配全局阈值 |

**倾向**：方案 C，附加全局 gate（声明变体数 ≤ 3 且整体一致性 ≥ 0.99，超出返回独立 failure code `sequence_identity_below_threshold`）。容忍度作为被 hash 的**数据**存在，代码中不出现任何静默路径。

**依赖**：本决策依赖 §5.1 第一轮 census 的命中率分布，不得在分布未知时拍板。

**注意**：paired backbone 仅含 N/CA/C/O 四原子，残基身份在 backbone 中是元数据而非几何。严格等价检查的科学目的是 **mapping sanity check**，不是要求两侧必须是同一变体。

### D2 长度与 fragment 入选标准

AFDB 对超长蛋白分片存储，禁止拼接、禁止最近片选择、禁止推断 offset（A6/A8 契约）。因此 Dataset A 实际限制在"单 fragment 可完整覆盖映射区间"的蛋白。Index 9（`7kr0_A__P0DTD1`，复制酶多聚蛋白）据此正确失败为 `unsupported_afdb_fragment`。

**需要冻结**：明确的长度上限表述、以及对由此产生的选择偏差的补偿方案。

**风险提示**：high-PAE long-range 机制在大型多域蛋白中最富集，而长度上限恰好从这一端截断。**必须在候选筛选期对该机制过采样**，不能等到 A1 抽样时才发现配额凑不满。

### D3 residue index 与 chain break 的传递

P1 输出使用连续 `output_position 1..L`。由于 PDB 缺失密度，mapped set 在 UniProt 上通常不连续，连续重编号会使跨 gap 的两个残基在输出编号上相邻，凭空制造肽键邻接关系。

两侧共用同一编号，因此不产生 source 间的差异性混杂，但有两个后果：

1. 绝对分数被扭曲，影响 P9 的跨蛋白汇总；
2. **PDB 缺失密度的位置系统性地就是 AFDB 低 pLDDT 的位置**（均为无序区），即 gap 边界天然富集 low-confidence local 机制，在此处伪造邻接等于在最关心的机制类别上引入人为几何特征。

**需要冻结**：P2 构造 ProteinMPNN 输入时按真实 UniProt 间隔生成 residue index 并保留跳变；segment 结构（segment 数、最长 gap、gap 边界位点）写入 P1/P6 产物；gap-adjacent 位点排除出 P7 候选池（gap 距离属于 mapping cleanliness，P7 可见，不违反盲法）。

### D4 候选采样规格

当前草案为每 backbone 4 条、temperature 0.1、backbone noise 0。该设置下 8 条候选之间的 Hamming 距离可能仅个位数，彼此近乎重复，导致：

- rank flip rate、top-$k$ overlap、Kendall/Spearman 在 $n=8$ 且高度相关的候选池上统计功效不足；
- P4 的 matched control 跨度到 Hamming 32，比真实候选池内部多样性大一个量级，两者不在同一尺度，"超过普通扰动"的比较难以解释。

**需要冻结**：温度阶梯（建议 0.1 / 0.2 / 0.3 各若干条）、每档条数、temperature 作为候选属性记录、control 的 Hamming 网格对齐候选真实距离分布的方式。所有随机性仍由 A2 SHA-derived seed 决定。

**另需冻结候选池的 source 中性**：候选必须两个条件各生成后取并集（外加天然序列，可选同源序列），每条候选记录其生成来源，报告时按来源分层。若候选主要来自 PDB 骨架，rank flip 的绝对值不可解释。

### D5 选择与估计的样本分裂

P7 selector 可读 P5 的 $G$（source probability contrast），而 P8 要估计的正是同一批 ProteinMPNN 概率运行所反映的 source × sequence decision 交互。在效应变量上选点再估效应会产生 winner's curse，$I_{\text{DID}}$ 系统性偏大，**bootstrap CI 无法修正**（重采样的是同一批已被选中的位点）。

**需要冻结**：P5 的 16 次 repeat 拆为 selection half（前 8）与 estimation half（后 8），或使用独立 seed 重跑。P7 的 hash 与盲法规则不变，但 P8 的估计变为 out-of-selection。

---

## 4. 数据集构建规范

### 4.1 候选来源与分层

```text
原始 eligible pool
      ↓ diversity / quality screening
primary pool + replacement pool
      ↓ preflight
进入正式 pipeline 的 proteins
```

历史规模：213 eligible candidates → candidate pool 36 → replacement pool 20 → replacement preflight pass 12。

**注意**：A0-dev v0.9 已冻结为 development dataset，其最终 12-protein panel 为空（诊断与类别 gate 未通过）。它可支持方法开发与探索性机制分析，**不得表述为最终 held-out evaluation set**，也不存在可回退的中间结果——所有论文级观测必须由新 pipeline 从头产出。

### 4.2 规模规划

| 版本 | 规模 | 用途 |
| --- | --- | --- |
| A0-dev v0.9 | 已冻结 | proof-of-concept / development set |
| A1 | 约 64 蛋白 | 四机制平衡、正式消融、蛋白级统计检验、冻结评估 |
| A2 | 128–256 蛋白 | 跨家族泛化、长度与结构域分层、OOD、风险参数敏感性、大规模 Pareto 可靠性 |

**规模数字在 §5.1 census 产出入选率之前均为估计值。**

### 4.3 机制分层定义

| 机制 | 定义 | 研究作用 |
| --- | --- | --- |
| Easy control | PDB 与 AFDB 整体与局部均较一致 | 建立正常条件下的背景波动 |
| Low-confidence local | AFDB 存在内部连续低 pLDDT 区段 | 局部结构不确定性是否集中影响附近残基选择 |
| High-PAE long-range | 局部可信但域间/长程残基对关系不确定 | 长程约束不确定性 |
| High-confidence state disagreement | AFDB 局部 pLDDT 高但与 PDB 实验构象几何差异显著 | 模型是否把高置信预测状态误当唯一真实状态 |

**关键约束**：`mechanism_label` 已是 A1 manifest 的 canonical identity 字段，但该标签由**旧筛选流程**基于旧叠合方式与旧残基集合算出，而 P1 的配对集合（PDB 已解析 ∩ AFDB fragment 覆盖 ∩ mapping 干净）通常更小。因此：

> manifest 中的 `mechanism_label` 只能用作**分层抽样的先验**，不得用作分析中的自变量。P6 必须在 P1 输出的配对残基集合上**重算**机制证据，并产出 `mechanism_label_recomputed`。二者不一致的蛋白需单独报告。

### 4.4 两个隐藏混杂源

**（a）复合物 vs 单体。** 若 PDB chain 在实验结构中处于复合物或界面接触状态，剥离伙伴链后得到的是 complex-state monomer backbone，而 AFDB 是 free monomer 预测。这类差异会被归入 high-confidence state disagreement，但成因完全不同（与 ligand-induced change 同属真实生物状态差异）。

**要求**：P1/P6 记录该链是否存在跨链界面接触、界面残基数、5 Å 内是否有 ligand。这些字段 **P7 不可见**，但 P6/P9 分层必须有。

**（b）叠合方式决定机制分类。** 对多域蛋白做全局 Kabsch 叠合时，单一铰链角差会在两个域上同时制造大量虚假的"局部几何分歧"，而这恰是 high-PAE long-range 想描述的现象，导致第三、四类样本互相污染。

**要求**：局部分歧一律使用叠合无关指标（lDDT，或局部窗口内的距离矩阵差）；域间关系单独用刚体变换参数描述。两个量分开后，"局部不准"与"局部准但域间不准"才可分。**backbone 文件保持各自原生坐标系，不得把叠合烘焙进 P1 产物**（ProteinMPNN 为 SE(3) 不变，全局叠合不影响打分，但严重影响 P6 的 displacement）。

### 4.5 打分区间

所有条件下的评分限制在**同一组 mapped residues** 上，该集合写入样本记录。若两侧打分残基集合不同，长度效应会直接混入 $\alpha_s^B$。

---

## 5. 阶段规格（P0–P9）

工程契约以 A8 handoff 为准。以下只写**科学侧**的新增要求。

### 5.0 前置：eligibility census（新增，无需 P2 授权）

`resolve_p0_inputs` / `resolve_p1_inputs` 均为只读、不写、不下载。据此构建最小 census harness，分两轮：

**第一轮（不带任何容忍度）**：仅统计 failure_code 分布，重点是 `pdb_amino_acid_mismatch` 的命中数与每蛋白 mismatch 计数分布、`unsupported_afdb_fragment` 占比、文件缺失占比（单列为独立 code，使第一轮可只跑本地已落盘蛋白，不申请下载授权）。→ 供 D1/D2 决策。

**第二轮（协议冻结后）**：产出最终入选表与 attrition 表，后者进入论文 dataset construction 章节。

### 5.1 P0 Resolve & Freeze Inputs ✅ 已实现

七类输入解析、科学身份校验、immutable lock。无新增科学要求。

### 5.2 P1 Build & Validate Paired Backbones ✅ 已实现

新增记录要求（§4.4a、§D3）：segment 结构、gap 边界、界面/ligand 标记。这些为**追加字段**，不改变现有严格性契约。

### 5.3 P2 Generation

在两个 backbone 上分别运行 ProteinMPNN。冻结项见 D3、D4。产出候选池并记录每条候选的 source、temperature、seed。

### 5.4 P3 Cross-score

每条候选在两个 backbone 上都打分，得到 home advantage、cross-source preference、rank shift。回答：结构输入变化时 ProteinMPNN 的序列偏好是否真的改变。

### 5.5 P4 Matched neutral controls

Hamming 1/2/4/8/16/32，突变残基组成 matched（非随机突变）。回答：observed backbone preference 是否超过同等突变量下的普通序列扰动。网格需按 D4 对齐候选真实距离分布。

### 5.6 P5 Residue probability localization

unconditional / conditional repeats 各 16，按 D5 拆为 selection half 与 estimation half。产出 residue 级 source probability contrast、context contrast、entropy、favored AA，即 $G$ / $C$ / $J$ 证据。

### 5.7 P6 Evidence join

将 ProteinMPNN sensitivity + pLDDT + PDB/AFDB geometry + PAE + mapping quality 映射到同一 UniProt residue。按 §4.4b 使用叠合无关的局部指标；产出 `mechanism_label_recomputed`（§4.3）。P6 可见全部机制证据。

### 5.8 P7 Mechanism-blind site freeze

仅可读：sequence completeness、mapping cleanliness（含 gap 距离）、P5 $G$（selection half）、P5 $C$、rank/percentile、favored AA。

**禁止读取**：mechanism label（含 manifest 中的 `mechanism_label`）、PAE、pLDDT、CA displacement、RMSD、界面/ligand 标记。

冻结约 9 site/protein。**验收测试必须包含：置换 manifest 的 `mechanism_label` 字段不改变 P7 panel/hash。** Phase 0 已规定机制字段置换不改变 panel，本文件将该规定落到 **manifest 字段级**，而不只是证据表字段级——因为 manifest 是所有阶段的上游，P7 若按惯例读取 identity 字段就会自动破盲。

P7 的作用是阻断"先看见 high PAE，再专门挑 high PAE site，再证明它与 high PAE 有关"的循环论证，是全文因果可信度的关键防线。

### 5.9 P8 Counterfactual scoring

对冻结位点做 PDB vs AFDB 的局部 counterfactual cross-scoring，产出 $I_{\text{DID}}$、reciprocal preference、bootstrap CI、robust positive $I$。估计仅用 estimation half（D5）。

可选扩展（若时间允许）：局部结构替换（把 AFDB 局部区段替换为 PDB 坐标）后检验残基归因与候选排序是否同步恢复，为局部解释提供真正的反事实证据。

### 5.10 P9 Protein summary

汇总为三层：`dataset_a_candidates` / `dataset_a_sites` / `dataset_a_proteins`。**推断只在 protein 层进行**（§2.2）。

---

## 6. 新增 track：评估器维度（P3′ / P3″）

### 6.1 为什么必须新增

现有 P0–P9 全程只有 ProteinMPNN 单一打分来源，因子只有结构条件一个维度。在此设置下只能估 $\alpha_s^B$，**无法估 $\gamma^{B\times M}$**，即 H3 无阶段承载。同时，多目标 Pareto、FalseParetoRate、probabilistic dominance、held-out oracle 这一整块贡献同样悬空。

### 6.2 P3′ Multi-evaluator scoring

对每条候选 × 每个结构条件 × 每个评估器打分：

```text
candidate × {PDB, AFDB} × {m_1 ... m_K, oracle_heldout}
```

**observed 评估器集合**：建议至少覆盖 foldability / solubility / thermostability 三个属性。可复用已有的 LoRA property teacher 以降低实现成本。

**held-out oracle**：架构与训练数据均须独立于 observed 集合。**若二者共享训练数据，FalseParetoRate 测量的是相关误差而非过度优化，数值会系统性偏乐观**——这一点必须在论文中明确说明并给出独立性证据。

**成本提示**：评估器数量对候选打分成本是线性放大，因此 P3′ 的评估器集合必须在 **P2 候选池冻结之前**定好。

### 6.3 P3″ Pareto 与交互分析

- 观测 Pareto $\mathcal P_{\text{observed}}$ 与 held-out Pareto $\mathcal P_{\text{held-out}}$ 比较；
- $\mathrm{FalseParetoRate} = |\mathcal P_{\text{observed}}\setminus\mathcal P_{\text{held-out}}| / |\mathcal P_{\text{observed}}|$；
- 混合模型分解 $\alpha^B$ / $\beta^M$ / $\gamma^{B\times M}$，protein 为随机效应；
- 评估器分歧（评估器间方差）在四机制间的分层比较；
- 不确定性感知的 Pareto 定义对比：probabilistic dominance、lower-confidence-bound Pareto、posterior Pareto membership、CVaR Pareto、worst-case Pareto。

### 6.4 阶段图更新

```text
P2 ──┬── P3 cross-score ──┬── P3′ multi-evaluator ── P3″ Pareto & interaction
     │                    │
     │                    └── P4 matched controls
     └── P5 residue probability ── P6 evidence join ── P7 blind freeze ── P8 counterfactual ── P9
```

---

## 7. 统计分析计划（pre-registered）

### 7.1 主要终点

| 假设 | 主要终点 | 检验 |
| --- | --- | --- |
| H1 | protein 级（source-conditioned effect − matched-control effect） | 配对 Wilcoxon signed-rank |
| H2 | 机制作为固定因子的效应差异 | 混合模型，protein 随机效应；机制间对比 |
| H3 | $\gamma^{B\times M}$ | 混合模型似然比检验 + 蛋白内置换检验 |
| H4 | protein 级 out-of-selection $I_{\text{DID}}$ | 配对检验；robust positive 计数作次要终点 |

### 7.2 多重性控制

四个主要终点采用 Holm 校正；机制间的两两对比在 H2 内部单独校正。次要终点（top-$k$ overlap、Kendall $\tau$、Pareto membership stability、评估器分歧等）标注为探索性，不参与确证性结论。

### 7.3 功效与规模

在 §5.0 第二轮 census 产出入选率后，用 A0-dev 的效应量估计做功效分析，反推每机制所需蛋白数，据此确认或修订 A1 = 64 / A2 = 128–256。**功效分析结果须在 A1 冻结前写入 PDR。**

### 7.4 抽样框留档

A1/A2 按机制平衡抽样时，完整候选池与各机制入选概率必须留档，否则 A2 声称的"跨家族泛化"无法说明这些机制在真实分布中的稀有程度。

---

## 8. 门禁与验收

| 门 | 条件 | 阻断内容 |
| --- | --- | --- |
| **Gate 0** | census 两轮完成；§3 五项决策全部冻结并进入 manifest schema | 阻断 P2 实现 |
| **Gate 1** | Index 103 + Index 36 完整通过新 P0–P9（含 P3′） | 阻断 pilot |
| **Gate 2** | 5–10 protein pilot 通过：runtime / GPU 成本 / 磁盘 / P0-P1 失败率 / P7 site quota 成功率 / resume 稳健性 / fragment 可用性 | 阻断 scale-out |

不得因旧 V1/V2 曾跑通就跳过 Gate 1 直接批量处理候选池。

---

## 9. 分工建议

| Track | 职责 | 交付物 | 依赖 |
| --- | --- | --- | --- |
| **T0 协议** | 起草并冻结 PDR（§3 五项）、维护本文件 | PDR v1.0、manifest schema 扩展 | census 第一轮 |
| **T1 数据入选** | census harness、两轮 census、attrition 表、机制配额补偿 | 入选表、attrition 表 | 只读接口（已有） |
| **T2 pipeline** | P2–P9 实现，遵循 A8 工程契约与 §5 科学要求 | 各 stage 实现 + RED 测试 | Gate 0 |
| **T3 评估器** | P3′ 评估器集合、held-out oracle 独立性论证、P3″ 分析 | 评估器 registry、Pareto 分析代码 | P2 候选池规格（D4） |
| **T4 编排** | 最小 census harness → 完整 orchestrator（含目录级原子重命名） | CLI / runner | — |
| **T5 统计** | 功效分析、混合模型、置换检验、多重性控制 | 分析脚本、pre-registration 统计章节 | P9 输出格式 |

**协作纪律**（承自 A8 handoff §13）：每个任务先读上游契约 → 记录脏工作区且不清理 → 先写聚焦测试确认 RED → 只实现规定范围 → 精确 stage 文件（禁用 `git add .` / `git add -A`）→ 单一目的提交 → 到任务边界即停止。若新任务与已冻结的 A1–A8 契约冲突，**报告冲突并请求显式协议决策，不得静默修改早期模块**。

**工程改进建议（T4）**：当前"孤儿输出需人工审计后整目录移走、不加 force flag"的策略在 128–256 蛋白规模下会成为吞吐瓶颈。建议改为写入同级临时目录，全部 declared outputs 完成且校验通过后用目录级原子重命名迁移到最终路径。这样部分状态永不出现在 stage 目录中，immutability 与"无 force flag"两条契约均无需放松。

---

## 10. 风险登记

| # | 风险 | 影响 | 缓解 |
| --- | --- | --- | --- |
| R1 | 评估器维度缺失 | H3 不成立，核心贡献落空 | §6 新增 track，且须在 P2 前定好评估器集合 |
| R2 | 入选率未知 | 规模承诺无依据 | Gate 0 census |
| R3 | 无中间结果可回退（A0-dev panel 为空） | 任何协议返工 = 全量重跑 | 决策全部前置到 Gate 0 |
| R4 | fragment 长度上限与 high-PAE 机制冲突 | 机制配额不平衡 | 筛选期主动过采样 |
| R5 | held-out oracle 与 observed 共享训练数据 | FalseParetoRate 偏乐观 | 独立性证据写入论文 |
| R6 | 叠合方式污染机制分类 | H2 结论不可靠 | 叠合无关指标（§4.4b） |
| R7 | 选择偏倚（winner's curse） | $I_{\text{DID}}$ 偏大 | D5 样本分裂 |
| R8 | pilot 期需连续人工审计，碎片时间不可用 | 排期延误 | 与其他投稿窗口错开 |

---

## 11. 动机表述（论文用，英文定稿）

Existing inverse-folding benchmarks generally treat the input backbone as a fixed and error-free structural condition. However, practical protein design increasingly relies on predicted, generated, incomplete, or state-specific structures, for which the observed backbone may deviate from the latent design-relevant conformational ensemble. This discrepancy is not adequately represented by a single global confidence score: local pLDDT, long-range PAE, and high-confidence disagreement with experimental structures correspond to distinct uncertainty mechanisms and may affect sequence decisions differently. We therefore construct Dataset A, a paired PDB–AFDB benchmark that preserves protein identity while varying structural conditions and retaining residue-level confidence, long-range alignment uncertainty, and local geometric disagreement. Dataset A is designed to evaluate whether structural uncertainty changes sequence ranking, multi-objective Pareto membership, evaluator disagreement, and residue-level design decisions, and to test whether joint structure–evaluator uncertainty modeling provides benefits beyond independent confidence weighting.

---

## 附录 A：已知真实数据约束

| Index | 蛋白 | 状态 | 处置 |
| --- | --- | --- | --- |
| 9 | `7kr0_A__P0DTD1` | `unsupported_afdb_fragment`：映射区间与所选 fragment 覆盖不足 | 正确失败。**不得**改标为编号失败、选 F1、搜索最近 fragment 或猜测 offset |
| 36 | `3zoj_A__F2QVG4` | `missing_mapping_column`：历史 mapping 缺 T4 auth/label schema | P1 不得加 legacy fallback 绕过 P0。其 mmCIF 含等占据 A/B altloc，A4 在 `A:33` 已确定性选中 altloc A |
| 103 | `5avd_A__P00772` | P0 通过（AF-P00772-F1，240 mapped residues，UniProt 27–266）；P1 在 output 66 / UniProt 92 处 `pdb_amino_acid_mismatch`（D vs N），一致性约 0.9958 | **科学 eligibility 决策（D1），不是局部放宽 P1 的许可** |

## 附录 B：禁止迁移的历史行为

固定 F1 选择、最近 fragment 选择、fragment 拼接、硬编码或推断 AFDB offset、auth↔label fallback join、chainless join、structure-order pairing、nearest-residue matching、把 PDB/AFDB trim 到交集、丢弃缺 N/CA/C/O 的残基、用 `X` 替代不支持残基、对不一致的 pLDDT/PAE/NPZ 做 padding/truncate/reshape、仅凭文件存在判定可复用、强制覆盖已完成或部分完成的 immutable stage。

`dataset_a_scale` 之外的旧模块可能为历史 A0 workflow 保留兼容行为，其存在不构成在 Dataset A 阶段使用该行为的授权。
