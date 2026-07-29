# Project Charter

## 中心问题

输入骨架不确定性是否会系统改变属性评价器的可靠性，并与 evaluator uncertainty 共同影响不同 Pareto 偏好下的局部序列决策？

## 第一阶段研究问题

- RQ1：输入骨架不确定性是否改变 evaluator 排序？
- RQ2：Structure-UQ 与 Evaluator-UQ 是否存在非加性交互？
- RQ3：点估计 Pareto 前沿是否包含不可迁移的假阳性序列？
- RQ4：联合 UQ 是否值得进入模型开发？

## 假设

- H1：评价器分歧随骨架不确定性增加。
- H2：结构依赖属性的交互强于纯序列属性。
- H3：Nominal Pareto 集合包含非平凡 False Pareto 候选。
- H4：交互集中在低 pLDDT、高 PAE、loop 和 domain boundary。
- H5：Joint-UQ 优于 Independent-UQ，而不仅是参数量增加。

## A0 Go 条件

满足以下至少三项：

1. 至少一个结构依赖属性出现稳定的 backbone × evaluator 交互；
2. joint pair-flip 高于单独 backbone/evaluator pair-flip；
3. 高 PAE 或低 pLDDT 区域排序更不稳定；
4. nominal Pareto 存在非平凡 False Pareto 比例；
5. 结果不是由少数异常蛋白或严重 clash 驱动；
6. 低置信扰动效应强于匹配高置信负对照。

## 第一篇论文暂不包含

- Binder / complex design
- 大规模湿实验
- 完整 Bayesian neural network 参数后验
- 五个以上 developability 目标
- 未通过干预验证的“生物学因果”表述
