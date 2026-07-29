# Dual-UQ Protein Inverse Folding

面向“输入骨架不确定性 × 属性评价器不确定性”的结构条件蛋白序列设计项目。

## 第一阶段目标

完成 A0 机制数据闭环，验证：

1. 骨架不确定性是否改变评价器排序；
2. Structure-UQ 与 Evaluator-UQ 是否存在交互；
3. 点估计 Pareto 前沿是否包含 False Pareto 候选；
4. 是否值得进入 Joint-UQ 模型开发。

## 快速开始

```bash
conda create -n dual-uq python=3.11 -y
conda activate dual-uq
pip install -e ".[dev]"
python scripts/00_check_environment.py
python scripts/01_init_manifests.py
pytest -q
```
