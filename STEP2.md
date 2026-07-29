# Step 2 — 置信度解析与PDB–AFDB几何诊断

诊断输出显示当前实际项目路径为：

```text
/mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ
```

后续命令以该路径为准；脚本本身通过 `pair_qc.json` 读取绝对数据路径。

## 安装

```bash
cd /mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ
conda activate dual-uq
pip install -r requirements-step2.txt
pip install -e .
pytest -q
```

## 运行1AKE

```bash
python scripts/06_analyze_pair_geometry.py   --pair-report data/processed/pairs/1ake_A__P69441/pair_qc.json   2>&1 | tee reports/1ake_geometry.log
```

## 查看局部差异最大的残基

```bash
python scripts/08_inspect_residue_geometry.py   --residue-table data/processed/pairs/1ake_A__P69441/residue_geometry.parquet   --top-k 20
```

## 汇总所有配对

```bash
python scripts/07_summarize_pair_geometry.py   --project-root /mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ
```

## 输出

```text
data/processed/pairs/1ake_A__P69441/
├── residue_geometry.parquet
├── pairwise_geometry.npz
└── pair_geometry_qc.json

reports/
├── 1ake_geometry.log
└── pair_geometry_summary.csv
```
