# Step 1 — PDB–UniProt–AlphaFoldDB 配对

## 安装新增依赖

```bash
cd /home/zbc/data/AI4S/ProteinDesign/Dual-UQ
conda activate dual-uq
pip install -r requirements-step1.txt
```

## 单个 easy pair

```bash
python scripts/03_build_pair.py \
  --pdb-id 1ake \
  --chain-id A \
  --uniprot-id P69441
```

## 单个 mapping edge case

```bash
python scripts/03_build_pair.py \
  --pdb-id 1lyz \
  --chain-id A \
  --uniprot-id P00698
```

`1lyz/P00698` 的实验结构是成熟链，而 UniProt/AFDB 包含信号肽和前肽，因此预期出现 coverage warning；该样本用于检查流程是否诚实记录构建体差异，不用于第一轮 pass 样本。

## 批量 smoke

```bash
python scripts/04_run_smoke_pairs.py
python scripts/05_summarize_pairs.py
python scripts/02_validate_manifests.py
pytest -q
```

## 预期输出

```text
data/raw/pdb/<pdb>.cif
data/raw/afdb/<accession>/metadata.json
data/raw/afdb/<accession>/model.cif
data/raw/afdb/<accession>/plddt.json
data/raw/afdb/<accession>/pae.json
data/raw/mappings/<pdb>.xml.gz
data/processed/pairs/<pair>/residue_mapping.parquet
data/processed/pairs/<pair>/pair_qc.json
reports/pair_smoke_summary.json
reports/pair_manifest_summary.csv
```
