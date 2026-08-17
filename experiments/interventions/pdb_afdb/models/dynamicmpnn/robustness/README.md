# DynamicMPNN baseline evaluation

This directory owns the DynamicMPNN baseline for the frozen PDB/AFDB
intervention cohort. The model is the official DynamicMPNN source at commit
`1f3e326c0f4d275ee8b3918e4726e19d3eef6c3f` with the two-conformation
`single_chain_k2.ckpt` checkpoint. The checkpoint identity and input contract
are frozen in `protocol.json`.

The baseline uses the complete aligned canonical residue axis. Missing
coordinates are retained as virtual nodes; no sequence or coordinate
imputation, gap compression, or candidate re-admission is performed.

Generated sequences and evaluator-specific comparison tables are runtime
artifacts under `runs/analysis/dynamicmpnn_baseline/`. This baseline does not
rewrite any existing ProteinMPNN, ESM-IF1, APO/HOLO, or generative artifacts.
