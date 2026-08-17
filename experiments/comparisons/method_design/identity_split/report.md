# Dual-UQ Identity-Disjoint Split Freeze

- VERDICT: `READY`
- Clustering: MMseqs2 `18.8cc5c`, min identity `0.3`, coverage `0.8`, cov-mode `0` (coverage of query and target), mode `connected_components` (connected components of thresholded MMseqs2 all-vs-all hits).
- Proteins: `1410`; clusters: `1017`; singleton fraction: `0.8328416912487709`; largest cluster: `{'cluster_id': 'A0A086RSH0', 'protein_count': 38}`.
- Canonical sequence freeze: `1410` proteins, `1404` unique sequence hashes, conflicting duplicate identity: `False`, nonstandard-character records retained exactly: `['A5YV76']`.
- Split target: deterministic `20260817` assignment of complete clusters to TRAIN/VALIDATION/LOCKED_TEST.
- Leakage audit: `0` cross-split violations across TRAIN/VALIDATION, TRAIN/LOCKED_TEST, and VALIDATION/LOCKED_TEST.
- Algorithm selection: `A diagnostic set-cover run produced 17 cross-split threshold hits; it was not frozen. The final contract uses connected components of the same MMseqs2 identity/coverage hits, with no protein deletion or manual cluster splitting.`
- Terminology: this is an internal Dual-UQ 30%-identity cluster split, not UniRef30 and not an external prospective benchmark.
- No model response, geometry, generation, or downstream outcome was used for clustering or balancing.
- V1 training was not started.
