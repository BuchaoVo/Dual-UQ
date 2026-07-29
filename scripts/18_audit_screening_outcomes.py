from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root', required=True); a=p.parse_args()
    root=Path(a.project_root).expanduser().resolve()
    df=pd.read_csv(root/'data/manifests/screening_pool_preflight.tsv', sep='\t')
    print('=== Overall ==='); print(df.preflight_status.value_counts(dropna=False).to_string())
    print('\n=== By provisional stratum ===')
    print(pd.crosstab(df.provisional_stratum, df.preflight_status, margins=True).to_string())
    passed=df[df.preflight_status=='pass_full_length']
    summary={
      'total': int(len(df)),
      'pass_full_length': int(len(passed)),
      'warn_construct_difference': int((df.preflight_status=='warn_construct_difference').sum()),
      'fail_preflight': int((df.preflight_status=='fail_preflight').sum()),
      'pass_by_stratum': passed.provisional_stratum.value_counts().to_dict(),
      'lower_confidence_pass_missing': int((passed.provisional_stratum=='lower_global_confidence').sum())==0,
    }
    out=root/'reports/screening_outcome_audit.json'; out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\n'+json.dumps(summary,indent=2)); print(f'\nSaved: {out}')
if __name__=='__main__': main()
