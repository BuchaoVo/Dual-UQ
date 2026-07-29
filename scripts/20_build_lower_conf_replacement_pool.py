from __future__ import annotations
import argparse, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, yaml
from dual_uq.net import request_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/lower_conf_replacement.yaml'); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); root=Path(cfg['project_root'])
    src=pd.read_parquet(root/cfg['source']['discovered_candidates'])
    cur=pd.read_csv(root/cfg['source']['current_pool'],sep='\t')
    used=set(cur.uniprot_id.astype(str).str.upper()); s=cfg['selection']
    cand=src[(src.discovery_status=='eligible') & src.pdb_id.notna() & src.uniprot_id.notna()]
    cand=cand[(cand.afdb_global_plddt<=s['global_plddt_max']) & cand.length.between(s['length_min'],s['length_max'])]
    cand=cand[~cand.uniprot_id.astype(str).str.upper().isin(used)]
    if s.get('require_single_chain_proxy',True) and 'protein_chain_count' in cand:
        cand=cand[pd.to_numeric(cand.protein_chain_count,errors='coerce').fillna(999)==1]
    rows=[]
    for i,row in enumerate(cand.itertuples(index=False),1):
        acc=str(row.uniprot_id).upper(); print(f'[{i}/{len(cand)}] {acc}',flush=True)
        try:
            recs=request_json(f'https://alphafold.ebi.ac.uk/api/prediction/{acc}')
            if not isinstance(recs,list) or not recs: continue
            r=next((x for x in recs if str(x.get('modelEntityId','')).endswith('-F1')),recs[0])
            seq=''.join(str(r.get('uniprotSequence') or r.get('sequence') or '').split())
            if not seq: continue
            ratio=float(row.length)/len(seq)
            if not (s['min_pdb_to_uniprot_length_ratio']<=ratio<=s['max_pdb_to_uniprot_length_ratio']): continue
            x=row._asdict(); x['uniprot_length']=len(seq); x['pdb_to_uniprot_length_ratio']=ratio; rows.append(x)
        except Exception as e: print(f'  failed: {type(e).__name__}: {e}',flush=True)
        time.sleep(0.05)
    out=pd.DataFrame(rows)
    if out.empty: raise RuntimeError('No replacement candidates passed length-ratio filtering.')
    rng=np.random.default_rng(s['random_seed']); out['_r']=rng.random(len(out))
    out=out.sort_values(['afdb_global_plddt','resolution','_r'],ascending=[True,True,True])
    selected=[]; org=Counter()
    for _,r in out.iterrows():
        o=str(r.get('organism') or 'UNKNOWN')
        if org[o]>=s['max_per_organism']: continue
        selected.append(r.to_dict()); org[o]+=1
        if len(selected)>=s['target_count']: break
    out=pd.DataFrame(selected).drop(columns=['_r'],errors='ignore')
    out.insert(0,'screening_index',range(101,101+len(out)))
    out['provisional_stratum']='lower_global_confidence_replacement'
    path=root/cfg['output']['replacement_pool']; path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(path,sep='\t',index=False)
    cols=[c for c in ['screening_index','pdb_id','chain_id','uniprot_id','length','uniprot_length','pdb_to_uniprot_length_ratio','afdb_global_plddt','resolution','organism'] if c in out]
    print('\n'+out[cols].to_string(index=False)); print(f'\nSaved: {path}')
if __name__=='__main__': main()
