from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from dual_uq.structure_io import load_chain_ca_table
from dual_uq.net import request_json

def norm(s): return set(pd.Series(s).astype(str).str.strip())
def main():
    p=argparse.ArgumentParser(); p.add_argument('--pair-dir', required=True); a=p.parse_args()
    d=Path(a.pair_dir).expanduser().resolve(); report=json.loads((d/'pair_qc.json').read_text())
    m=pd.read_parquet(d/'residue_mapping.parquet')
    pdb=load_chain_ca_table(report['pdb_path'], report['chain_id'])
    af=load_chain_ca_table(report['afdb_model_path'], 'A')
    mpdb=norm(m.pdb_residue_number); ppdb=norm(pdb.pdb_residue_number)
    muni=set(m.uniprot_residue_number.dropna().astype(int)); ares=set(af.residue_number_int.astype(int))
    recs=request_json(f"https://alphafold.ebi.ac.uk/api/prediction/{report['uniprot_id']}")
    ranges=[]
    if isinstance(recs,list):
      for r in recs:
        ranges.append({k:r.get(k) for k in ['modelEntityId','sequenceStart','sequenceEnd','latestVersion']})
    result={
      'pair':d.name,'mapping_rows':len(m),'pdb_ca_rows':len(pdb),'afdb_ca_rows':len(af),
      'pdb_number_overlap_count':len(mpdb&ppdb),
      'uniprot_afdb_number_overlap_count':len(muni&ares),
      'mapping_uniprot_min':min(muni) if muni else None,'mapping_uniprot_max':max(muni) if muni else None,
      'afdb_residue_min':min(ares) if ares else None,'afdb_residue_max':max(ares) if ares else None,
      'afdb_prediction_ranges':ranges,
    }
    if result['pdb_number_overlap_count']==0: result['likely_issue']='pdb_sifts_numbering_mismatch'
    elif result['uniprot_afdb_number_overlap_count']==0: result['likely_issue']='afdb_fragment_or_offset_mismatch'
    else: result['likely_issue']='no_obvious_numbering_mismatch'
    out=d/'indexing_diagnosis.json'; out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2)); print(f'\nSaved: {out}')
if __name__=='__main__': main()
