import pytest
from pydantic import ValidationError

from dual_uq.schema import ProteinRecord


def test_valid_protein_record() -> None:
    record = ProteinRecord(protein_id="P001",uniprot_id="P12345",pdb_id="1ABC",chain_id="A",sequence="ACDE",length=4,mapping_coverage=1.0,sequence_identity=1.0)
    assert record.length == 4

def test_invalid_coverage_rejected() -> None:
    with pytest.raises(ValidationError):
        ProteinRecord(protein_id="P001",uniprot_id="P12345",pdb_id="1ABC",chain_id="A",sequence="ACDE",length=4,mapping_coverage=1.2,sequence_identity=1.0)
