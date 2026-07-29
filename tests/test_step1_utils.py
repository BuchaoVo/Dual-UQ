from dual_uq.afdb import _select_monomer_prediction
from dual_uq.pdb_archive import normalize_pdb_id


def test_normalize_legacy_pdb_id() -> None:
    assert normalize_pdb_id("1AKE") == "1ake"


def test_selects_f1_prediction() -> None:
    records = [
        {"modelEntityId": "AF-X-F2", "isComplex": False},
        {"modelEntityId": "AF-X-F1", "isComplex": False},
    ]
    selected = _select_monomer_prediction(records)
    assert selected["modelEntityId"] == "AF-X-F1"
