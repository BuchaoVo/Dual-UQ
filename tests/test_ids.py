from dual_uq.ids import stable_id

def test_stable_id_is_deterministic() -> None:
    a = stable_id("protein", "P12345", "1ABC", "A")
    b = stable_id("protein", "P12345", "1ABC", "A")
    assert a == b
    assert a.startswith("protein_")
