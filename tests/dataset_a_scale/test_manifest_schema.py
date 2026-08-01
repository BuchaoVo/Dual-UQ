from pathlib import Path

from dual_uq.dataset_a_scale.manifest import (
    PROTEIN_MANIFEST_REQUIRED_COLUMNS,
    load_protein_manifest,
    validate_protein_manifest,
    write_protein_manifest,
)


def test_load_manifest_preserves_empty_fields_and_crlf_without_column_shift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "proteins.tsv"
    path.write_bytes(
        b"protein_id\tscreening_index\tpair_id\tmechanism_label\ttier\r\n"
        b"index103\t103\t7xyz_A__P12345\t\t2\r\n"
    )

    frame = load_protein_manifest(path)

    assert frame.columns.tolist() == [
        "protein_id",
        "screening_index",
        "pair_id",
        "mechanism_label",
        "tier",
    ]
    assert len(frame) == 1
    assert frame.loc[0, "pair_id"] == "7xyz_A__P12345"
    assert frame.loc[0, "mechanism_label"] == ""
    assert frame.loc[0, "tier"] == 2

    result = validate_protein_manifest(frame)

    assert result.validation_pass is False
    assert result.error_codes == ["missing_mechanism_label"]


def test_valid_core_manifest_passes_with_extension_columns(tmp_path: Path) -> None:
    path = tmp_path / "proteins.tsv"
    path.write_text(
        "protein_id\tscreening_index\tpair_id\tmechanism_label\ttier\tpair_dir\n"
        "index36\t36\t3zoj_A__F2QVG4\thigh_pae_long_range\t2\tdata/processed/pairs/x\n",
        encoding="utf-8",
        newline="",
    )

    frame = load_protein_manifest(path)
    result = validate_protein_manifest(frame)

    assert PROTEIN_MANIFEST_REQUIRED_COLUMNS == (
        "protein_id",
        "screening_index",
        "pair_id",
        "mechanism_label",
        "tier",
    )
    assert result.validation_pass is True
    assert result.error_codes == []
    assert result.issues == []


def test_missing_required_column_is_structured_failure() -> None:
    frame = load_protein_manifest_data(
        "protein_id\tscreening_index\tpair_id\ttier\n"
        "index36\t36\t3zoj_A__F2QVG4\t2\n"
    )

    result = validate_protein_manifest(frame)

    assert result.validation_pass is False
    assert result.error_codes == ["missing_required_column"]
    assert result.issues[0].column == "mechanism_label"
    assert result.issues[0].row is None


def test_invalid_values_and_duplicate_identities_are_reported() -> None:
    frame = load_protein_manifest_data(
        "protein_id\tscreening_index\tpair_id\tmechanism_label\ttier\n"
        "index36\t0\t3zoj_A__F2QVG4\thigh_pae_long_range\t3\n"
        "index36\tbad\t3zoj_A__F2QVG4\tordinary\t2\n"
    )

    result = validate_protein_manifest(frame)

    assert result.validation_pass is False
    assert result.error_codes == [
        "invalid_screening_index",
        "invalid_tier",
        "duplicate_protein_id",
        "duplicate_pair_id",
    ]
    assert {issue.row for issue in result.issues if issue.row is not None} == {2, 3}


def test_blank_core_identities_do_not_receive_fallbacks() -> None:
    frame = load_protein_manifest_data(
        "protein_id\tscreening_index\tpair_id\tmechanism_label\ttier\n"
        "\t103\t\t\t1\n"
    )

    result = validate_protein_manifest(frame)

    assert result.error_codes == [
        "missing_protein_id",
        "missing_pair_id",
        "missing_mechanism_label",
    ]


def test_write_manifest_uses_lf_and_round_trips_empty_fields(tmp_path: Path) -> None:
    frame = load_protein_manifest_data(
        "protein_id\tscreening_index\tpair_id\tmechanism_label\ttier\tafdb_pae_path\n"
        "index103\t103\t7xyz_A__P12345\teasy_control\t2\t\n"
    )
    path = tmp_path / "proteins.tsv"

    write_protein_manifest(frame, path)

    payload = path.read_bytes()
    assert b"\r" not in payload
    assert payload.endswith(b"\n")
    reloaded = load_protein_manifest(path)
    assert reloaded.loc[0, "afdb_pae_path"] == ""


def load_protein_manifest_data(text: str):
    from io import StringIO

    import pandas as pd

    return pd.read_csv(StringIO(text), sep="\t", keep_default_na=False)
