"""Canonical protein normalization without model-specific sanitization."""


def normalize_protein(uniprot_id: str, sequence: str, *, source_provenance: str | None = None) -> dict[str, object]:
    accession = str(uniprot_id).strip().upper()
    sequence = str(sequence).strip().upper()
    if not accession or not sequence:
        raise ValueError("protein identity and sequence are required")
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    status = "STANDARD_20AA" if set(sequence) <= standard else "NONSTANDARD"
    return {
        "protein_id": accession,
        "uniprot_id": accession,
        "canonical_sequence": sequence,
        "canonical_length": len(sequence),
        "canonical_sequence_status": status,
    }
