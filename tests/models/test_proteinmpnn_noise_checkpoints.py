from pathlib import Path

from dual_uq.core.hashing import sha256_file
from dual_uq.models.proteinmpnn import (
    AUTHORIZED_CHECKPOINT_SHA256,
    AUTHORIZED_VANILLA_CHECKPOINTS,
)


ROOT = Path(__file__).resolve().parents[2]


def test_official_v48_noise_checkpoint_allowlist_matches_local_bytes() -> None:
    expected = {
        "v_48_002.pt": ("925f2ca1007bf9b02e0e7f420ff00eb91f50fcc2722f64b42e644ae95adaa131", 0.02),
        "v_48_010.pt": ("db866fae956a28661f926053d630610c55e9fc4bc03922f2aeeb98a37435ccce", 0.10),
        "v_48_020.pt": ("c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd", 0.20),
        "v_48_030.pt": ("c34b7bfb38418ea30989fda3314f4781ac4e3920f9825731cf555f1fed44ac66", 0.30),
    }

    assert AUTHORIZED_VANILLA_CHECKPOINTS == expected
    for filename, (digest, _training_noise) in expected.items():
        path = ROOT / "third_party/ProteinMPNN/vanilla_model_weights" / filename
        assert sha256_file(path) == digest
    assert AUTHORIZED_CHECKPOINT_SHA256 == expected["v_48_020.pt"][0]
