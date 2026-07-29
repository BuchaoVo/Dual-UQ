import json

import numpy as np

from dual_uq.confidence import load_pae, load_plddt


def test_load_plddt_flat_list(tmp_path) -> None:
    path = tmp_path / "plddt.json"
    path.write_text(json.dumps([90.0, 80.0, 70.0]), encoding="utf-8")
    result = load_plddt(path, expected_length=3)
    assert np.allclose(result, [90.0, 80.0, 70.0])


def test_load_pae_matrix_record(tmp_path) -> None:
    path = tmp_path / "pae.json"
    path.write_text(
        json.dumps([{"predicted_aligned_error": [[0.0, 1.0], [2.0, 0.0]]}]),
        encoding="utf-8",
    )
    result = load_pae(path, expected_length=2)
    assert result.shape == (2, 2)
    assert np.isclose(result[0, 1], 1.0)
