import numpy as np

from dual_uq.geometry import kabsch_align, rmsd


def test_kabsch_recovers_rigid_transform() -> None:
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mobile = target @ rotation.T + np.array([5.0, -3.0, 2.0])
    aligned, _, _ = kabsch_align(mobile, target)
    assert rmsd(aligned, target) < 1e-8
