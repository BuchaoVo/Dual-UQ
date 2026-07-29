from dual_uq.context_analysis import classify_segment


def test_rigid_state_shift_classification() -> None:
    result = classify_segment(
        local_fit_rmsd=0.3,
        flank_fit_segment_rmsd=2.2,
        internal_distance_mae=0.2,
    )
    assert result.label == "rigid_or_state_shift"


def test_local_deformation_classification() -> None:
    result = classify_segment(
        local_fit_rmsd=1.4,
        flank_fit_segment_rmsd=1.8,
        internal_distance_mae=0.9,
    )
    assert result.label == "local_deformation"
