"""Generic benchmark eligibility, independent of model availability."""

def build_generic_eligibility(*, mapping_available: bool, geometry_available: bool, ligand_available: bool = False) -> dict[str, bool | str | None]:
    return {
        "local_sensitivity_eligible": bool(mapping_available),
        "generation_eligible": bool(mapping_available),
        "compatibility_eligible": bool(mapping_available),
        "multistate_eligible": bool(mapping_available),
        "geometry_evaluable": bool(geometry_available),
        "ligand_evaluable": bool(ligand_available),
        "eligibility_reason": None if mapping_available else "MAPPING_UNAVAILABLE",
    }
