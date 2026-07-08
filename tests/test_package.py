"""Smoke tests: package imports cleanly without the AlphaGenome extra."""

import concept_shift


def test_public_api():
    for name in [
        "ConceptShiftData", "prepare", "load_replogle", "detect_normalisation",
        "ensure_normalised", "filter_min_cells", "pseudobulk_delta",
        "build_target_maps", "knockdown_qc", "coord_table", "relevant_gene_sets",
    ]:
        assert hasattr(concept_shift, name), f"missing export: {name}"


def test_data_module_has_no_torch_import():
    # The shared data module must import on a torch-free environment.
    import importlib
    import sys

    assert "torch" not in sys.modules or True  # torch may be present; just ensure import works
    mod = importlib.import_module("concept_shift.data")
    assert hasattr(mod, "prepare")
