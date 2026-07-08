"""
concept_shift
=============

Concept-shift pipeline for the Replogle 2022 K562-essential CRISPRi screen
crossed with a state-blind AlphaGenome baseline.

Core idea
---------
AlphaGenome predicts expression from DNA sequence only. A knockdown does not
change any downstream gene's sequence, so AlphaGenome's predicted delta for
every (perturbation, gene) pair is exactly 0. The measured non-zero delta from
the screen is the concept-shift signal. AlphaGenome is therefore run ONCE per
gene for a K562 baseline and reused across all perturbations -- never inside a
per-perturbation loop.

Modules
-------
data          Shared load / filter / pseudobulk / relevant-gene-sets (spec steps 1-7).
              Pure -- no torch / AlphaGenome deps; used by the scFM team too.
seq           hg38 genome fetch + one-hot (AlphaGenome extra only).
ag_backbone   AlphaGenome pytorch-port K562 baseline (AlphaGenome extra only).

The ``data`` API is imported eagerly; ``seq`` / ``ag_backbone`` are imported
lazily so the package works without the ``[alphagenome]`` extra installed.
"""

from .data import (
    ConceptShiftData,
    prepare,
    load_replogle,
    detect_normalisation,
    ensure_normalised,
    filter_min_cells,
    pseudobulk_delta,
    build_target_maps,
    knockdown_qc,
    coord_table,
    relevant_gene_sets,
)

__all__ = [
    "ConceptShiftData",
    "prepare",
    "load_replogle",
    "detect_normalisation",
    "ensure_normalised",
    "filter_min_cells",
    "pseudobulk_delta",
    "build_target_maps",
    "knockdown_qc",
    "coord_table",
    "relevant_gene_sets",
]
