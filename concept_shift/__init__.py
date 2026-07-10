"""
concept_shift
=============

Concept-shift analysis of the Replogle 2022 K562-essential CRISPRi screen
against a state-blind AlphaGenome baseline.

Core idea
---------
AlphaGenome predicts expression from DNA sequence only. A knockdown does not
change any downstream gene's sequence, so AlphaGenome makes the SAME prediction
in every cell state. It is therefore run once per gene for a K562 baseline and
reused across all 1,971 perturbations -- never inside a per-perturbation loop.

The question is then: **does AlphaGenome's accuracy degrade as the cell state
shifts away from control?** We answer it per state with a Spearman correlation
against that state's measured pseudobulk, excluding the perturbed gene, and
correct for the cell-count confound with a matched null. See
:mod:`concept_shift.state_shift`.

Modules
-------
data          Shared load / filter / pseudobulk / coords (torch-free; the scFM
              team imports this to get an identical filtered cell x gene set).
state_shift   The analysis: per-state Spearman, gene filtering, matched null, plots.
networks      STRING functional networks -> per-state gene sets (torch-free).
seq           hg38 genome fetch + one-hot (AlphaGenome extra only).
ag_backbone   AlphaGenome pytorch-port K562 baseline (AlphaGenome extra only).

``data`` and ``state_shift`` import without torch / AlphaGenome; ``seq`` and
``ag_backbone`` need the ``[alphagenome]`` extra and are imported lazily.
"""

from . import networks, state_shift
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
)
from .networks import (
    load_string_edges,
    matched_background,
    per_state_genes_from_network,
    string_neighbours,
)
from .state_shift import (
    compute_state_shift,
    filter_genes,
    load_inputs,
    matched_null,
    noise_floor_curve,
    plot_rho_vs_cells,
    plot_state_shift,
    state_gene_masks,
    state_spearman,
)

__all__ = [
    # data
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
    # state_shift
    "state_shift",
    "load_inputs",
    "filter_genes",
    "state_gene_masks",
    "state_spearman",
    "matched_null",
    "noise_floor_curve",
    "compute_state_shift",
    "plot_state_shift",
    "plot_rho_vs_cells",
    # networks
    "networks",
    "load_string_edges",
    "string_neighbours",
    "per_state_genes_from_network",
    "matched_background",
]
