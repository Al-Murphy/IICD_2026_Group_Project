"""
concept_shift.ag_backbone
=========================

AlphaGenome (pytorch port) K562 baseline: ONE state-blind prediction per gene.

Loads the ``gtca/alphagenome_pytorch`` all-folds weights, selects K562 tracks
from ``track_metadata.parquet`` (strand-aware), and reduces a single forward
pass to two scalars per gene:

- ``cage``    (PRIMARY): sense-strand CAGE/PRO-cap integrated over TSS +/- window.
- ``rna_seq`` (SECONDARY, noisier): sense-strand RNA-Seq integrated over the gene body.

Because AlphaGenome is state-blind, this baseline is computed once per gene and
reused across all perturbations; the predicted delta for every (pert, gene)
pair is exactly 0. Track-selection / aggregation logic is adapted from
``seq2func_crispri_eval`` (Alan Murphy).

Requires the ``[alphagenome]`` extra.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

# AlphaGenome pytorch-port human 1 bp head sizes and native input length.
AG_SEQ_LEN = 1_048_576
AG_NUM_TRACKS = {"rna_seq": 768, "cage": 640}
DEFAULT_CAGE_WINDOW_BP = 401  # TSS +/- 200 bp (sense strand)


# ---------------------------------------------------------------------------
# Model + track selection
# ---------------------------------------------------------------------------
def load_model(backbone_path: str = None, device: str = None):
    """Load pretrained AlphaGenome (pytorch port).

    ``backbone_path=None`` downloads the all-folds weights from HuggingFace
    (``gtca/alphagenome_pytorch`` / ``model_all_folds.safetensors``). Always go
    through ``from_pretrained`` -- a bare ``AlphaGenome()`` is randomly init'd.
    """
    import torch
    from alphagenome_pytorch import AlphaGenome

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if backbone_path is None:
        from huggingface_hub import hf_hub_download

        print("[ag] loading base AlphaGenome (gtca/alphagenome_pytorch all-folds)")
        backbone_path = hf_hub_download(
            repo_id="gtca/alphagenome_pytorch", filename="model_all_folds.safetensors"
        )
    else:
        print(f"[ag] loading AlphaGenome from: {backbone_path}")
    model = AlphaGenome.from_pretrained(backbone_path, device=device)
    model.eval()
    return model, device


def k562_track_indices(metadata_path: str, output_type: str, cell_line: str = "K562"):
    """Strand-aware K562 track index bundle for one 1 bp head.

    Returns ``({'all','plus','minus': [idx...]}, description)``. Selection matches
    the AlphaGenome track order (human + output_type filtered, reset index).
    """
    import pandas as pd

    n_total = AG_NUM_TRACKS[output_type]
    assay = "RNA-Seq" if output_type == "rna_seq" else output_type.upper()

    if not metadata_path or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"track_metadata.parquet not found: {metadata_path}. Copy it from "
            "seq2func_crispri_eval/metadata/ or regenerate via the alphagenome-pytorch "
            "extract_track_metadata.py script."
        )

    df = pd.read_parquet(metadata_path)
    df_m = df[(df["organism"].str.lower() == "human")
              & (df["output_type"].str.lower() == output_type.lower())].reset_index(drop=True)
    if len(df_m) != n_total:
        warnings.warn(f"expected {n_total} human {assay} tracks, found {len(df_m)}")

    col = "biosample_name" if "biosample_name" in df_m.columns else "track_name"
    df_cl = df_m[df_m[col].str.contains(cell_line, case=False, na=False)]
    if len(df_cl) == 0:
        raise ValueError(f"no {assay} tracks matched {cell_line!r} in column {col!r}")

    idx = df_cl.index.tolist()
    if "track_strand" not in df_cl.columns:
        bundle = {"all": idx, "plus": idx, "minus": idx}
    else:
        ts = df_cl["track_strand"].astype(str).fillna(".")
        plus = df_cl[ts.isin(["+", "."])].index.tolist() or idx
        minus = df_cl[ts.isin(["-", "."])].index.tolist() or idx
        bundle = {"all": idx, "plus": plus, "minus": minus}
    desc = f"{len(idx)} {cell_line} {assay} tracks (plus={len(bundle['plus'])}, minus={len(bundle['minus'])})"
    return bundle, desc


# ---------------------------------------------------------------------------
# Aggregation geometry (sense-oriented sequence: TSS at centre, gene 5'->3')
# ---------------------------------------------------------------------------
def _genebody_indices(tss: int, strand: str, gstart: int, gend: int, rf: int):
    """Positions of the gene body within the sense-oriented one-hot window."""
    t = tss + 1 if (strand == "-" and rf % 2 == 0) else tss
    seq_start = t - rf // 2
    lo = max(gstart, seq_start) - seq_start
    hi = min(gend, seq_start + rf) - seq_start
    if lo >= hi:
        return np.array([], dtype=np.int64)
    pos = np.arange(lo, hi, dtype=np.int64)
    if strand == "-":
        pos = rf - 1 - pos
    return pos[(pos >= 0) & (pos < rf)]


def _forward(model, x, device, organism_index: int = 0):
    """One forward pass; return dict of head -> (B, L, n_tracks) signal tensors."""
    import torch

    org = torch.full((x.shape[0],), organism_index, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x.to(device), org)
    return out


def predict_gene(model, loader, device, chrom, tss, strand, gstart, gend,
                 cage_bundle, rna_bundle, cage_window_bp: int = DEFAULT_CAGE_WINDOW_BP,
                 organism_index: int = 0):
    """Return ``(cage_scalar, rna_scalar)`` K562 baseline for one gene.

    A single forward pass drives both heads. Strand matters twice: the sequence
    is fetched sense-oriented (``loader.get_seq`` reverse-complements the minus
    strand) and the sense-strand track subset is read from each head.
    """
    import torch

    x = loader.get_seq(chrom, int(tss), strand, ohe=True)  # (1, 4, rf)
    out = _forward(model, x, device, organism_index=organism_index)

    for key in ("cage", "rna_seq"):
        if key not in out:
            raise KeyError(f"AlphaGenome output missing {key!r}; keys={list(out.keys())}")

    L = out["cage"][1].shape[1]
    center = L // 2

    # (B) CAGE primary: sum sense-strand tracks over TSS +/- window.
    cage_sig = out["cage"][1][:, :, cage_bundle["plus"]]
    half_lo = cage_window_bp // 2
    start = max(0, center - half_lo)
    end = min(L, center + (cage_window_bp - half_lo))
    cage_scalar = float(cage_sig[:, start:end, :].sum(dim=(1, 2)).item())

    # (C) RNA-Seq secondary: sum sense-strand tracks over the gene body.
    rna_sig = out["rna_seq"][1][:, :, rna_bundle["plus"]]
    body = _genebody_indices(int(tss), strand, int(gstart), int(gend), L)
    if body.size == 0:
        rna_scalar = float("nan")
    else:
        idx_t = torch.tensor(body, dtype=torch.long, device=rna_sig.device)
        rna_scalar = float(rna_sig[:, idx_t, :].sum(dim=(1, 2)).item())

    return cage_scalar, rna_scalar
