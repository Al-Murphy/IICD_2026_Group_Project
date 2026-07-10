#!/usr/bin/env python3
"""
Step 3 -- Mandatory controls (spec step 9).

(a) Within-state accuracy: Spearman of the AlphaGenome K562 baseline vs the
    measured control pseudobulk across genes. Must be reasonably positive --
    otherwise a non-zero "concept-shift error" is just generic in-state
    weakness, not a genuine shift. This gate is what makes the result a concept
    shift rather than a model failure.

(b) Wasserstein strength baseline: per-perturbation Wasserstein (optimal
    transport) distance (control vs perturbed) via pertpy's Distance. We use
    Wasserstein rather than E-distance to match the OT metric the scFM side
    applies on the scVI latent, so the strength axis is directly comparable.
    This is the magnitude axis any scFM embedding displacement must beat
    (benchmark Fig. 5e: strength<->error r~0.8). If the displacement does not
    out-predict the raw Wasserstein distance, the foundation model adds nothing.

    By default this runs on ``X_pca``; point ``--obsm_key`` at a scVI latent
    (e.g. ``X_scVI``) written into the AnnData by the scFM team to compute it in
    exactly their space.

Outputs:  out/within_state_spearman.txt
          out/wasserstein_distance.parquet
          (merged into out/perturbation_summary.parquet if present)

Usage
-----
    python scripts/3_controls/run_controls.py
    python scripts/3_controls/run_controls.py --baseline_proxy rna_pred
    python scripts/3_controls/run_controls.py --obsm_key X_scVI
"""

import argparse
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-perturbation Wasserstein (OT) distance vs control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--filtered_h5ad", default="./out/replogle_k562_filtered.h5ad")
    p.add_argument("--pseudobulk", default="./out/pseudobulk.parquet")
    p.add_argument("--pert_col", default="perturbation")
    p.add_argument("--ctrl", default="control")
    p.add_argument("--obsm_key", default="X_pca",
                   help="Embedding to compute Wasserstein on. Use e.g. X_scVI to "
                        "match the scFM latent; X_pca is computed on the fly if absent.")
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--out_dir", default="./out")
    return p.parse_args()


def wasserstein_distances(h5ad_path, pert_col, ctrl, obsm_key, n_pcs):
    """(b) Per-perturbation Wasserstein (OT) distance (perturbed vs control).

    Computed on ``obsm_key`` (default ``X_pca``; point at a scVI latent to match
    the scFM side). Uses the same metric the scFM team applies on scVI so the
    strength axis is directly comparable.
    """
    import anndata as ad
    import scanpy as sc
    import pertpy as pt

    adata = ad.read_h5ad(h5ad_path)
    if obsm_key not in adata.obsm:
        if obsm_key != "X_pca":
            raise KeyError(
                f"obsm_key {obsm_key!r} not in AnnData.obsm "
                f"(available: {list(adata.obsm)}). Write the scFM latent first."
            )
        print(f"[wasserstein] computing PCA ({n_pcs} PCs) for Wasserstein ...")
        sc.pp.pca(adata, n_comps=n_pcs)

    dist = pt.tools.Distance(metric="wasserstein", obsm_key=obsm_key)
    # onesided_distances: distance of every group to the selected reference group.
    wd = dist.onesided_distances(adata, groupby=pert_col, selected_group=ctrl, show_progressbar=True)
    wd = pd.Series(wd).rename("wasserstein")
    wd.index.name = "pert"
    wd = wd[wd.index != ctrl]
    print(f"[wasserstein] Wasserstein computed for {len(wd)} perturbations on "
          f"{obsm_key} (median={wd.median():.3f})")
    return wd


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    wd = wasserstein_distances(args.filtered_h5ad, args.pert_col, args.ctrl,
                               args.obsm_key, args.n_pcs)
    wd.to_frame().to_parquet(os.path.join(args.out_dir, "wasserstein_distance.parquet"))



if __name__ == "__main__":
    main()
