#!/usr/bin/env python3
"""
Step 2 -- Per-perturbation Wasserstein (OT) distance: the strength axis.

For each perturbed state, the optimal-transport distance between its cells and
the control cells, in an embedding space. We use **Wasserstein** rather than
E-distance to match the OT metric the scFM side applies on the scVI latent, so
the strength axis is directly comparable. This is the magnitude any scFM
embedding displacement must beat (cf. benchmark Fig. 5e, strength<->effect
r~0.8): if the displacement does not out-predict raw Wasserstein, the foundation
model adds nothing.

How it works
------------
The distances are **not stored in the AnnData** -- they are computed from it:

1. read ``out/replogle_k562_filtered.h5ad``;
2. if ``--obsm_key`` is absent from ``adata.obsm``, compute it (``X_pca``, 50 PCs)
   on the fly -- it is not cached back into the h5ad;
3. ``pertpy.tools.Distance(metric="wasserstein", obsm_key=...)`` then
   ``onesided_distances(groupby="perturbation", selected_group="control")``
   gives every state's OT distance to control;
4. drop the control-vs-control row and write the 1,971-row Series.

To run in the scFM space instead, write the latent into ``adata.obsm["X_scVI"]``
and pass ``--obsm_key X_scVI``; the PCA fallback then raises rather than silently
using PCA.

Output: out/wasserstein_distance.parquet  (read back by step 3 via
        ``concept_shift.state_shift.load_inputs``)

Usage
-----
    python scripts/2_wasserstein/run_wasserstein.py
    python scripts/2_wasserstein/run_wasserstein.py --obsm_key X_scVI
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
    """Per-perturbation Wasserstein (OT) distance (perturbed vs control).

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
