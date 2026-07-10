#!/usr/bin/env python3
"""
Step 0 -- Preprocess the Replogle K562-essential screen.

Runs the shared ``concept_shift.data`` pipeline and caches every artifact the
downstream AlphaGenome steps and the scFM team consume:

    out/replogle_k562_raw.h5ad          raw pertpy download (cache)
    out/replogle_k562_filtered.h5ad      30-cell filtered + normalised AnnData
    out/pseudobulk.parquet               per-perturbation mean expression
    out/delta.parquet                    measured delta vs control
    out/knockdown_check.csv              soft percent-of-control QC flag
    out/coords.parquet                   strand-aware hg38 TSS table

Usage
-----
    # Full run (downloads the 1.44 GB scPerturb h5ad the first time)
    python scripts/0_preprocess/prepare_data.py

    # Reuse an existing filtered cache
    python scripts/0_preprocess/prepare_data.py \
        --filtered_h5ad out/replogle_k562_filtered.h5ad
"""

import argparse
import os

import pandas as pd

from concept_shift import data


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess Replogle K562-essential screen (concept-shift state side).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", default="./out")
    p.add_argument("--raw_h5ad", default="./out/ReplogleWeissman2022_K562_essential.h5ad",
                   help="Raw scPerturb Replogle h5ad. Downloaded from Zenodo if absent.")
    p.add_argument("--filtered_h5ad", default="./out/replogle_k562_filtered.h5ad",
                   help="Filtered+normalised AnnData cache (loaded if present).")
    p.add_argument("--min_cells", type=int, default=data.DEFAULT_MIN_CELLS)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csd = data.prepare(
        cache_h5ad=(args.raw_h5ad or None),
        filtered_h5ad=(args.filtered_h5ad or None),
        min_cells=args.min_cells,
    )

    # --- cache tables ---------------------------------------------------------
    csd.pb.to_parquet(os.path.join(args.out_dir, "pseudobulk.parquet"))
    csd.delta.to_parquet(os.path.join(args.out_dir, "delta.parquet"))
    csd.knockdown_qc.to_csv(os.path.join(args.out_dir, "knockdown_check.csv"), index=False)
    csd.coords.to_parquet(os.path.join(args.out_dir, "coords.parquet"))


    # --- summary --------------------------------------------------------------
    n_perts = len(csd.perturbations)
    kd = csd.knockdown_qc
    n_not_measured = int((kd["status"] == "target_not_measured").sum())
    n_ok = int(kd["knockdown_ok"].sum())
    print("\n=== preprocess summary ===")
    print(f"perturbations retained : {n_perts}  (expected ~1971)")
    print(f"genes retained         : {csd.adata.n_vars}  (expected 8563; NO HVG filter)")
    print(f"genes with valid hg38  : {len(csd.coords)}")
    print(f"knockdown_ok           : {n_ok}/{n_perts}")
    print(f"target_not_measured    : {n_not_measured}  (expected ~180; kept, not dropped)")
    print(f"artifacts written to   : {args.out_dir}/")


if __name__ == "__main__":
    main()
