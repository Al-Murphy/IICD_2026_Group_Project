#!/usr/bin/env python3
"""
Step 0 -- Preprocess the Replogle K562-essential screen (spec steps 1-7).

Runs the shared ``concept_shift.data`` pipeline and caches every artifact the
downstream AlphaGenome steps and the scFM team consume:

    out/replogle_k562_raw.h5ad          raw pertpy download (cache)
    out/replogle_k562_filtered.h5ad      30-cell filtered + normalised AnnData
    out/pseudobulk.parquet               per-perturbation mean expression
    out/delta.parquet                    measured delta vs control
    out/knockdown_check.csv              soft percent-of-control QC flag
    out/coords.parquet                   strand-aware hg38 TSS table
    out/relevant_genes.parquet           long-form (pert, gene) trans pairs

Usage
-----
    # Full run (downloads ~a few GB the first time; DE step is the slow part)
    python scripts/0_preprocess/prepare_data.py

    # Skip the DE step (scFM team only needs filtered cells + pseudobulk)
    python scripts/0_preprocess/prepare_data.py --no_relevant

    # Reuse an existing filtered cache
    python scripts/0_preprocess/prepare_data.py \
        --filtered_h5ad out/replogle_k562_filtered.h5ad
"""

import argparse
import json
import os

import pandas as pd

from concept_shift import data


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess Replogle K562-essential screen (concept-shift state side).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", default="./out")
    p.add_argument("--raw_h5ad", default="./out/replogle_k562_raw.h5ad",
                   help="Raw pertpy download cache. Set '' to always re-fetch.")
    p.add_argument("--filtered_h5ad", default="./out/replogle_k562_filtered.h5ad",
                   help="Filtered+normalised AnnData cache (loaded if present).")
    p.add_argument("--min_cells", type=int, default=data.DEFAULT_MIN_CELLS)
    p.add_argument("--cis_window_bp", type=int, default=data.DEFAULT_CIS_WINDOW_BP)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Adjusted p-value cutoff for DE relevant genes.")
    p.add_argument("--min_abs_lfc", type=float, default=0.0,
                   help="Minimum |log fold-change| for DE relevant genes.")
    p.add_argument("--top_n", type=int, default=None,
                   help="If set, keep only the top-N DE genes by |logFC| per perturbation.")
    p.add_argument("--de_method", default="wilcoxon")
    p.add_argument("--no_relevant", action="store_true",
                   help="Skip the (slow) DE relevant-gene step.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csd = data.prepare(
        cache_h5ad=(args.raw_h5ad or None),
        filtered_h5ad=(args.filtered_h5ad or None),
        min_cells=args.min_cells,
        cis_window_bp=args.cis_window_bp,
        alpha=args.alpha,
        min_abs_lfc=args.min_abs_lfc,
        top_n=args.top_n,
        de_method=args.de_method,
        compute_relevant=not args.no_relevant,
    )

    # --- cache tables ---------------------------------------------------------
    csd.pb.to_parquet(os.path.join(args.out_dir, "pseudobulk.parquet"))
    csd.delta.to_parquet(os.path.join(args.out_dir, "delta.parquet"))
    csd.knockdown_qc.to_csv(os.path.join(args.out_dir, "knockdown_check.csv"), index=False)
    csd.coords.to_parquet(os.path.join(args.out_dir, "coords.parquet"))

    if csd.relevant_genes:
        long = pd.DataFrame(
            [(P, g) for P, gs in csd.relevant_genes.items() for g in gs],
            columns=["pert", "gene"],
        )
        long.to_parquet(os.path.join(args.out_dir, "relevant_genes.parquet"))
        # Also a compact json map for quick programmatic reuse.
        with open(os.path.join(args.out_dir, "relevant_genes.json"), "w") as fh:
            json.dump(csd.relevant_genes, fh)

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
    if csd.relevant_genes:
        tot = sum(len(v) for v in csd.relevant_genes.values())
        print(f"trans (pert,gene) pairs: {tot}")
    print(f"artifacts written to   : {args.out_dir}/")


if __name__ == "__main__":
    main()
