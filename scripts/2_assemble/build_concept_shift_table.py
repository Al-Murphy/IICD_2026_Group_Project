#!/usr/bin/env python3
"""
Step 2 -- Assemble the concept-shift table (spec step 8).

For every perturbation P and every gene g in its comparison set the record is
(measured_delta, predicted_delta = 0, error = |measured_delta|). AlphaGenome is
state-blind, so its predicted delta is 0 by construction: this step never calls
AlphaGenome, it just joins the measured delta with that implicit zero.

Trans filtering is applied HERE, not baked into the saved matrices
--------------------------------------------------------------------
We care about *trans* effects, so for each perturbation we drop that
perturbation's OWN target gene (``--exclude_target``, default on) and,
optionally, its cis neighbours within +/- ``--cis_window_bp`` of the target TSS
(``--exclude_cis``, KRAB spreading). The gene universe is either:

  --gene_set all       every measured gene (default; nothing pre-filtered), or
  --gene_set relevant  the per-perturbation DE set from step 0 --relevant.

Because ``delta.parquet`` keeps all 1,971 x 8,563 measured values, every
filtering choice stays open downstream.

Inputs (step 0):  out/delta.parquet, out/knockdown_check.csv, out/coords.parquet
                  [out/relevant_genes.json], [out/wasserstein_distance.parquet]
Outputs:          out/concept_shift_table.parquet
                  out/perturbation_summary.parquet

Usage
-----
    python scripts/2_assemble/build_concept_shift_table.py
    python scripts/2_assemble/build_concept_shift_table.py --exclude_cis
    python scripts/2_assemble/build_concept_shift_table.py --gene_set relevant
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Assemble concept-shift table (predicted delta == 0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--delta", default="./out/delta.parquet")
    p.add_argument("--knockdown_csv", default="./out/knockdown_check.csv")
    p.add_argument("--coords", default="./out/coords.parquet")
    p.add_argument("--relevant_json", default="./out/relevant_genes.json")
    p.add_argument("--wasserstein", default="./out/wasserstein_distance.parquet")
    p.add_argument("--ctrl", default="control")
    p.add_argument("--gene_set", choices=["all", "relevant"], default="all",
                   help="Compare over every measured gene, or the per-pert DE set.")
    p.add_argument("--no_exclude_target", dest="exclude_target", action="store_false",
                   help="Keep each perturbation's own target gene (default: excluded).")
    p.add_argument("--exclude_cis", action="store_true",
                   help="Also drop cis neighbours within --cis_window_bp of the target TSS.")
    p.add_argument("--cis_window_bp", type=int, default=2_000_000)
    p.add_argument("--valid_coords_only", dest="valid_coords_only", action="store_true",
                   default=True,
                   help="Restrict to genes with valid hg38 coords (the AG-comparable set).")
    p.add_argument("--out_table", default="./out/concept_shift_table.parquet")
    p.add_argument("--out_summary", default="./out/perturbation_summary.parquet")
    return p.parse_args()


def main():
    args = parse_args()

    delta = pd.read_parquet(args.delta)
    kd = pd.read_csv(args.knockdown_csv)
    coords = pd.read_parquet(args.coords)

    perts = [p for p in delta.index if p != args.ctrl]
    genes = [g for g in delta.columns
             if (not args.valid_coords_only) or (g in coords.index)]

    relevant = None
    if args.gene_set == "relevant":
        if not os.path.exists(args.relevant_json):
            raise SystemExit("--gene_set relevant needs out/relevant_genes.json "
                             "(run step 0 with --relevant).")
        with open(args.relevant_json) as fh:
            relevant = json.load(fh)

    # Dense measured-delta block: (n_perts x n_genes). ~1971 x 8545 -> 135 MB f64.
    D = delta.loc[perts, genes].to_numpy(dtype=np.float64)
    gpos = {g: j for j, g in enumerate(genes)}
    mask = np.ones(D.shape, dtype=bool)

    if relevant is not None:
        mask[:] = False
        for i, P in enumerate(perts):
            for g in relevant.get(P, []):
                j = gpos.get(g)
                if j is not None:
                    mask[i, j] = True

    t2v = kd.set_index("pert")["target_var"].to_dict()
    g_chr = coords.loc[genes, "chr"].astype(str).to_numpy()
    g_tss = coords.loc[genes, "tss"].astype(np.int64).to_numpy()

    n_target_dropped = n_cis_dropped = 0
    for i, P in enumerate(perts):
        tv = t2v.get(P)
        if not isinstance(tv, str):          # target_not_measured -> NaN in the csv
            continue
        if args.exclude_target:
            j = gpos.get(tv)
            if j is not None and mask[i, j]:
                mask[i, j] = False
                n_target_dropped += 1
        if args.exclude_cis and tv in coords.index:
            t_chr = str(coords.loc[tv, "chr"])
            t_tss = int(coords.loc[tv, "tss"])
            cis = (g_chr == t_chr) & (np.abs(g_tss - t_tss) <= args.cis_window_bp)
            n_cis_dropped += int((mask[i] & cis).sum())
            mask[i] &= ~cis

    rows, cols = np.nonzero(mask)
    measured = D[rows, cols]
    cs = pd.DataFrame({
        "pert": pd.Categorical(np.asarray(perts, dtype=object)[rows]),
        "gene": pd.Categorical(np.asarray(genes, dtype=object)[cols]),
        "measured_delta": measured,
        "pred_delta": np.zeros(measured.shape[0], dtype=np.float64),
        "error": np.abs(measured),
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out_table)), exist_ok=True)
    cs.to_parquet(args.out_table, index=False)

    # --- per-perturbation summary --------------------------------------------
    n_genes = mask.sum(axis=1)
    err = np.where(mask, np.abs(D), np.nan)
    summary = pd.DataFrame({
        "pert": perts,
        "n_genes": n_genes,
        "mean_error": np.nanmean(err, axis=1),
        "median_error": np.nanmedian(err, axis=1),
    }).set_index("pert")
    summary = summary.join(
        kd.set_index("pert")[["target_var", "pct_change", "status", "knockdown_ok"]]
    )
    if os.path.exists(args.wasserstein):
        w = pd.read_parquet(args.wasserstein)
        summary = summary.join(w, how="left")
    summary.to_parquet(args.out_summary)

    print(f"[assemble] gene universe      : {len(genes)} genes "
          f"({'valid-coord' if args.valid_coords_only else 'all measured'}), "
          f"gene_set={args.gene_set}")
    print(f"[assemble] trans filter       : exclude_target={args.exclude_target} "
          f"({n_target_dropped} dropped), exclude_cis={args.exclude_cis} "
          f"({n_cis_dropped} dropped)")
    print(f"[assemble] concept-shift rows : {len(cs):,}")
    print(f"[assemble] perturbations      : {len(perts)}")
    print(f"[assemble] genes/pert (mean)  : {n_genes.mean():.1f}")
    print(f"[assemble] mean |measured Δ|   : {cs['error'].mean():.4f}")
    print(f"[assemble] wrote {args.out_table} and {args.out_summary}")


if __name__ == "__main__":
    main()
