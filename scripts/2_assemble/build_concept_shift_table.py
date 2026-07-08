#!/usr/bin/env python3
"""
Step 2 -- Assemble the concept-shift table (spec step 8).

For every perturbation P and every gene g in its trans relevant set, the record
is (measured_delta, predicted_delta = 0, error = |measured_delta|). AlphaGenome
is state-blind so the predicted delta is 0 by construction -- this step never
calls AlphaGenome, it just joins the measured delta with the (implicit) zero.

Also emits ``perturbation_summary.parquet`` (per-perturbation rollup). The
AlphaGenome baseline is only needed here to attach the within-state context and
is optional at this stage (E-distance / within-state Spearman live in step 3).

Inputs (from step 0):  out/delta.parquet, out/relevant_genes.{parquet,json},
                       out/knockdown_check.csv
Outputs:               out/concept_shift_table.parquet
                       out/perturbation_summary.parquet

Usage
-----
    python scripts/2_assemble/build_concept_shift_table.py
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
    p.add_argument("--relevant_json", default="./out/relevant_genes.json")
    p.add_argument("--knockdown_csv", default="./out/knockdown_check.csv")
    p.add_argument("--pseudobulk", default="./out/pseudobulk.parquet")
    p.add_argument("--ctrl", default="control")
    p.add_argument("--out_table", default="./out/concept_shift_table.parquet")
    p.add_argument("--out_summary", default="./out/perturbation_summary.parquet")
    return p.parse_args()


def main():
    args = parse_args()

    delta = pd.read_parquet(args.delta)
    with open(args.relevant_json) as fh:
        relevant = json.load(fh)
    kd = pd.read_csv(args.knockdown_csv)
    pb = pd.read_parquet(args.pseudobulk)

    # --- concept-shift table --------------------------------------------------
    records = []
    for P, genes in relevant.items():
        if P == args.ctrl or P not in delta.index:
            continue
        d = delta.loc[P]
        for g in genes:
            if g not in d.index:
                continue
            m = float(d[g])
            records.append((P, g, m, 0.0, abs(m)))
    cs = pd.DataFrame(
        records, columns=["pert", "gene", "measured_delta", "pred_delta", "error"]
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_table)), exist_ok=True)
    cs.to_parquet(args.out_table)

    # --- per-perturbation summary --------------------------------------------
    n_cells = None  # cell counts live in the AnnData; filled by step 3 if desired.
    kd_map = kd.set_index("pert")
    grp = cs.groupby("pert")
    summary = pd.DataFrame({
        "n_relevant_genes": grp.size(),
        "mean_error": grp["error"].mean(),
        "median_error": grp["error"].median(),
    })
    summary = summary.join(kd_map[["target_var", "pct_change", "status", "knockdown_ok"]])
    summary.to_parquet(args.out_summary)

    print(f"[assemble] concept-shift rows : {len(cs)}")
    print(f"[assemble] perturbations       : {cs['pert'].nunique()}")
    print(f"[assemble] mean |measured delta|: {cs['error'].mean():.4f}")
    print(f"[assemble] wrote {args.out_table} and {args.out_summary}")


if __name__ == "__main__":
    main()
