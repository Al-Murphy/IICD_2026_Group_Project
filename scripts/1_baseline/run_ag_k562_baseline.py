#!/usr/bin/env python3
"""
Step 1 -- AlphaGenome K562 baseline: ONE prediction per gene (spec step 7).

AlphaGenome is state-blind, so this is computed once per gene and reused across
all perturbations. A single forward pass yields both proxies:

    cage_pred  (PRIMARY)   sense-strand CAGE/PRO-cap integrated over TSS +/- window
    rna_pred   (SECONDARY) sense-strand RNA-Seq integrated over the gene body

Writes ``out/ag_k562_baseline.parquet`` (index = gene var_name). Resumable:
re-running skips genes already present. Shardable for SLURM arrays.

Requires the ``[alphagenome]`` extra (torch, pysam, tangermeme, alphagenome-pytorch).

Usage
-----
    # Full run (downloads hg38 + AG weights on first use)
    python scripts/1_baseline/run_ag_k562_baseline.py

    # One shard of an 8-way SLURM array
    python scripts/1_baseline/run_ag_k562_baseline.py \
        --num_shards 8 --shard_idx $SLURM_ARRAY_TASK_ID \
        --output out/ag_k562_baseline_shard${SLURM_ARRAY_TASK_ID}of8.parquet
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from concept_shift import ag_backbone
from concept_shift.seq import SeqLoader


def parse_args():
    p = argparse.ArgumentParser(
        description="AlphaGenome K562 baseline (one prediction per gene).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coords", default="./out/coords.parquet",
                   help="Strand-aware coord table from step 0 (index=var_name).")
    p.add_argument("--output", default="./out/ag_k562_baseline.parquet")
    p.add_argument("--metadata_path", default="./metadata/track_metadata.parquet")
    p.add_argument("--backbone_model_path", default=None,
                   help="Local AG weights. If omitted, downloaded from HuggingFace.")
    p.add_argument("--genome_build", default="hg38", choices=["hg38", "hg19"])
    p.add_argument("--genome_cache", default="./.cache")
    p.add_argument("--cage_window_bp", type=int, default=ag_backbone.DEFAULT_CAGE_WINDOW_BP)
    p.add_argument("--organism_index", type=int, default=0, help="0=human, 1=mouse.")
    p.add_argument("--device", default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--save_every", type=int, default=200,
                   help="Checkpoint the parquet every N genes.")
    args = p.parse_args()
    if not (0 <= args.shard_idx < args.num_shards):
        p.error("--shard_idx must be in [0, num_shards)")
    return args


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    coords = pd.read_parquet(args.coords)
    genes = coords.index.tolist()[args.shard_idx::args.num_shards]

    # Resume: skip genes already in the output parquet.
    done = {}
    if os.path.exists(args.output):
        prev = pd.read_parquet(args.output)
        done = prev.to_dict("index")
        print(f"[baseline] resuming; {len(done)} genes already predicted")
    todo = [g for g in genes if g not in done]
    print(f"[baseline] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(todo)} genes to predict ({len(genes)} in shard)")
    if not todo:
        print("[baseline] nothing to do")
        return

    model, device = ag_backbone.load_model(args.backbone_model_path, args.device)
    print(f"[baseline] device: {device}")
    loader = SeqLoader(args.genome_build, ag_backbone.AG_SEQ_LEN, lcl_path=args.genome_cache)

    cage_bundle, cage_desc = ag_backbone.k562_track_indices(args.metadata_path, "cage")
    rna_bundle, rna_desc = ag_backbone.k562_track_indices(args.metadata_path, "rna_seq")
    print(f"[baseline] CAGE tracks: {cage_desc}")
    print(f"[baseline] RNA  tracks: {rna_desc}")

    records = dict(done)

    def flush():
        pd.DataFrame(records).T.rename_axis("gene").to_parquet(args.output)

    for i, g in enumerate(tqdm(todo, desc="genes")):
        row = coords.loc[g]
        try:
            cage, rna = ag_backbone.predict_gene(
                model, loader, device,
                chrom=str(row["chr"]), tss=int(row["tss"]), strand=str(row["strand"]),
                gstart=int(row["start"]), gend=int(row["end"]),
                cage_bundle=cage_bundle, rna_bundle=rna_bundle,
                cage_window_bp=args.cage_window_bp, organism_index=args.organism_index,
            )
        except Exception as exc:  # noqa: BLE001 -- keep the sweep alive; log the gene
            print(f"[baseline] WARN gene {g}: {exc}")
            cage, rna = np.nan, np.nan
        records[g] = {"cage_pred": cage, "rna_pred": rna}
        if (i + 1) % args.save_every == 0:
            flush()

    flush()
    print(f"[baseline] wrote {len(records)} gene predictions -> {args.output}")


if __name__ == "__main__":
    main()
