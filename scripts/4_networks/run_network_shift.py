#!/usr/bin/env python3
"""
Step 4 -- Concept shift on each state's own gene network (STRING).

Instead of scoring AlphaGenome against all 8,545 genes, score it against the
**functional neighbourhood of the perturbed gene** in that state. Hypothesis: the
concept shift is concentrated in the perturbed gene's network, so d_rho should be
larger there than on the transcriptome at large.

Network
-------
STRING v12 human, combined score >= ``--min_score``, restricted to edges whose
both ends are measured/AG-scored in this screen (the K562-active subnetwork).
STRING is a structure/function prior, not derived from perturbation responses, so
restricting to it does not leak the answer.

Only 162/1,791 targets are TFs, so TF-regulon resources would cover ~9% of the
screen; a functional network covers essentially all of it.

Neighbourhood sizes: score>=700 gives a median of 41 genes per target and 1,306
targets with >=20. A 41-gene rho has sd ~0.16, so per-state power is low -- read
the aggregate. The per-state matched null (control subsampled to each state's own
cell count AND scored on that state's own mask) absorbs both the cell-count and
the set-size noise.

Caveat (opt-in control)
-----------------------
Set size and expression level move rho on their own. ``--matched_background``
additionally scores size- and expression-decile-matched random gene sets, so you
can report ``d_rho(network) - d_rho(background)``. Off by default.

Outputs: out/ag_network_shift.parquet
         out/ag_network_sizes.parquet
         results/plots/ag_network_shift.png

Usage
-----
    python scripts/4_networks/run_network_shift.py
    python scripts/4_networks/run_network_shift.py --min_score 400 --min_genes 50
    python scripts/4_networks/run_network_shift.py --matched_background
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from concept_shift import networks as nw
from concept_shift import state_shift as ss


def parse_args():
    p = argparse.ArgumentParser(
        description="Concept shift scored on each perturbed gene's STRING neighbourhood.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", default="./out/ag_k562_baseline.parquet")
    p.add_argument("--pseudobulk", default="./out/pseudobulk.parquet")
    p.add_argument("--coords", default="./out/coords.parquet")
    p.add_argument("--knockdown_csv", default="./out/knockdown_check.csv")
    p.add_argument("--wasserstein", default="./out/wasserstein_distance.parquet")
    p.add_argument("--h5ad", default="./out/replogle_k562_filtered.h5ad")
    p.add_argument("--proxy", default="rna_pred", choices=["rna_pred", "cage_pred"])
    p.add_argument("--network", default="string",
                   choices=["string", "gwps_k562", "gwps_rpe1"],
                   help="string: physical/functional prior (non-circular). "
                        "gwps_k562: empirical downstream response from Replogle's "
                        "independent genome-wide screen, SAME cell line (circular by "
                        "construction -> positive control). gwps_rpe1: same, but a "
                        "DIFFERENT cell line -> cell-type-specificity control.")
    p.add_argument("--top_n", type=int, default=100,
                   help="gwps_*: take the top-N genes by |z| as the downstream network.")
    p.add_argument("--gwps_cache", default="./.cache/gwps")
    p.add_argument("--string_cache", default="./.cache/string")
    p.add_argument("--min_score", type=int, default=700,
                   help="STRING combined score cutoff (400 loose / 700 default / 900 strict).")
    p.add_argument("--min_genes", type=int, default=20,
                   help="Drop states whose target has fewer network genes than this.")
    p.add_argument("--max_genes", type=int, default=None,
                   help="Cap huge hubs at the top-N highest-scoring neighbours.")
    p.add_argument("--n_boot", type=int, default=20,
                   help="Control subsamples per state for the per-state matched null.")
    p.add_argument("--matched_background", action="store_true",
                   help="Also score size/expression-matched random sets (opt-in control).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./out/ag_network_shift.parquet")
    p.add_argument("--sizes_out", default="./out/ag_network_sizes.parquet")
    p.add_argument("--plot", default="./results/plots/ag_network_shift.png")
    return p.parse_args()


def _run(inp, genes, per_state_genes, args, seed):
    """compute_state_shift on the states covered by `per_state_genes`."""
    states = list(per_state_genes)
    pb_sub = inp["pb"].loc[["control"] + states]
    res, floor = ss.compute_state_shift(
        inp["base"], pb_sub, inp["coords"], inp["kd"], inp["wass"], inp["adata"],
        proxy=args.proxy, genes=genes, per_state_genes=per_state_genes,
        n_boot=args.n_boot, seed=seed,
    )
    return res, floor


def main():
    args = parse_args()
    inp = ss.load_inputs(args.baseline, args.pseudobulk, args.coords,
                         args.knockdown_csv, args.wasserstein, args.h5ad)
    genes = ss.filter_genes(inp["pb"], inp["base"])
    target_map = inp["kd"].set_index("pert")["target_var"].to_dict()

    if args.network == "string":
        edges = nw.load_string_edges(args.string_cache)
        nb = nw.string_neighbours(edges, min_score=args.min_score, restrict_to=genes)
        sizes = nw.network_size_table(nb, target_map, genes)
        per_state_genes = nw.per_state_genes_from_network(
            nb, target_map, genes, min_genes=args.min_genes,
            max_genes=args.max_genes, edges=edges, min_score=args.min_score,
        )
        print(f"STRING >= {args.min_score}: {len(nb)} genes in the K562-active subnetwork")
    else:
        which = args.network.split("_")[1]
        path = nw.download_gwps_bulk(which, args.gwps_cache)
        z = nw.load_gwps_zscores(path, restrict_to=genes)
        per_state_genes = nw.gwps_downstream_sets(
            z, target_map, genes, top_n=args.top_n, min_genes=args.min_genes)
        sizes = pd.DataFrame(
            {"n_network_genes": {s: len(v) for s, v in per_state_genes.items()}}
        ).rename_axis("state")
        print(f"GWPS ({which}): {z.shape[0]} perturbations x {z.shape[1]} genes; "
              f"top_n={args.top_n} by |z|")

    os.makedirs(os.path.dirname(os.path.abspath(args.sizes_out)), exist_ok=True)
    sizes.to_parquet(args.sizes_out)
    print(f"  network size: median {int(sizes.n_network_genes.median())}  "
          f"max {int(sizes.n_network_genes.max())}")
    print(f"  states kept (>= {args.min_genes} network genes): {len(per_state_genes)}/{len(sizes)}")

    res, floor = _run(inp, genes, per_state_genes, args, args.seed)
    res["n_network_genes"] = sizes["n_network_genes"].reindex(res.index)
    res = res.dropna(subset=["rho_pert", "d_rho_matched"])

    if args.matched_background:
        bg = nw.matched_background(per_state_genes, inp["pb"], genes, seed=args.seed)
        bg = {s: g for s, g in bg.items() if len(g) >= 3}
        res_bg, _ = _run(inp, genes, bg, args, args.seed + 1)
        res["d_rho_matched_bg"] = res_bg["d_rho_matched"].reindex(res.index)
        res["d_rho_net_minus_bg"] = res["d_rho_matched"] - res["d_rho_matched_bg"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_parquet(args.out)

    # --- report ---------------------------------------------------------------
    W = res["wasserstein"].to_numpy()
    print(f"\nn={len(res)} states, network genes/state: median "
          f"{int(res.n_network_genes.median())}\n")
    print(f"  rho control (network genes) = {res.rho_ctrl.mean():.4f}")
    print(f"  rho perturbed               = {res.rho_pert.mean():.4f}")
    print(f"  mean d_rho raw              = {res.d_rho.mean():+.4f}")
    print(f"  mean d_rho MATCHED          = {res.d_rho_matched.mean():+.4f}  "
          f"({100*(res.d_rho_matched>0).mean():.1f}% degrade)")
    print(f"  states with z > 2           : {100*(res.z>2).mean():.1f}%")
    print(f"\n  Spearman(d_rho MATCHED, W)  = {spearmanr(res.d_rho_matched, W).statistic: .3f}")

    if args.matched_background:
        d = res["d_rho_net_minus_bg"].dropna()
        stat, p = wilcoxon(d)
        print(f"\n  BACKGROUND CONTROL (size + expression-decile matched):")
        print(f"    d_rho background      = {res.d_rho_matched_bg.mean():+.4f}")
        print(f"    network - background  = {d.mean():+.4f}  "
              f"(Wilcoxon p={p:.2e}, {100*(d>0).mean():.1f}% positive)")

    # --- plot -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    warm, cool, grey = "#d1495b", "#2e6f95", "#9aa5ad"
    ncol = 3 if args.matched_background else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.2))

    ax = axes[0]
    sns.histplot(res.n_network_genes, bins=40, color=grey, ax=ax)
    ax.set(xlabel="network genes per state", ylabel="states",
           title=f"{args.network} network size")

    ax = axes[1]
    sig = (res.z > 2).to_numpy()
    rho = spearmanr(res.d_rho_matched, W).statistic
    ax.scatter(W[~sig], res.d_rho_matched[~sig], s=13, alpha=.35, color=grey,
               edgecolor="none", label=r"n.s. ($z\leq2$)")
    ax.scatter(W[sig], res.d_rho_matched[sig], s=15, alpha=.65, color=cool,
               edgecolor="none", label=f"$z>2$ ({100*sig.mean():.0f}%)")
    ax.axhline(0, color="k", lw=.8, ls="--")
    ax.set(xlabel="Wasserstein", ylabel=r"$\Delta\rho$ (network genes)",
           title=f"Concept shift on the network\nSpearman = {rho:.2f}")
    ax.legend(frameon=False, fontsize=8)

    if args.matched_background:
        ax = axes[2]
        d = res[["d_rho_matched", "d_rho_matched_bg"]].dropna()
        ax.scatter(d.d_rho_matched_bg, d.d_rho_matched, s=13, alpha=.45,
                   color=warm, edgecolor="none")
        lo = float(min(d.min())); hi = float(max(d.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(xlabel=r"$\Delta\rho$ matched background",
               ylabel=r"$\Delta\rho$ network",
               title="Above the line = network-specific")

    for a in np.atleast_1d(axes):
        sns.despine(ax=a, top=True, right=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    fig.savefig(args.plot, dpi=200, bbox_inches="tight")
    print(f"\nwrote {args.out}, {args.sizes_out} and {args.plot}")


if __name__ == "__main__":
    main()
