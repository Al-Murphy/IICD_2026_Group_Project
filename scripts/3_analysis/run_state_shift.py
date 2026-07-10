#!/usr/bin/env python3
"""
Step 3 -- Does AlphaGenome's accuracy degrade when the cell state shifts?

Thin CLI over :mod:`concept_shift.state_shift`. For each of the 1,971 perturbed
states, compute Spearman(AG, measured pseudobulk) over a gene set that excludes
the perturbed gene (and optionally its cis neighbours), compare against the
control on the *same* genes, and correct for the cell-count confound with a
matched null (control subsampled to each state's own cell count).

Spearman is monotone-invariant, so AG needs no calibration onto the expression
scale -- no shrinkage, no heteroscedasticity artifacts.

Gene filtering
--------------
Global (one list for every state):

    --expr_threshold 0.1 --require_expressed_in_control   expressed in control
    --expr_threshold 0.01 --min_states_expressed 1971     expressed in EVERY state
    --min_mean_expr 0.05                                  drop weakly-expressed genes
    --gene_whitelist genes.txt                            a pathway / regulatory network

.. note::
   ``--expr_threshold 0`` drops nothing: this screen's gene set is already
   expression-filtered (no gene is 0 anywhere; min control pseudobulk = 0.022).
   "Expressed in >=1 state" and "expressed in control" are therefore no-ops
   unless you pass a positive threshold. Reference counts (8,545 genes total):
   >0.05 in control -> 8,519; >0.1 in control -> 6,875; >0.01 in all states -> 6,995.

Per-state (a different gene set per perturbation, e.g. the regulatory network of
the perturbed gene) is supported by the library:

    from concept_shift import state_shift as ss
    res, floor = ss.compute_state_shift(..., per_state_genes={"GATA1": [...], ...})

which automatically switches the matched null to ``per_state`` -- the gene sets
differ too much between states to share one null.

Outputs: out/ag_state_shift.parquet, out/rho_noise_floor.parquet,
         results/plots/ag_state_shift.png

Usage
-----
    python scripts/3_analysis/run_state_shift.py
    python scripts/3_analysis/run_state_shift.py --min_states_expressed 1
    python scripts/3_analysis/run_state_shift.py --gene_whitelist my_network.txt
"""

import argparse
import os

from scipy.stats import pearsonr, spearmanr

from concept_shift import state_shift as ss


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-cell-state Spearman of AlphaGenome vs measured expression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", default="./out/ag_k562_baseline.parquet")
    p.add_argument("--pseudobulk", default="./out/pseudobulk.parquet")
    p.add_argument("--coords", default="./out/coords.parquet")
    p.add_argument("--knockdown_csv", default="./out/knockdown_check.csv")
    p.add_argument("--wasserstein", default="./out/wasserstein_distance.parquet")
    p.add_argument("--h5ad", default="./out/replogle_k562_filtered.h5ad",
                   help="Needed for cell counts and the matched null.")
    p.add_argument("--proxy", default="rna_pred", choices=["rna_pred", "cage_pred"],
                   help="rna_pred wins the within-state control (0.666 vs 0.339): the "
                        "measured observable is scRNA-seq expression, not TSS initiation.")

    # --- global gene filtering ---
    p.add_argument("--min_states_expressed", type=int, default=0,
                   help="Keep genes expressed (> --expr_threshold) in >= this many states. "
                        "Use 1 to drop genes unexpressed in every state, or 1971 to demand "
                        "expression everywhere.")
    p.add_argument("--expr_threshold", type=float, default=0.0,
                   help="'Expressed' means pseudobulk strictly above this. NOTE: at 0.0 nothing "
                        "is dropped -- this screen's gene set is already expression-filtered "
                        "(min control pseudobulk 0.022). Use e.g. 0.1 for a real filter.")
    p.add_argument("--include_control_as_state", action="store_true",
                   help="Count the control row as a state for the two filters above.")
    p.add_argument("--require_expressed_in_control", action="store_true",
                   help="Keep only genes expressed (> --expr_threshold) in the control state. "
                        "Natural when AG's baseline accuracy is the reference: a gene silent in "
                        "control only adds noise to rho_ctrl.")
    p.add_argument("--min_mean_expr", type=float, default=0.0,
                   help="Keep genes whose mean pseudobulk across states is >= this.")
    p.add_argument("--gene_whitelist", default=None,
                   help="File with one gene per line; restrict to these genes.")

    # --- per-state exclusions ---
    p.add_argument("--no_exclude_target", dest="exclude_target", action="store_false",
                   help="Keep each state's own perturbed gene (default: excluded).")
    p.add_argument("--no_exclude_cis", dest="exclude_cis", action="store_false",
                   help="Keep cis neighbours (default: excluded).")
    p.add_argument("--cis_window_bp", type=int, default=2_000_000)

    # --- matched null ---
    p.add_argument("--n_boot", type=int, default=20,
                   help="Control subsamples per distinct cell count (matched null).")
    p.add_argument("--null_mode", default=None, choices=["by_n", "per_state"],
                   help="Default by_n; auto-switches to per_state for per-state gene sets.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--out", default="./out/ag_state_shift.parquet")
    p.add_argument("--floor_out", default="./out/rho_noise_floor.parquet")
    p.add_argument("--plot", default="./results/plots/ag_state_shift.png")
    return p.parse_args()


def main():
    args = parse_args()
    inp = ss.load_inputs(args.baseline, args.pseudobulk, args.coords,
                         args.knockdown_csv, args.wasserstein, args.h5ad)

    whitelist = None
    if args.gene_whitelist:
        with open(args.gene_whitelist) as fh:
            whitelist = [ln.strip() for ln in fh if ln.strip()]

    genes = ss.filter_genes(
        inp["pb"], inp["base"],
        expr_threshold=args.expr_threshold,
        min_states_expressed=args.min_states_expressed,
        include_control_as_state=args.include_control_as_state,
        require_expressed_in_control=args.require_expressed_in_control,
        min_mean_expr=args.min_mean_expr,
        whitelist=whitelist,
    )
    print(f"gene set: {len(genes)} genes  (expr_threshold={args.expr_threshold}, "
          f"min_states_expressed={args.min_states_expressed}, "
          f"require_expressed_in_control={args.require_expressed_in_control}, "
          f"min_mean_expr={args.min_mean_expr}, whitelist={bool(whitelist)})")

    res, floor = ss.compute_state_shift(
        inp["base"], inp["pb"], inp["coords"], inp["kd"], inp["wass"], inp["adata"],
        proxy=args.proxy, genes=genes,
        exclude_target=args.exclude_target, exclude_cis=args.exclude_cis,
        cis_window_bp=args.cis_window_bp,
        n_boot=args.n_boot, null_mode=args.null_mode, seed=args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_parquet(args.out)
    floor.to_frame().to_parquet(args.floor_out)

    W = res["wasserstein"].to_numpy()
    print(f"\nn={len(res)} states, {res.attrs['n_genes']} genes, "
          f"null_mode={res.attrs['null_mode']}\n")
    print(f"  rho control   = {res.rho_ctrl.mean():.4f}")
    print(f"  rho perturbed = {res.rho_pert.mean():.4f}")
    print(f"  cells/state: min={res.n_cells.min()} median={int(res.n_cells.median())} "
          f"max={res.n_cells.max()}")
    print(f"\n  CONFOUND  Spearman(n_cells, d_rho)       = "
          f"{spearmanr(res.n_cells, res.d_rho).statistic: .3f}   <- fewer cells, lower rho for free")
    print(f"            Spearman(n_cells, wasserstein) = "
          f"{spearmanr(res.n_cells, W).statistic: .3f}   <- but n_cells barely tracks strength")
    print(f"\n  mean d_rho raw     = {res.d_rho.mean():+.4f}  "
          f"({100*(res.d_rho>0).mean():.1f}% degrade)")
    print(f"  mean d_rho MATCHED = {res.d_rho_matched.mean():+.4f}  "
          f"({100*(res.d_rho_matched>0).mean():.1f}% degrade)   <- most of the raw drop was noise")
    print(f"  states with z > 2  : {100*(res.z>2).mean():.1f}%       z > 3: {100*(res.z>3).mean():.1f}%")
    print(f"\n  Spearman(d_rho raw,     W) = {spearmanr(res.d_rho, W).statistic: .3f}")
    print(f"  Spearman(d_rho MATCHED, W) = {spearmanr(res.d_rho_matched, W).statistic: .3f}")
    print(f"  Pearson (d_rho MATCHED, W) = {pearsonr(res.d_rho_matched, W).statistic: .3f}")
    print(f"\n  effect size: {res.d_rho_matched.mean():.4f} on a base of "
          f"{res.rho_ctrl.mean():.3f} = "
          f"{100*res.d_rho_matched.mean()/res.rho_ctrl.mean():.2f}% relative")

    import matplotlib
    matplotlib.use("Agg")
    fig = ss.plot_state_shift(res, floor)
    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    fig.savefig(args.plot, dpi=200, bbox_inches="tight")
    print(f"\nwrote {args.out}, {args.floor_out} and {args.plot}")


if __name__ == "__main__":
    main()
