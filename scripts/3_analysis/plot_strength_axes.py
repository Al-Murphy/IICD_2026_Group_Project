#!/usr/bin/env python3
"""
Plot AlphaGenome's per-state concept shift against one or more strength axes
(e.g. PCA vs scVI Wasserstein distance).

Thin CLI over ``concept_shift.state_shift.{compare_strength_axes, plot_strength_axes}``.
Reads the step-3 per-state result and one Wasserstein Series per axis.

``--sign``
    +1  y = d_rho = rho_ctrl - rho_pert   (degradation; up = AG worse)
    -1  y = rho_pert - rho_ctrl           (signed performance change; a DROP is negative)

The y-axis is shared across panels, so its label is drawn on the leftmost panel
only; x-axes read "<axis> Wasserstein distance".

Usage
-----
    # default: PCA vs scVI, signed performance change (drop = negative)
    python scripts/3_analysis/plot_strength_axes.py

    # degradation orientation instead
    python scripts/3_analysis/plot_strength_axes.py --sign 1 --out results/plots/ag_shift_vs_PCA_scVI_allgenes.png

    # arbitrary axes (repeat --axis LABEL=path); e.g. add a network-set result
    python scripts/3_analysis/plot_strength_axes.py \
        --axis "PCA=out/wasserstein_distance.parquet" \
        --axis "scVI=out/scVI_wasserstein_distance.parquet"
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from concept_shift import state_shift as ss


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot AG concept shift vs strength axes (PCA / scVI Wasserstein).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--state_shift", default="./out/ag_state_shift.parquet",
                   help="Per-state result from scripts/3_analysis/run_state_shift.py.")
    p.add_argument("--axis", action="append", metavar="LABEL=PARQUET",
                   help="A strength axis as LABEL=path to a Wasserstein parquet "
                        "(column 'wasserstein'). Repeatable. Defaults to PCA + scVI.")
    p.add_argument("--shift_col", default="d_rho_matched")
    p.add_argument("--sign", type=int, default=-1, choices=[1, -1],
                   help="-1: signed performance change (drop = negative). "
                        "+1: degradation d_rho (up = AG worse).")
    p.add_argument("--out", default="./results/plots/ag_shift_vs_PCA_scVI_allgenes_neg.png")
    return p.parse_args()


def main():
    args = parse_args()
    pairs = args.axis or [
        "PCA=./out/wasserstein_distance.parquet",
        "scVI=./out/scVI_wasserstein_distance.parquet",
    ]
    axes = {}
    for pair in pairs:
        label, _, path = pair.partition("=")
        axes[label] = pd.read_parquet(path)["wasserstein"]

    res = pd.read_parquet(args.state_shift)
    print(ss.compare_strength_axes(res, axes, shift_col=args.shift_col).round(3).to_string())

    fig = ss.plot_strength_axes(res, axes, shift_col=args.shift_col, sign=args.sign)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {args.out}  (sign={args.sign:+d}, "
          f"{'signed performance change' if args.sign < 0 else 'degradation d_rho'})")


if __name__ == "__main__":
    main()
