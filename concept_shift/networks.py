"""
concept_shift.networks
======================

Gene networks for the perturbed genes, to score the concept shift on each state's
**own regulatory / functional neighbourhood** rather than on all 8,545 genes.

Feeds :func:`concept_shift.state_shift.compute_state_shift` via its
``per_state_genes`` argument, which auto-switches the matched null to
``null_mode="per_state"`` (the gene sets differ too much between states to share
one null).

Why STRING
----------
Only **162 of the 1,791** mapped targets are transcription factors (Lambert 2018),
so TF-regulon resources (DoRothEA, CollecTRI, ChIP-seq) cover ~9% of the screen.
The rest are machinery: ribosome (77), mito-ribosome (57), proteasome (34),
translation factors (34), spliceosome (31), RNA pol (22), exosome (11),
Integrator (9). A *functional* network covers all of them.

STRING is a structure/function prior, **not** derived from perturbation responses,
so restricting to it does not leak the answer (contrast: a network derived from
Perturb-seq responses trivially contains the responding genes). K562-specificity
comes from intersecting the network with the genes actually measured/expressed in
this screen -- see ``restrict_to``.

Neighbourhood sizes (targets of this screen, both edge ends measured):

    score >= 400 : median 139 neighbours, 1,707/1,791 targets with >=20
    score >= 700 : median  41 neighbours, 1,306/1,791 targets with >=20   <- default
    score >= 900 : median  18 neighbours,   834/1,791 targets with >=20

.. warning::
   A ~41-gene set gives ``sd(rho) ~ 0.16``. The per-state matched null absorbs that
   noise (it re-scores the null on each state's own mask), but per-state power is
   low -- read the aggregate, not individual states.

.. warning::
   Set size and expression level move rho on their own, independently of any
   concept shift. :func:`matched_background` draws size- and expression-matched
   random gene sets so you can report ``d_rho(network) - d_rho(background)``.
   It is opt-in, not the default.
"""

from __future__ import annotations

import gzip
import os
import urllib.request
from collections import defaultdict
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

__all__ = [
    "STRING_LINKS_URL",
    "STRING_INFO_URL",
    "download_string",
    "load_string_edges",
    "string_neighbours",
    "per_state_genes_from_network",
    "network_size_table",
    "matched_background",
]

STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/protein.links.v12.0/"
    "9606.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL = (
    "https://stringdb-downloads.org/download/protein.info.v12.0/"
    "9606.protein.info.v12.0.txt.gz"
)
# Cache the symbol-level edge list at this score; higher cutoffs filter in memory.
CACHE_MIN_SCORE = 400


def download_string(cache_dir: str = "./.cache/string") -> tuple[str, str]:
    """Download STRING v12 human links (~80 MB) + protein->symbol map (~2 MB)."""
    os.makedirs(cache_dir, exist_ok=True)
    links = os.path.join(cache_dir, os.path.basename(STRING_LINKS_URL))
    info = os.path.join(cache_dir, os.path.basename(STRING_INFO_URL))
    for url, dest in ((STRING_LINKS_URL, links), (STRING_INFO_URL, info)):
        if not os.path.exists(dest):
            print(f"[networks] downloading {os.path.basename(dest)} ...")
            urllib.request.urlretrieve(url, dest)
    return links, info


def load_string_edges(cache_dir: str = "./.cache/string",
                      min_score: int = CACHE_MIN_SCORE,
                      force: bool = False) -> pd.DataFrame:
    """Symbol-level STRING edge list ``[a, b, score]``, cached as parquet.

    The cache is built once at ``CACHE_MIN_SCORE``; stricter cutoffs are applied
    in memory by :func:`string_neighbours`. Edges are undirected and stored once.
    """
    cache = os.path.join(cache_dir, f"string_v12_symbol_edges_min{CACHE_MIN_SCORE}.parquet")
    if os.path.exists(cache) and not force:
        edges = pd.read_parquet(cache)
    else:
        links, info = download_string(cache_dir)
        p2s = pd.read_csv(info, sep="\t", usecols=[0, 1])
        p2s = dict(zip(p2s.iloc[:, 0], p2s.iloc[:, 1]))
        a_, b_, s_ = [], [], []
        print(f"[networks] parsing STRING links (min_score={CACHE_MIN_SCORE}) ...")
        with gzip.open(links, "rt") as fh:
            next(fh)
            for line in fh:
                a, b, s = line.split()
                score = int(s)
                if score < CACHE_MIN_SCORE:
                    continue
                sa, sb = p2s.get(a), p2s.get(b)
                if sa is None or sb is None or sa == sb:
                    continue
                if sa > sb:                       # store each undirected edge once
                    sa, sb = sb, sa
                a_.append(sa); b_.append(sb); s_.append(score)
        edges = (pd.DataFrame({"a": a_, "b": b_, "score": np.asarray(s_, dtype=np.int16)})
                 .drop_duplicates(subset=["a", "b"]))
        os.makedirs(cache_dir, exist_ok=True)
        edges.to_parquet(cache, index=False)
        print(f"[networks] cached {len(edges):,} symbol edges -> {cache}")

    return edges[edges["score"] >= min_score] if min_score > CACHE_MIN_SCORE else edges


def string_neighbours(edges: pd.DataFrame, min_score: int = 700,
                      restrict_to: Optional[Iterable[str]] = None) -> dict:
    """``{gene: set(neighbours)}`` at ``min_score``.

    ``restrict_to`` keeps only edges whose **both** ends are in that set -- pass
    the measured / AG-scored genes to get the K562-active subnetwork.
    """
    e = edges[edges["score"] >= min_score]
    if restrict_to is not None:
        keep = set(restrict_to)
        e = e[e["a"].isin(keep) & e["b"].isin(keep)]
    nb = defaultdict(set)
    for a, b in zip(e["a"].to_numpy(), e["b"].to_numpy()):
        if a == b:                    # defensive: self-loops are never neighbours
            continue
        nb[a].add(b)
        nb[b].add(a)
    return dict(nb)


def per_state_genes_from_network(neighbours: Mapping[str, Iterable[str]],
                                 target_map: Mapping[str, str],
                                 measured_genes: Iterable[str], *,
                                 min_genes: int = 20,
                                 max_genes: Optional[int] = None,
                                 edges: Optional[pd.DataFrame] = None,
                                 min_score: int = 700) -> dict:
    """``{state: [network genes]}`` for states whose target has >= ``min_genes`` neighbours.

    The target itself is not removed here -- :func:`state_shift.state_gene_masks`
    drops each state's own perturbed gene (and cis neighbours) afterwards, so the
    exclusion stays in one place.

    ``max_genes`` keeps the top-scoring neighbours (needs ``edges``), which is the
    principled way to cap huge hubs rather than truncating arbitrarily.
    """
    measured = set(measured_genes)
    top_by_score = None
    if max_genes is not None:
        if edges is None:
            raise ValueError("max_genes needs `edges` to rank neighbours by score")
        e = edges[edges["score"] >= min_score]
        top_by_score = e

    out = {}
    for state, target in target_map.items():
        if not isinstance(target, str):
            continue                                   # target gene not measured
        nb = [g for g in neighbours.get(target, ()) if g in measured]
        if max_genes is not None and len(nb) > max_genes:
            e = top_by_score
            sub = e[((e["a"] == target) & e["b"].isin(nb)) |
                    ((e["b"] == target) & e["a"].isin(nb))].copy()
            sub["partner"] = np.where(sub["a"] == target, sub["b"], sub["a"])
            nb = sub.nlargest(max_genes, "score")["partner"].tolist()
        if len(nb) >= min_genes:
            out[state] = nb
    return out


def network_size_table(neighbours: Mapping[str, Iterable[str]],
                       target_map: Mapping[str, str],
                       measured_genes: Iterable[str]) -> pd.DataFrame:
    """Per-target neighbourhood size, for coverage diagnostics."""
    measured = set(measured_genes)
    rows = [(s, t, len([g for g in neighbours.get(t, ()) if g in measured]))
            for s, t in target_map.items() if isinstance(t, str)]
    return pd.DataFrame(rows, columns=["state", "target", "n_network_genes"]).set_index("state")


def matched_background(per_state_genes: Mapping[str, Iterable[str]],
                       pb: pd.DataFrame, measured_genes: Iterable[str], *,
                       ctrl: str = "control", n_bins: int = 10,
                       seed: int = 0) -> dict:
    """Random gene sets matched to each state's network on **size and expression decile**.

    Opt-in control. Set size and expression level shift rho on their own, so
    ``d_rho(network) - d_rho(background)`` isolates what is specific to the
    network. Draws one background set per state.
    """
    rng = np.random.default_rng(seed)
    genes = [g for g in measured_genes if g in pb.columns]
    expr = pb.loc[ctrl, genes]
    # Rank-based deciles so bins are equally populated.
    bins = pd.qcut(expr.rank(method="first"), n_bins, labels=False).to_numpy()
    by_bin = {b: np.array([g for g, bb in zip(genes, bins) if bb == b], dtype=object)
              for b in range(n_bins)}
    bin_of = dict(zip(genes, bins))
    expr_rank = expr.rank(method="first").to_dict()

    out = {}
    for state, net in per_state_genes.items():
        net = list(net)
        net_set = set(net)
        want = defaultdict(int)
        for g in net:
            if g in bin_of:
                want[bin_of[g]] += 1

        pick, short = [], 0
        for b, k in want.items():
            pool = np.array([g for g in by_bin[b] if g not in net_set], dtype=object)
            take = min(k, len(pool))
            if take:
                pick.extend(rng.choice(pool, size=take, replace=False).tolist())
            short += k - take                     # decile exhausted by the network

        if short:
            # Top up with the nearest-expression genes outside the network, so the
            # background always MATCHES THE NETWORK'S SIZE (that is the whole point).
            chosen = set(pick) | net_set
            target_rank = np.mean([expr_rank[g] for g in net if g in expr_rank])
            cand = sorted((g for g in genes if g not in chosen),
                          key=lambda g: abs(expr_rank[g] - target_rank))
            pick.extend(cand[:short])
        out[state] = pick
    return out
