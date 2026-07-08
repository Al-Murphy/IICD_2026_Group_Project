"""
concept_shift.seq
=================

Minimal hg38 genome fetching + one-hot encoding for the AlphaGenome baseline.

Vendored / trimmed from ``seq2func_crispri_eval.crispri_eval.dataset_utils``
(Alan Murphy) so this repo's AlphaGenome step is self-contained. Only the TSS-
centred fetch path needed by the state-blind baseline is kept.

Requires the ``[alphagenome]`` extra (``torch``, ``pysam``, ``tangermeme``).
"""

from __future__ import annotations

import gzip
import os
import urllib.request

_UCSC = {
    "hg19": "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
    "hg38": "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
}


def get_genome(build: str, lcl_path: str = "./.cache", force: bool = False) -> str:
    """Download a UCSC genome fasta if absent; return the local path."""
    build = build.lower()
    assert build in _UCSC, "build must be one of ['hg19','hg38']"
    os.makedirs(lcl_path, exist_ok=True)
    gen_pth = os.path.join(lcl_path, build + ".fa")
    if (not os.path.exists(gen_pth)) or force:
        print(f"[seq] downloading {build} genome (one-time, ~1GB compressed)...")
        urllib.request.urlretrieve(_UCSC[build], gen_pth + ".gz")
        with gzip.open(gen_pth + ".gz", "rb") as f_in, open(gen_pth, "wb") as f_out:
            f_out.write(f_in.read())
    return gen_pth


def reverse_complement(seq):
    """Reverse-complement a one-hot tensor ``(B, 4, L)`` via a flip on both axes."""
    import torch

    if seq.dim() == 2:
        seq = seq.unsqueeze(0)
    return torch.flip(seq, dims=[1, 2])


class SeqLoader:
    """TSS-centred one-hot sequence fetcher over a cached genome fasta.

    Parameters
    ----------
    build : str
        ``hg38`` (or ``hg19``).
    receptive_field : int
        Model input length in bp (AlphaGenome pytorch port = 1,048,576).
    lcl_path : str
        Genome cache directory.
    """

    def __init__(self, build: str, receptive_field: int, lcl_path: str = "./.cache"):
        import pysam

        assert receptive_field > 0
        self.genome = pysam.Fastafile(get_genome(build, lcl_path=lcl_path))
        self.rf = receptive_field
        self.mod = receptive_field % 2

    def chrom_len(self, chrom: str) -> int:
        return self.genome.get_reference_length(chrom)

    def get_seq(self, chrom: str, tss: int, strand: str, ohe: bool = True):
        """One-hot ``(1, 4, rf)`` window centred on ``tss``, strand-corrected.

        On the minus strand the returned sequence is reverse-complemented so the
        gene reads 5'->3', matching the AlphaGenome track convention.
        """
        from tangermeme.utils import one_hot_encode

        # For even receptive fields, nudge minus-strand TSS by 1 so the centre
        # is symmetric after reverse-complementing.
        if strand == "-" and self.rf % 2 == 0:
            tss = tss + 1

        start = max(0, tss - self.rf // 2)
        pad_lo = max((tss - self.rf // 2) * -1, 0)
        clen = self.genome.get_reference_length(chrom)
        end = min(tss + self.rf // 2 + self.mod, clen)
        pad_hi = max(tss + self.rf // 2 + self.mod - clen, 0)

        seq = ("N" * pad_lo) + self.genome.fetch(chrom, start, end).upper() + ("N" * pad_hi)
        if ohe:
            seq = one_hot_encode(seq, force=True).unsqueeze(0)
        if strand == "-":
            seq = reverse_complement(seq) if ohe else seq[::-1]
        return seq
