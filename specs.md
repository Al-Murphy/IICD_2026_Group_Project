Concept-shift pipeline: Replogle K562-essential × AlphaGenome — build spec
0. The one principle that governs everything
AlphaGenome is state-blind: it predicts expression from DNA sequence only, and a knockdown does not change the sequence at any downstream gene's locus. Therefore its predicted effect of any knockdown on any downstream gene is exactly 0. You do not run AlphaGenome per (perturbation, gene) pair. You run it once per gene for a K562 baseline, cache it, reuse across all 1,971 perturbations. The measured non-zero change against that zero prediction is the concept-shift signal. Any implementation that calls AlphaGenome inside a per-perturbation loop is wrong. This collapses compute from ~1,971 × 8,563 to ~8,563 predictions.
1. Environment
Python ≥3.10; pertpy, scanpy, anndata, numpy, pandas, scipy; alphagenome (prefer local weights over API to avoid rate limits). Build = hg38/GRCh38 throughout.
2. Input data (already inspected)
adata = pertpy.data.replogle_2022_k562_essential() — 310,385 cells × 8,563 genes.

obs['perturbation']: label, control = 'control' (10,691 cells).
obs['gene_id']: target Ensembl ID. obs['gene']: target symbol.
obs['nperts']: verified 1 for all perturbed cells, 0 for control → clean single-gene screen, no combinatorial entries.
var: chr, start, end, strand, length, class, ensembl_id. Some var_names are SYMBOL_ENSG… style, so match targets via var['ensembl_id'], never by symbol.

3. Step 1 — Detect .X normalisation (branches everything)
pythonimport numpy as np, scanpy as sc
x0 = adata.X[:50].toarray() if hasattr(adata.X,'toarray') else np.asarray(adata.X[:50])
is_raw = np.allclose(x0, np.round(x0)) and x0.min() >= 0
if is_raw:
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
# if not raw (floats/negatives -> already normalised): do NOT transform
4. Step 2 — 30-cell filter (agreed)
pythonpert_col, ctrl = 'perturbation', 'control'
vc = adata.obs[pert_col].value_counts()
keep = vc[vc >= 30].index
adata_f = adata[adata.obs[pert_col].isin(keep)].copy()   # expect 1,971 perts + control
Do not apply the benchmark's 5,000-HVG filter; keep all 8,563 genes.
5. Step 3 — Pseudobulk and measured Δ
pythonimport pandas as pd
X = adata_f.X.toarray() if hasattr(adata_f.X,'toarray') else np.asarray(adata_f.X)
pb = pd.DataFrame(X, index=adata_f.obs[pert_col].values, columns=adata_f.var_names).groupby(level=0).mean()
delta = pb.sub(pb.loc[ctrl], axis=1)
ctrl_expr = pb.loc[ctrl]
6. Step 4 — Knockdown QC (soft flag, percent-based, do not filter)
Use percent-of-control on the target, not raw logFC (logFC gives false "weak" calls on lowly-expressed targets).
pythonpert_to_ens = adata_f.obs.drop_duplicates(pert_col).set_index(pert_col)['gene_id'].astype(str).to_dict()
ens_to_var  = dict(zip(adata_f.var['ensembl_id'].astype(str), adata_f.var_names))
rows=[]
for P in [p for p in pb.index if p!=ctrl]:
    tv = ens_to_var.get(str(pert_to_ens.get(P)))
    if tv is None: rows.append((P,None,np.nan,'target_not_measured')); continue
    c,k = np.expm1(ctrl_expr[tv]), np.expm1(pb.loc[P,tv])
    rows.append((P,tv,100*(k-c)/(c+1e-9),'ok'))
kd = pd.DataFrame(rows, columns=['pert','target_var','pct_change','status'])
kd['knockdown_ok'] = kd['pct_change'] < -25
Expect ~180 target_not_measured (keep), the majority OK, a small handful truly flat/up (HLA-C, SNAPC4, EXOC3, RTTN). Carry the flag; do not drop rows.
7. Step 5 — Relevant (trans) gene set per perturbation
Per P (target g_P): DE genes vs control (sc.tl.rank_genes_groups(..., reference='control', method='wilcoxon'), or magnitude+significance), then exclude g_P, exclude cis neighbours (same chromosome within ±2 Mb of the target TSS — KRAB spreading), and keep only genes with valid hg38 coords on chr1..22,X,Y. Store relevant_genes[P].
8. Step 6 — Coordinate table (from var)
pythonv = adata_f.var.copy()
plus = v['strand'].astype(str).isin(['+','1'])
v['tss'] = np.where(plus, v['start'], v['end']).astype(int)
v = v[v['chr'].astype(str).str.match(r'^chr(\d+|X|Y)$')]
9. Step 7 — AlphaGenome baseline: ONE prediction per gene
pythonfrom alphagenome.models import dna_client
from alphagenome.data import genome
model = dna_client.create(API_KEY)   # or local
SEQ_LEN = 524_288                     # 2**19, supported input length
K562 = "EFO:0002067"                  # VERIFY from output metadata

def predict_gene(chrom, tss, strand, gstart, gend):
    half = SEQ_LEN//2
    iv = genome.Interval(chromosome=chrom, start=tss-half, end=tss+half)
    out = model.predict_interval(interval=iv,
            requested_outputs=[dna_client.OutputType.RNA_SEQ, dna_client.OutputType.CAGE],
            ontology_terms=[K562])
    # (A) select K562 SENSE-strand track(s) via out.*.metadata; average if >1
    # (B) CAGE/PRO-cap (PRIMARY): integrate over TSS ± 200bp, sense strand
    # (C) RNA_SEQ (SECONDARY): integrate gene body [gstart,gend], sense strand
    return cage_scalar, rna_scalar

baseline = {g: predict_gene(v.loc[g,'chr'], int(v.loc[g,'tss']), v.loc[g,'strand'],
                            int(v.loc[g,'start']), int(v.loc[g,'end'])) for g in v.index}
pd.DataFrame(baseline, index=['cage_pred','rna_pred']).T.to_parquet('ag_k562_baseline.parquet')

Primary proxy = CAGE/PRO-cap TSS signal; RNA-seq gene-body is secondary (noisier).
Strand matters twice: track selection and integration direction.
K562 often has several RNA-seq tracks → average.
Attribute names and the K562 ontology term vary by AlphaGenome version — verify against the installed build. Cache aggressively; resume from parquet if interrupted (~8,563 calls).

10. Step 8 — Assemble concept-shift table (predicted Δ ≡ 0)
pythonrecords=[]
for P in [p for p in delta.index if p!=ctrl]:
    for g in relevant_genes[P]:
        m = delta.loc[P,g]
        records.append((P,g,m,0.0,abs(m)))
cs = pd.DataFrame(records, columns=['pert','gene','measured_delta','pred_delta','error'])
cs.to_parquet('concept_shift_table.parquet')
11. Step 9 — Mandatory controls
(a) Within-state accuracy: Spearman of AlphaGenome baseline vs measured control pseudobulk across genes; must be reasonably positive, else restrict to genes AG predicts acceptably in-state. This is what makes it concept shift rather than generic weakness.
pythonfrom scipy.stats import spearmanr
common=[g for g in v.index if g in pb.columns]
rho = spearmanr([baseline[g][0] for g in common], pb.loc[ctrl,common].values)[0]
(b) E-distance strength baseline (the magnitude axis the scFM displacement must beat): per-perturbation E-distance (control vs perturbed) on PCA-reduced data via pertpy Distance, matching the benchmark's definition. Store e_distance[P]. If the eventual embedding displacement does not out-predict raw E-distance, the foundation model adds nothing (cf. benchmark Fig. 5e, strength↔error r≈0.8).
12. Outputs
knockdown_check.csv; ag_k562_baseline.parquet; concept_shift_table.parquet; perturbation_summary.parquet (per pert: n_cells, target, n_relevant_genes, e_distance, knockdown_ok, mean_error); printed retained count (expect 1,971) and within-state Spearman.
13. Critical do-nots

No per-(perturbation,gene) AlphaGenome calls; one per gene, predicted Δ ≡ 0.
Don't apply the 5,000-HVG filter; keep all 8,563 genes.
Don't drop perturbations by knockdown QC; flag only, on percent-change not logFC.
Exclude target + its ±2 Mb cis neighbours from relevant sets.
Don't skip the within-state Spearman control.
Don't trust hardcoded API attribute names or the K562 ontology term; verify against the build.
hg38 and strand-aware everywhere.
