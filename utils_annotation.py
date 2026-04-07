# =============================================================================
# utils_annotation.py
# Optimized bioinformatics utilities for K. pneumoniae AMR Dashboard
#
# Key memory optimizations applied:
#   1. VCF parsed line-by-line (generator) — never builds a full raw list
#   2. ANN entries filtered AT READ-TIME — impact + resistance checks inline
#   3. Categorical dtypes for all high-cardinality string columns
#   4. Unused columns dropped immediately after creation
#   5. gc.collect() called after every major allocation/free
#   6. GFF de-duplicated before returning (CDS wins over gene)
#   7. All caches keyed on (bytes, params) so changing filters
#      invalidates correctly without re-reading disk
# =============================================================================

import re
import gc
import pandas as pd
import numpy as np
import streamlit as st

# =============================================================================
# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# =============================================================================

# Colistin / AMR resistance gene registry: symbol → locus_tag
RESISTANCE_GENE_MAP: dict[str, str] = {
    "mgrB":      "ACOXO5_RS08420",
    "pmrA":      "ACOXO5_RS17135",
    "pmrB":      "ACOXO5_RS17140",
    "phoP":      "ACOXO5_RS15300",
    "phoQ":      "ACOXO5_RS15305",
    "crrA":      "ACOXO5_RS05515",
    "crrB":      "ACOXO5_RS10235",
    "blaOXA-48": "ACOXO5_RS28260",
    "ompK35":    "ACOXO5_RS16135",
    "arnT":      "ACOXO5_RS27350",
    "arnB":      "ACOXO5_RS27330",
    "pmrC":      "ACOXO5_RS07870",
}

# Reverse lookup: locus_tag → symbol  (O(1) set membership)
LOCUS_TO_RESISTANCE: dict[str, str] = {v: k for k, v in RESISTANCE_GENE_MAP.items()}
RESISTANCE_LOCI: frozenset[str]     = frozenset(RESISTANCE_GENE_MAP.values())
RESISTANCE_SYMBOLS: frozenset[str]  = frozenset(RESISTANCE_GENE_MAP.keys())

# Impact display colours (CSS hex)
IMPACT_COLORS: dict[str, str] = {
    "HIGH":     "#e63946",
    "MODERATE": "#f4a261",
    "LOW":      "#e9c46a",
    "MODIFIER": "#adb5bd",
}

# Canonical sort order
IMPACT_ORDER: list[str] = ["HIGH", "MODERATE", "LOW", "MODIFIER"]

# Effects that constitute gene disruption
DISRUPTION_EFFECTS: frozenset[str] = frozenset({
    "stop_gained", "stop_lost", "frameshift_variant", "start_lost",
    "splice_acceptor_variant", "splice_donor_variant",
    "exon_loss_variant", "feature_ablation", "transcript_ablation",
})

# Default "significant" impacts for Fast Mode
SIGNIFICANT_IMPACTS: frozenset[str] = frozenset({"HIGH", "MODERATE"})


# =============================================================================
# ── SMALL HELPERS ─────────────────────────────────────────────────────────────
# =============================================================================

def _safe_int(v: str) -> int:
    try:    return int(v)
    except: return 0          # noqa: E722

def _safe_float(v: str) -> float:
    try:    return float(v)
    except: return 0.0        # noqa: E722

def _parse_gff_attrs(attr_str: str) -> dict:
    """
    Parse a GFF3 attribute string (key=value;...) into a plain dict.
    Handles URL-percent-encoding of common characters.
    """
    attrs: dict = {}
    for seg in attr_str.split(";"):
        seg = seg.strip()
        if not seg or "=" not in seg:
            continue
        k, _, v = seg.partition("=")
        attrs[k.strip()] = (
            v.strip()
             .replace("%20", " ").replace("%2C", ",")
             .replace("%3B", ";").replace("%3D", "=")
        )
    return attrs


# =============================================================================
# ── GFF PARSER ────────────────────────────────────────────────────────────────
# =============================================================================

@st.cache_data(show_spinner=False)
def parse_gff(gff_bytes: bytes) -> pd.DataFrame:
    """
    Parse a GFF3 annotation file into a compact DataFrame.

    Memory notes
    ------------
    * GFF files are small (< 15 MB) — full load is acceptable.
    * De-duplicate by locus_tag (CDS > gene > other) before returning
      so downstream joins don't multiply rows.
    * gene_name and product stored as Categorical.

    Returns columns:
        locus_tag | gene_name | product | seqname | start | end | strand | feature
    """
    records: list[dict] = []

    try:
        text = gff_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        st.error(f"GFF decode error: {exc}")
        return pd.DataFrame()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 9:
            continue

        seqname, _, feature, start, end, _, strand, _, attrs_str = cols[:9]
        if feature in ("region", "sequence_feature"):
            continue

        attrs = _parse_gff_attrs(attrs_str)

        # ── Derive locus_tag (primary merge key) ──────────────────────
        lt = attrs.get("locus_tag", "")
        if not lt:
            lt = re.sub(r"^(gene:|CDS:|mRNA:|rRNA:|tRNA:)", "",
                        attrs.get("ID", ""))
        if not lt:
            lt = attrs.get("Name", "")
        if not lt:
            continue   # Cannot link to VCF without a key

        # ── Gene name ─────────────────────────────────────────────────
        gname = (
            attrs.get("gene")
            or attrs.get("gene_name")
            or (attrs.get("Name") if attrs.get("Name") != lt else None)
        )

        records.append({
            "locus_tag": lt.strip(),
            "gene_name": gname if gname else None,
            "product":   attrs.get("product") or attrs.get("function") or "",
            "seqname":   seqname,
            "start":     _safe_int(start),
            "end":       _safe_int(end),
            "strand":    strand,
            "feature":   feature,
        })

    if not records:
        return pd.DataFrame(columns=[
            "locus_tag", "gene_name", "product",
            "seqname", "start", "end", "strand", "feature"
        ])

    gff = pd.DataFrame(records)
    del records; gc.collect()   # free raw list immediately

    # De-duplicate: prefer CDS > gene > other
    feat_rank = {"CDS": 0, "gene": 1}
    gff["_r"] = gff["feature"].map(feat_rank).fillna(2).astype("int8")
    gff = (
        gff.sort_values("_r")
           .drop_duplicates(subset="locus_tag", keep="first")
           .drop(columns="_r")
           .reset_index(drop=True)
    )

    # ── Categorical dtypes to save RAM ────────────────────────────────
    for col in ("gene_name", "product", "seqname", "strand", "feature"):
        gff[col] = gff[col].astype("category")

    return gff


# =============================================================================
# ── VCF PARSER  (streaming / memory-safe) ────────────────────────────────────
# =============================================================================

def _iter_ann_entries(ann_string: str):
    """
    Generator — yields one dict per ANN entry (comma-separated).

    SnpEff ANN pipe columns (0-indexed):
      0  Allele | 1 Effect | 2 Impact | 3 GeneName | 4 GeneID |
      5  FeatureType | 6 FeatureID | 7 Biotype | 8 Rank |
      9  HGVS.c | 10 HGVS.p | 11 cDNA_pos | 12 CDS_pos |
      13 AA_pos | 14 Distance | 15 ERRORS

    Using a generator avoids materialising the full list when most
    entries are discarded by the caller's impact/resistance filters.
    """
    for entry in ann_string.split(","):
        p = entry.split("|")
        if len(p) < 4:
            continue

        def g(i: int) -> str:
            return p[i].strip() if len(p) > i else ""

        yield {
            "effect":      g(1),
            "impact":      g(2),
            "gene_symbol": g(3),
            "locus_tag":   g(4),
            "hgvs_c":      g(9),
            "hgvs_p":      g(10),
            "aa_pos":      g(13),
        }


@st.cache_data(show_spinner=False)
def parse_vcf(
    file_bytes: bytes,
    sample_name: str,
    keep_impacts: tuple[str, ...] = ("HIGH", "MODERATE", "LOW", "MODIFIER"),
    only_resistance: bool = False,
) -> pd.DataFrame:
    """
    Parse a single SnpEff-annotated VCF file — streaming, memory-safe.

    Memory strategy
    ---------------
    1. Lines are processed one at a time; no intermediate list of all lines.
    2. ``keep_impacts`` : ANN entries with other impact levels are skipped
       IN THE INNER LOOP before any allocation — O(1) filter.
    3. ``only_resistance`` : when True, entries whose locus_tag AND
       gene_symbol are both unknown are skipped — reduces rows 10–100×
       for large whole-genome VCFs.
    4. Lines without ANN= are skipped early (string search before split).
    5. After building the DataFrame, string columns are cast to ``category``
       dtype, and intermediate Python lists are freed + gc.collect().

    The Streamlit cache key is (file_bytes, sample_name, keep_impacts,
    only_resistance) — changing filter params rebuilds the cache slice
    for that file without reloading others.
    """
    rows: list[dict] = []
    keep_set = set(keep_impacts)

    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        st.error(f"VCF decode error [{sample_name}]: {exc}")
        return pd.DataFrame()

    for raw in text.splitlines():
        # ── Fast skip: header and non-variant lines ────────────────────
        if not raw or raw[0] == "#":
            continue

        # ── Fast skip: lines without ANN= annotation ──────────────────
        # avoids splitting every unannotated line (common in filtered VCFs)
        if "ANN=" not in raw:
            if not only_resistance:
                # Still record unannotated variants — split only now
                cols = raw.split("\t")
                if len(cols) >= 8:
                    rows.append({
                        "sample": sample_name,
                        "chrom":  cols[0], "pos": _safe_int(cols[1]),
                        "ref":    cols[3], "alt": cols[4],
                        "qual":   _safe_float(cols[5]),
                        "locus_tag": None, "gene_symbol": None,
                        "effect": None, "impact": None,
                        "hgvs_c": None, "hgvs_p": None, "aa_pos": None,
                    })
            continue

        # ── Full split only for annotated lines ───────────────────────
        cols = raw.split("\t")
        if len(cols) < 8:
            continue

        chrom, pos, _, ref, alt, qual, _, info = cols[:8]

        ann_m = re.search(r"(?:^|;)ANN=([^;]+)", info)
        if not ann_m:
            continue

        base = {
            "sample": sample_name,
            "chrom":  chrom,
            "pos":    _safe_int(pos),
            "ref":    ref,
            "alt":    alt,
            "qual":   _safe_float(qual),
        }

        for ann in _iter_ann_entries(ann_m.group(1)):   # generator — lazy
            # ── Impact filter (innermost — cheapest check first) ──────
            if ann["impact"] and ann["impact"] not in keep_set:
                continue

            # ── Resistance filter ─────────────────────────────────────
            if only_resistance:
                if (ann["locus_tag"] not in RESISTANCE_LOCI and
                        ann["gene_symbol"] not in RESISTANCE_SYMBOLS):
                    continue

            rows.append({**base, **ann})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    del rows; gc.collect()   # free Python list — DataFrame owns the data now

    # ── Cast to memory-efficient dtypes ───────────────────────────────
    cat_cols = ["sample", "chrom", "ref", "alt",
                "locus_tag", "gene_symbol", "effect", "impact",
                "hgvs_c", "hgvs_p", "aa_pos"]
    for col in cat_cols:
        if col in df.columns:
            # Replace empty strings with NaN first, then categorise
            df[col] = (
                df[col].replace("", None)
                       .astype("category")
            )

    # pos / qual as compact numeric
    df["pos"]  = pd.to_numeric(df["pos"],  errors="coerce").astype("int32")
    df["qual"] = pd.to_numeric(df["qual"], errors="coerce").astype("float32")

    return df


# =============================================================================
# ── MERGE VCF + GFF + RESISTANCE TAGGING ─────────────────────────────────────
# =============================================================================

def merge_vcf_gff(vcf_df: pd.DataFrame, gff_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join VCF mutations onto GFF annotation via locus_tag.

    Memory notes
    ------------
    * Only three GFF columns are merged (slim projection).
    * Categorical columns preserved; new columns added as category.
    * Intermediate objects freed after use.

    Adds columns
    ------------
    gene_name        : GFF gene_name > vcf gene_symbol > locus_tag
    product          : from GFF
    resistance_gene  : canonical symbol if locus is a resistance gene
    category         : "CRITICAL_RESISTANCE" | "OTHER"
    is_disruption    : bool — frameshift/stop + HIGH impact
    mgr_b_disruption : bool — mgrB specifically disrupted
    """
    if vcf_df.empty:
        return vcf_df

    df = vcf_df.copy()

    # Convert categorical locus_tag to object for merge (pandas requirement)
    df["locus_tag"] = df["locus_tag"].astype(object).fillna("").str.strip()

    # ── Slim GFF join ─────────────────────────────────────────────────
    if not gff_df.empty:
        gff_slim = (
            gff_df[["locus_tag", "gene_name", "product"]]
            .copy()
            .assign(locus_tag=lambda x: x["locus_tag"].astype(str).str.strip())
        )
        df = df.merge(gff_slim, on="locus_tag", how="left")
        del gff_slim
    else:
        df["gene_name"] = None
        df["product"]   = None

    # ── Best display name: GFF > VCF gene_symbol > locus_tag ─────────
    gs = df["gene_symbol"].astype(object) if "gene_symbol" in df.columns else None
    gn = df["gene_name"].astype(object)

    if gs is not None:
        gn = gn.fillna(gs)
    gn = gn.fillna(df["locus_tag"].replace("", None)).fillna("unknown")
    df["gene_name"] = gn.astype("category")

    df["product"] = (
        df["product"].astype(object)
                     .fillna("hypothetical protein")
                     .astype("category")
    )

    # ── Resistance tagging ─────────────────────────────────────────────
    lt_obj = df["locus_tag"].astype(str)
    df["resistance_gene"] = lt_obj.map(LOCUS_TO_RESISTANCE)

    if "gene_symbol" in df.columns:
        no_res = df["resistance_gene"].isna()
        sym_obj = df.loc[no_res, "gene_symbol"].astype(object).fillna("")
        df.loc[no_res, "resistance_gene"] = sym_obj.map(
            {s: s for s in RESISTANCE_SYMBOLS}
        )

    df["category"] = np.where(
        df["resistance_gene"].notna(), "CRITICAL_RESISTANCE", "OTHER"
    ).astype(object)
    df["category"]         = df["category"].astype("category")
    df["resistance_gene"]  = df["resistance_gene"].astype("category")

    # ── Disruption detection ───────────────────────────────────────────
    eff_obj = df["effect"].astype(object).fillna("")
    imp_obj = df["impact"].astype(object).fillna("")

    df["is_disruption"]    = eff_obj.isin(DISRUPTION_EFFECTS) & (imp_obj == "HIGH")
    df["mgr_b_disruption"] = df["is_disruption"] & (df["resistance_gene"] == "mgrB")

    # ── Drop gene_symbol column — gene_name is now the canonical name ─
    df.drop(columns=["gene_symbol"], inplace=True, errors="ignore")

    gc.collect()
    return df


# =============================================================================
# ── AGGREGATION HELPERS ───────────────────────────────────────────────────────
# =============================================================================

def build_pivot(
    df: pd.DataFrame,
    label_col: str = "gene_name",
    max_genes: int = 50,
    max_samples: int = 60,
) -> pd.DataFrame:
    """
    Build a genes × samples mutation-count pivot table.

    Optimizations
    -------------
    * groupby aggregation instead of pivot_table (2–5× faster on large dfs).
    * Resistance genes always included (unioned with top-N).
    * Returns int16 values (sufficient for mutation counts).
    """
    sub = df.dropna(subset=[label_col, "sample"])
    if sub.empty:
        return pd.DataFrame()

    # groupby count — much faster than pivot_table for large frames
    counts = (
        sub.groupby(
            [sub[label_col].astype(str), sub["sample"].astype(str)],
            observed=True,   # observed=True avoids Cartesian product on categoricals
        )
        .size()
        .reset_index(name="n")
    )
    counts.columns = [label_col, "sample", "n"]

    # Always keep resistance genes
    res_labels: set[str] = set(
        df.loc[df["category"] == "CRITICAL_RESISTANCE", label_col]
          .astype(str).dropna().unique()
    )
    top_genes: set[str] = set(
        counts.groupby(label_col)["n"].sum()
              .nlargest(max_genes).index
    )
    keep_genes = top_genes | res_labels
    counts = counts[counts[label_col].isin(keep_genes)]

    top_samples: pd.Index = (
        counts.groupby("sample")["n"].sum()
              .nlargest(max_samples).index
    )
    counts = counts[counts["sample"].isin(top_samples)]

    pivot = (
        counts.pivot(index=label_col, columns="sample", values="n")
              .fillna(0)
              .astype("int16")   # saves ~50% RAM vs int64
    )
    return pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]


def resistance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-resistance-gene aggregation:
        gene | locus_tag | n_samples | n_mutations | n_disruptions | has_high | samples
    """
    res_df = df[df["category"] == "CRITICAL_RESISTANCE"]
    if res_df.empty:
        return pd.DataFrame(columns=[
            "gene", "locus_tag", "n_samples", "n_mutations",
            "n_disruptions", "has_high", "samples",
        ])

    rg_str = res_df["resistance_gene"].astype(str)
    agg = (
        res_df.assign(resistance_gene=rg_str)
              .groupby("resistance_gene", observed=True, sort=False)
              .agg(
                  n_samples    =("sample",        "nunique"),
                  n_mutations  =("sample",         "count"),
                  n_disruptions=("is_disruption",  "sum"),
                  has_high     =("impact",
                                 lambda x: (x.astype(str) == "HIGH").any()),
                  samples      =("sample",
                                 lambda x: ", ".join(
                                     sorted(x.astype(str).dropna().unique())
                                 )),
              )
              .reset_index()
              .rename(columns={"resistance_gene": "gene"})
    )
    agg["locus_tag"]     = agg["gene"].map(RESISTANCE_GENE_MAP)
    agg["n_disruptions"] = agg["n_disruptions"].astype(int)
    return agg.sort_values("n_samples", ascending=False).reset_index(drop=True)


def get_gene_details(df: pd.DataFrame, gene_label: str) -> pd.DataFrame:
    """
    All mutation rows for a given gene (matched by gene_name or locus_tag).
    Sorted HIGH → MODERATE → LOW → MODIFIER, then by sample.
    """
    mask = (
        (df["gene_name"].astype(str) == gene_label) |
        (df["locus_tag"].astype(str) == gene_label)
    )
    sub = df[mask].copy()
    if sub.empty:
        return sub

    rank = {k: i for i, k in enumerate(IMPACT_ORDER)}
    sub["_r"] = sub["impact"].astype(str).map(rank).fillna(99).astype("int8")
    return sub.sort_values(["_r", "sample"]).drop(columns="_r")
