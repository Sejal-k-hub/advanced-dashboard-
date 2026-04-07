import gc
import io
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

st.set_page_config(layout="wide")
st.title("🧬 VCF Mutation Dashboard")

# ── session state ─────────────────────────────────────────────────────────────
if "df"  not in st.session_state: st.session_state.df  = None
if "key" not in st.session_state: st.session_state.key = ""
if "l2g" not in st.session_state: st.session_state.l2g = {}

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### 📁 Step 1 — Upload VCF files")
vcf_files = st.sidebar.file_uploader(
    "Upload one or more .vcf files",
    type=["vcf"],
    accept_multiple_files=True,
    help="Select all your annotated .vcf files (SnpEff ANN= format)"
)

st.sidebar.markdown("### 📄 Step 2 — GFF file (optional)")
gff_file = st.sidebar.file_uploader(
    "Upload GFF/GFF3 file",
    type=["gff", "gff3", "txt"],
    accept_multiple_files=False,
)

st.sidebar.markdown("### ⚡ Step 3 — Mode")
fast_mode = st.sidebar.toggle("Fast Mode (HIGH + MOD only)", value=True)

st.sidebar.markdown("### ▶ Step 4")
load_btn = st.sidebar.button("LOAD FILES", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")
gene_q = st.sidebar.text_input("Gene name", placeholder="e.g. mgrB")
mut_q  = st.sidebar.text_input("Protein change", placeholder="e.g. p.Tyr")

# ── GFF parser (from uploaded file object) ────────────────────────────────────
def parse_gff(file_obj) -> dict:
    out = {}
    if file_obj is None:
        return out
    content = file_obj.read().decode("utf-8", errors="replace")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        cols = line.split("\t")
        if len(cols) < 9: continue
        attrs = {}
        for seg in cols[8].split(";"):
            if "=" not in seg: continue
            k, _, v = seg.partition("=")
            attrs[k.strip()] = v.strip().replace("%20", " ")
        lt = attrs.get("locus_tag", "")
        if lt:
            out[lt] = attrs.get("gene") or attrs.get("Name") or lt
    return out

# ── VCF parser (from uploaded file object) ────────────────────────────────────
def parse_vcf(file_obj, sample: str, fast: bool, l2g: dict) -> pd.DataFrame:
    keep = {"HIGH","MODERATE"} if fast else {"HIGH","MODERATE","LOW","MODIFIER"}
    rows = []
    content = file_obj.read().decode("utf-8", errors="replace")
    for raw in content.splitlines():
        raw = raw.rstrip()
        if not raw or raw[0]=="#" or "ANN=" not in raw: continue
        cols = raw.split("\t")
        if len(cols) < 8: continue
        chrom,pos,_,ref,alt,qual,_,info = cols[:8]
        m = re.search(r"(?:^|;)ANN=([^;]+)", info)
        if not m: continue
        try:    pv = int(pos)
        except: pv = 0
        try:    qv = float(qual)
        except: qv = 0.0
        for entry in m.group(1).split(","):
            p = entry.split("|")
            if len(p) < 4: continue
            impact = p[2].strip() if len(p)>2 else ""
            if impact and impact not in keep: continue
            def g(i): return p[i].strip() if len(p)>i else ""
            gene = g(3)
            if gene and l2g: gene = l2g.get(gene, gene)
            rows.append({
                "Sample":  sample,
                "Gene":    gene   or None,
                "Effect":  g(1)   or None,
                "Impact":  impact or None,
                "Protein": g(10)  or None,
                "POS": pv, "REF": ref, "ALT": alt, "QUAL": qv,
            })
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    del rows; gc.collect()
    for c in ["Sample","Gene","Effect","Impact","Protein","REF","ALT"]:
        if c in df.columns: df[c] = df[c].astype("category")
    df["POS"]  = pd.to_numeric(df["POS"],  errors="coerce").astype("int32")
    df["QUAL"] = pd.to_numeric(df["QUAL"], errors="coerce").astype("float32")
    return df

# ── landing ───────────────────────────────────────────────────────────────────
if not vcf_files:
    st.markdown("""
    ## 👋 Welcome! Follow these 4 steps in the sidebar:

    **Step 1** — Upload your `.vcf` files (SnpEff annotated, with `ANN=` fields)

    **Step 2** — Upload your GFF file (optional, for gene name mapping)

    **Step 3** — Keep **Fast Mode ON** for faster loading

    **Step 4** — Click **LOAD FILES**

    ---
    > 💡 Files are processed entirely in the browser session and never stored permanently.
    """)
    st.stop()

# ── cache key ─────────────────────────────────────────────────────────────────
def make_key(files, fast):
    parts = [f"{f.name}:{f.size}" for f in files]
    parts.append("fast" if fast else "full")
    return "|".join(parts)

new_key   = make_key(vcf_files, fast_mode)
need_load = load_btn or new_key != st.session_state.key or st.session_state.df is None

# ── parse ─────────────────────────────────────────────────────────────────────
if need_load:
    l2g = {}
    if gff_file is not None:
        with st.spinner("Loading GFF…"):
            l2g = parse_gff(gff_file)
        st.session_state.l2g = l2g
        st.sidebar.success(f"GFF: {len(l2g):,} genes mapped")

    l2g    = st.session_state.l2g
    bar    = st.progress(0, text="Starting…")
    frames = []
    n      = len(vcf_files)

    for i, f in enumerate(vcf_files):
        sample = f.name.split(".")[0]
        bar.progress((i+1)/n, text=f"{sample}  ({i+1}/{n})")
        df_one = parse_vcf(f, sample, fast_mode, l2g)
        if not df_one.empty:
            frames.append(df_one)
        del df_one; gc.collect()

    bar.empty()

    if not frames:
        st.error("No variants found. Check files have ANN= annotations.")
        st.stop()

    with st.spinner("Merging…"):
        st.session_state.df  = pd.concat(frames, ignore_index=True)
        st.session_state.key = new_key
        del frames; gc.collect()

# ── retrieve ──────────────────────────────────────────────────────────────────
df_all = st.session_state.df
l2g    = st.session_state.l2g

st.success(
    f"✅ {len(df_all):,} variants · "
    f"{df_all['Sample'].nunique()} samples · "
    f"GFF {'✓' if l2g else '✗'} · "
    f"{'Fast' if fast_mode else 'Full'} mode"
)

# ── filters ───────────────────────────────────────────────────────────────────
df = df_all.copy()
if gene_q: df = df[df["Gene"].astype(str).str.contains(gene_q, case=False, na=False)]
if mut_q:  df = df[df["Protein"].astype(str).str.contains(mut_q,  case=False, na=False)]
st.info(f"Showing **{len(df):,}** variants")
if df.empty:
    st.error("No variants match filters.")
    st.stop()

# ── pivot (reused in 2 charts) ────────────────────────────────────────────────
_h  = df.dropna(subset=["Gene"])
_gc = (_h.assign(Gene=_h["Gene"].astype(str), Sample=_h["Sample"].astype(str))
         .groupby(["Gene","Sample"], observed=True).size().reset_index(name="n"))
_t30 = _gc.groupby("Gene")["n"].sum().nlargest(30).index
pivot = (_gc[_gc["Gene"].isin(_t30)]
           .pivot(index="Gene", columns="Sample", values="n")
           .fillna(0).astype("int16"))
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
del _h, _gc; gc.collect()

# ── helpers ───────────────────────────────────────────────────────────────────
def show(fig, key):
    try:    st.plotly_chart(fig, width="stretch", key=key)
    except: st.plotly_chart(fig, use_container_width=True, key=key)

def show_mpl(fig):
    try:    st.pyplot(fig, use_container_width=True)
    except: st.pyplot(fig)
    plt.close(fig); gc.collect()

# ── CHART 1 + 2 ───────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.subheader("📊 Impact Distribution")
    _ic = df["Impact"].astype(str).value_counts().reset_index()
    _ic.columns = ["Impact","Count"]
    show(px.bar(_ic, x="Impact", y="Count", color="Impact"), "ic")

with c2:
    st.subheader("🧪 Mutation Types (Top 10)")
    _pie = df["Effect"].astype(str).value_counts().head(10)
    show(px.pie(names=_pie.index, values=_pie.values, hole=0.35), "pie")

# ── CHART 3 + 4 ───────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)
with c3:
    st.subheader("🏆 Top 20 Genes")
    _top = df["Gene"].astype(str).value_counts().head(20)
    show(px.bar(x=_top.values, y=_top.index, orientation="h",
                color=_top.values, color_continuous_scale="Reds",
                labels={"x":"Mutations","y":"Gene"}), "top")

with c4:
    st.subheader("📋 Variants per Sample")
    _sp = df["Sample"].astype(str).value_counts().reset_index()
    _sp.columns = ["Sample","Count"]
    fig = px.bar(_sp, x="Sample", y="Count",
                 color="Count", color_continuous_scale="Blues")
    fig.update_layout(xaxis_tickangle=45, coloraxis_showscale=False)
    show(fig, "sp")

# ── CHART 5 — Sunburst ────────────────────────────────────────────────────────
st.subheader("🌍 Sunburst — Top 15 Genes")
_t15 = df["Gene"].astype(str).value_counts().head(15).index
_sun = (df[df["Gene"].astype(str).isin(_t15)].dropna(subset=["Gene","Effect"])
          .assign(Gene=lambda x: x["Gene"].astype(str),
                  Effect=lambda x: x["Effect"].astype(str))
          .groupby(["Gene","Effect"], observed=True)
          .size().reset_index(name="Count"))
if not _sun.empty:
    show(px.sunburst(_sun, path=["Gene","Effect"], values="Count",
                     color="Count", color_continuous_scale="RdBu"), "sun")
del _sun; gc.collect()

# ── CHART 6 — Heatmap ────────────────────────────────────────────────────────
st.subheader("🧬 Mutation Heatmap — Top 30 Genes")
if not pivot.empty and pivot.shape[0]>=2 and pivot.shape[1]>=2:
    fw = max(8, min(20, pivot.shape[1]*0.45+3))
    fh = max(5, min(14, pivot.shape[0]*0.36+2))
    fig6, ax = plt.subplots(figsize=(fw,fh))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.2,
                linecolor="white", cbar_kws={"shrink":0.6})
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout(); show_mpl(fig6)
else:
    st.info("Need ≥2 genes and ≥2 samples.")

# ── CHART 7 — Protein scatter ─────────────────────────────────────────────────
st.subheader("🔬 Protein Mutation Map")
_tr = df.dropna(subset=["Protein"])
if not _tr.empty:
    _tr = _tr.sample(min(2000,len(_tr)), random_state=1)
    show(px.scatter(_tr, x="POS", y=_tr["Gene"].astype(str),
                    color=_tr["Impact"].astype(str),
                    hover_data=["Protein","Sample"],
                    opacity=0.7, labels={"y":"Gene"}), "prot")
    del _tr; gc.collect()
else:
    st.info("No protein annotation data.")

# ── CHART 8 — Annotation table ────────────────────────────────────────────────
st.subheader("📋 Annotation Summary")
_sm = df["Effect"].astype(str).value_counts().reset_index()
_sm.columns = ["Effect","Count"]
_sm["Percent (%)"] = (_sm["Count"]/_sm["Count"].sum()*100).round(2)
try:    st.dataframe(_sm, width="stretch")
except: st.dataframe(_sm, use_container_width=True)

# ── CHART 9 — Presence/absence heatmap ───────────────────────────────────────
st.subheader("🧬 Presence / Absence Heatmap")
if not pivot.empty:
    _pr = (pivot>0).astype("int8")
    fig9, ax = plt.subplots(figsize=(max(8,min(20,_pr.shape[1]*0.45+3)),
                                     max(5,min(14,_pr.shape[0]*0.36+2))))
    sns.heatmap(_pr, cmap="coolwarm", ax=ax, linewidths=0.2, linecolor="white")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout(); show_mpl(fig9)
    del _pr; gc.collect()

# ── CHART 10 — All samples ────────────────────────────────────────────────────
st.subheader("📊 All Samples — Variant Count")
_sp2 = df["Sample"].astype(str).value_counts().reset_index()
_sp2.columns = ["Sample","Count"]
fig10 = px.bar(_sp2, x="Sample", y="Count")
fig10.update_layout(xaxis_tickangle=45)
show(fig10, "sp2"); del _sp2; gc.collect()

# ── Full data + CSV download ──────────────────────────────────────────────────
with st.expander("📂 Full Data (first 5,000 rows)"):
    try:    st.dataframe(df.head(5000), width="stretch")
    except: st.dataframe(df.head(5000), use_container_width=True)

buf = io.BytesIO()
out = df.copy()
for c in out.select_dtypes(["category"]).columns:
    out[c] = out[c].astype(str).replace("nan","")
out.to_csv(buf, index=False)
st.download_button("⬇️ Download CSV", buf.getvalue(), "variants.csv", "text/csv")
del buf, out; gc.collect()
