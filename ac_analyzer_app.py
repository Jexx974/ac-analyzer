#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AC Analyzer – Outil Streamlit pour l'analyse d'écarts de prix sur Accords Cadres (CDG Habitat)
=============================================================================================

- Parsing **robuste** d'Excel hétérogènes (.xlsx/.xls/.xlsm) :
  - Détection première feuille non vide + ligne d’en-tête probable.
  - Normalisation colonnes (FR/EN, accents, variantes).
  - Lecture spéciale des tableaux "larges" (prestataires en colonnes + sous-entête "Prix unitaire").
- Fuzzy match (regroupement) des libellés d'articles.
- Comparaisons **Solo ↔ Groupement** (indices, panier commun, heatmaps).
- Bench prestataires, articles sensibles, qualité de données.
- Exports CSV/Parquet.

Lancez :
    streamlit run ac_analyzer_app.py
"""

from __future__ import annotations
import io
import os
import re
import json
import math
import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
from unidecode import unidecode

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

# =====================
# Paramétrage Streamlit
# =====================
st.set_page_config(page_title="AC Analyzer – CDG Habitat", layout="wide")
st.title("AC Analyzer – Analyse d'écarts Accords Cadres")
st.caption("Chargement multi-Excel hétérogènes, parsing robuste, datavisualisation des écarts")

# =====================
# Dictionnaire de colonnes
# =====================
CANON = {
    "item": [r"^item$", r"^code(\s*article|\s*item)?$", r"^ref(\.|erence)?$", r"^reference$", r"^id$", r"^article$", r"^poste$"],
    "description": [r"^designation$", r"^description$", r"^intitule$", r"^libelle$", r"^objet$"],
    "unit": [r"^unite$", r"^unite\s*de\s*mesure$", r"^um$", r"^u$", r"^unit$", r"^unity?$"],
    "quantity": [r"^qte$", r"^quantite$", r"^quantité$", r"^qty$", r"^quantities$", r"^nb$", r"^nombre$"],
    "unit_price": [r"^pu(ht)?$", r"^prix\s*unitaire(\s*ht)?$", r"^unit(\s*price)?$", r"^prix$", r"^price$"],
    "total_price": [r"^total(\s*ht)?$", r"^montant(\s*ht)?$", r"^amount$", r"^prix\s*total$"],
    "lot": [r"^lot$", r"^lot\s*n(o|°)$"],
    "provider": [r"^fournisseur$", r"^prestataire$", r"^entreprise$", r"^titulaire$", r"^provider$", r"^vendor$"],
    "date": [r"^date$", r"^date\s*commande$", r"^date\s*de\s*bc$", r"^periode$", r"^période$"],
    "order_id": [r"^(n(o|°)\s*bc|bon\s*de\s*commande|bc\s*n(o|°))$", r"^order(|\s*id)$"],
}
PATTERNS = {k: [re.compile(pat, re.I) for pat in v] for k, v in CANON.items()}
NUM_PAT = re.compile(r"[\s\u00A0]", re.U)  # espaces (y compris insécables)

# =====================
# Utilitaires parsing
# =====================

def to_num_safe(x: t.Any) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = NUM_PAT.sub("", s)
    if "," in s and s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    s = s.replace("€", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_col(c: str) -> str:
    c = unidecode(str(c)).strip().lower()
    c = re.sub(r"[\n\r\t]", " ", c)
    c = re.sub(r"\s+", " ", c)
    c = c.replace("/", " ")
    c = re.sub(r"[^a-z0-9%°\-\s]", "", c)
    c = c.strip().replace(" ", "_")
    return c


def find_header_row(df: pd.DataFrame, lookahead: int = 50) -> int | None:
    best_row, best_score = None, -1
    n = min(len(df), lookahead)
    for i in range(n):
        row = df.iloc[i]
        cleaned = [clean_col(x) for x in row.values]
        score = 0
        for val in cleaned:
            for pats in PATTERNS.values():
                if any(p.search(val.replace("_", " ")) for p in pats):
                    score += 1
                    break
        nan_penalty = sum(pd.isna(row))
        score = score - nan_penalty * 0.1
        if score > best_score and score >= 2:
            best_score, best_row = score, i
    return best_row


@dataclass
class ParseReport:
    sheet_name: str
    header_row: int | None
    recognized: dict
    unknown: list[str]


def map_columns(cols: list[str]) -> tuple[dict, dict, list[str]]:
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    unknown: list[str] = []
    for c in cols:
        cc = clean_col(c)
        matched = False
        for canon, pats in PATTERNS.items():
            if any(p.search(cc.replace("_", " ")) for p in pats):
                mapping[c] = canon
                reverse[canon] = c
                matched = True
                break
        if not matched:
            unknown.append(c)
    return mapping, reverse, unknown

# --- Détection et unpivot des tableaux "larges" (prestataires en colonnes + sous-entête "Prix unitaire") ---
PU_RE = re.compile(r"prix\s*unitaire", re.I)

def unpivot_wide_price_table(df_multi: pd.DataFrame, filename: str, sheet: str) -> pd.DataFrame | None:
    if not isinstance(df_multi.columns, pd.MultiIndex):
        return None
    price_cols = [(a, b) for (a, b) in df_multi.columns if PU_RE.search(str(b) or "")]
    if len(price_cols) < 2:
        return None

    # Description éventuelle dans l'un des niveaux d'entête
    def _is_desc(a, b) -> bool:
        a_ = unidecode(str(a or "")).lower()
        b_ = unidecode(str(b or "")).lower()
        return any(p.search(a_) or p.search(b_) for p in PATTERNS["description"])

    desc_cols = [(a, b) for (a, b) in df_multi.columns if _is_desc(a, b)]

    base = pd.DataFrame({"row_id": np.arange(len(df_multi))})
    if desc_cols:
        a, b = desc_cols[0]
        base["description"] = df_multi[(a, b)].astype(str)
    else:
        base["description"] = None

    parts = []
    for (a, b) in price_cols:
        provider = str(a)
        s = df_multi[(a, b)]
        temp = pd.DataFrame({
            "row_id": np.arange(len(s)),
            "provider": provider,
            "unit_price": s.map(to_num_safe),
        })
        parts.append(temp)
    long_prices = pd.concat(parts, ignore_index=True)

    out = long_prices.merge(base, on="row_id", how="left").drop(columns=["row_id"])  # + description si dispo
    out["source_file"], out["sheet_name"] = filename, sheet
    for miss in ["item", "unit", "quantity", "total_price", "lot", "date", "order_id"]:
        if miss not in out.columns:
            out[miss] = np.nan
    m = re.search(r"\blot\s*([0-9IVX]+)", unidecode(sheet).lower())
    if m:
        out["lot"] = m.group(1)
    return out

# =====================
# Lecture Excel robuste
# =====================

def read_excel_robust(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, list[ParseReport]]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    reports: list[ParseReport] = []
    frames: list[pd.DataFrame] = []

    for sheet in xls.sheet_names:
        try:
            raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)
            if raw.dropna(how="all").empty:
                continue
            header_row = find_header_row(raw)
            if header_row is None:
                first_non_empty = raw.index[~raw.isna().all(axis=1)][0]
                header_row = int(first_non_empty)

            # Lecture standard
            df = pd.read_excel(xls, sheet_name=sheet, header=header_row, dtype=object)
            df.columns = [str(c) for c in df.columns]
            df = df.loc[~df.isna().all(axis=1)].copy()

            # Lecture potentielle MultiIndex (header à 2 niveaux)
            try:
                df_multi = pd.read_excel(xls, sheet_name=sheet, header=[header_row, header_row + 1], dtype=object)
            except Exception:
                df_multi = None

            df_long = None
            if df_multi is not None and isinstance(df_multi.columns, pd.MultiIndex):
                df_long = unpivot_wide_price_table(df_multi, filename, sheet)

            if df_long is not None and not df_long.empty:
                df_ren = df_long.copy()
                mapping, reverse, unknown = map_columns(df_ren.columns.tolist())
                df_ren = df_ren.rename(columns=mapping)
            else:
                mapping, reverse, unknown = map_columns(df.columns.tolist())
                df_ren = df.rename(columns=mapping).copy()

            # Conversions numériques
            for col in ["unit_price", "total_price", "quantity"]:
                if col in df_ren.columns:
                    df_ren[col] = df_ren[col].map(to_num_safe)

            # Métadonnées de fichier
            df_ren["source_file"], df_ren["sheet_name"] = filename, sheet
            fname = unidecode(filename.lower())
            period = "groupement" if re.search(r"accord[_\-\s]?cadre|group(e|ement)|cdg|ec_?group", fname) else (
                "solo" if re.search(r"solo|ancien|historique|prestataire[_\-]?seul", fname) else "inconnu"
            )
            tag_ec = bool(re.search(r"\bec\b|entretien[_\-\s]?courant", fname))
            df_ren["period"], df_ren["tag_ec"] = period, tag_ec

            # Calcul total le cas échéant
            if ("total_price" not in df_ren.columns) or df_ren["total_price"].isna().all():
                if ("quantity" in df_ren.columns) and ("unit_price" in df_ren.columns):
                    df_ren["total_price"] = df_ren["quantity"] * df_ren["unit_price"]

            # Colonnes minimales
            for c in ["item", "description", "unit", "quantity", "unit_price", "total_price", "lot", "provider", "date", "order_id"]:
                if c not in df_ren.columns:
                    df_ren[c] = np.nan

            frames.append(df_ren)
            reports.append(ParseReport(sheet, header_row, mapping, unknown))

        except Exception as e:
            reports.append(ParseReport(sheet, None, {}, [f"Erreur: {type(e).__name__}: {e}"]))
            continue

    if not frames:
        return pd.DataFrame(), reports

    merged = pd.concat(frames, ignore_index=True)
    for c in ["item", "description", "unit", "quantity", "unit_price", "total_price", "lot", "provider", "date", "order_id"]:
        if c not in merged.columns:
            merged[c] = np.nan
    if merged["total_price"].isna().all() and ("quantity" in merged.columns) and ("unit_price" in merged.columns):
        merged["total_price"] = merged["quantity"] * merged["unit_price"]

    return merged, reports

# =====================
# Outliers & helpers
# =====================
@dataclass
class OutlierParams:
    method: str  # "iqr" | "z"
    z_thresh: float = 3.0
    iqr_k: float = 1.5

def flag_outliers(series: pd.Series, params: OutlierParams) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series(False, index=series.index)
    if params.method == "z":
        m = s.mean(); sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(False, index=series.index)
        z = (s - m) / sd
        return z.abs() > params.z_thresh
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - params.iqr_k * iqr, q3 + params.iqr_k * iqr
    return (s < lo) | (s > hi)

# =====================
# Sidebar
# =====================
with st.sidebar:
    st.header("Paramètres")
    st.markdown("**1) Import des fichiers**")
    files = st.file_uploader(
        "Déposez un ou plusieurs fichiers Excel (xlsx/xls/xlsm)",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
    )
    st.markdown("**2) Détection & Outliers**")
    method = st.selectbox("Méthode d'outliers", ["iqr", "z"], index=0)
    z_thresh = st.slider("Seuil Z-score", 2.0, 5.0, 3.0, 0.1)
    iqr_k = st.slider("IQR × k", 0.5, 4.0, 1.5, 0.1)

    st.markdown("**3) Fuzzy match d'articles**")
    use_fuzzy = st.toggle("Activer le regroupement d'articles similaires (fuzzy)")
    fuzzy_sim = st.slider("Seuil de similarité", 0.50, 0.95, 0.75, 0.01)
    fuzzy_min_len = st.slider("Longueur minimale des libellés", 3, 15, 4, 1)

    st.markdown("**4) Options**")
    use_llm = st.toggle("Aide LLM pour entêtes inconnues (optionnel)")
    save_outputs = st.toggle("Sauvegarder CSV & Parquet")

    st.markdown("""
    *Astuce :* si vos fichiers contiennent des feuilles très diverses, l'outil
    tentera d'identifier automatiquement la ligne d'entête et la première
    feuille non vide. Le *fuzzy match* rapproche les libellés proches.
    """)

params = OutlierParams(method=method, z_thresh=z_thresh, iqr_k=iqr_k)

# =====================
# Main flow
# =====================
if files:
    all_reports: list[ParseReport] = []
    dfs: list[pd.DataFrame] = []

    with st.spinner("Parsing des fichiers…"):
        for f in files:
            b = f.read()
            df, reports = read_excel_robust(b, f.name)
            all_reports.extend(reports)
            if not df.empty:
                dfs.append(df)

    if not dfs:
        st.error("Aucune donnée exploitable n'a été trouvée dans les fichiers fournis.")
        st.stop()

    data = pd.concat(dfs, ignore_index=True)

    # Option LLM (facultative pour mapper des colonnes inconnues)
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            unknown_cols = [c for c in data.columns if c not in CANON and c not in ["source_file", "sheet_name", "period", "tag_ec"]]
            if unknown_cols:
                prompt = (
                    "Tu es un assistant de normalisation de schémas pour des BPU en français.\n"
                    "Propose un mapping JSON des noms de colonnes suivants vers un des canons :\n"
                    f"{list(CANON.keys())}.\n"
                    f"Colonnes: {unknown_cols}\n"
                    "Réponds uniquement avec un objet JSON {col: canon_ou_null}.\n"
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = resp.choices[0].message.content.strip()
                try:
                    llm_map = json.loads(content)
                    for src, dst in llm_map.items():
                        if dst in CANON:
                            data = data.rename(columns={src: dst})
                except Exception:
                    pass
        except Exception as e:
            st.info(f"Aide LLM non disponible ({e}). L'analyse continue sans IA.")

    # Fuzzy grouping
    def _normalize_simple(s):
        if pd.isna(s):
            return ""
        s2 = unidecode(str(s).lower())
        s2 = re.sub(r"[^a-z0-9]+", " ", s2)
        return re.sub(r"\s+", " ", s2).strip()

    def build_fuzzy_groups_simple(df_in, text_col, min_len=4, sim_threshold=0.75):
        texts = df_in[text_col].fillna("").astype(str).map(_normalize_simple)
        mask = texts.str.len() >= min_len
        if mask.sum() < 2:
            return texts.rename("canonical_item")
        vect = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        X = vect.fit_transform(texts.where(mask, other=""))
        try:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - float(sim_threshold), linkage="average", metric="cosine")
        except TypeError:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - float(sim_threshold), linkage="average", affinity="cosine")
        labels = clustering.fit_predict(X.toarray())
        canonical = pd.Series(index=df_in.index, dtype=object)
        canonical.loc[mask] = labels
        canonical.loc[~mask] = texts.loc[~mask]
        rep = {}
        for lab in pd.unique(labels):
            idx = canonical[canonical == lab].index
            if len(idx):
                rep[lab] = str(df_in.loc[idx[0], text_col])
        return canonical.map(lambda v: rep.get(v, v)).rename("canonical_item")

    # Numérisation
    for c in ["unit_price", "total_price", "quantity"]:
        if c in data.columns:
            data[c] = data[c].map(to_num_safe)

    # Canonical item
    if use_fuzzy and "description" in data.columns and data["description"].notna().any():
        with st.spinner("Regroupement des articles similaires…"):
            data["canonical_item"] = build_fuzzy_groups_simple(data, "description", min_len=int(fuzzy_min_len), sim_threshold=float(fuzzy_sim))
    else:
        data["canonical_item"] = data.get("item") if "item" in data.columns and data["item"].notna().any() else data.get("description")

    # Outliers par groupe
    group_key = "canonical_item" if "canonical_item" in data.columns and data["canonical_item"].notna().any() else (
        "item" if "item" in data.columns and data["item"].notna().any() else (
        "description" if "description" in data.columns else None))

    if group_key and "unit_price" in data.columns:
        data["is_outlier"] = False
        for key, sub in data.groupby(group_key, dropna=False):
            flags = flag_outliers(sub["unit_price"], params)
            data.loc[sub.index, "is_outlier"] = flags.values
    else:
        data["is_outlier"] = False

    # KPIs
    kpi_total_rows = len(data)
    kpi_items = data[group_key].nunique() if group_key else 0
    kpi_providers = data["provider"].nunique() if "provider" in data.columns else 0

    st.subheader("Jeu consolidé")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lignes", kpi_total_rows)
    k2.metric("Articles (regroupés)", kpi_items)
    k3.metric("Prestataires", kpi_providers)
    has_both = ("period" in data.columns) and set(data["period"].dropna().unique()) >= {"solo", "groupement"}
    if has_both and group_key:
        common_mask = data[group_key].isin(
            set(data.loc[data["period"] == "solo", group_key].dropna()) &
            set(data.loc[data["period"] == "groupement", group_key].dropna())
        )
        coverage = (common_mask.mean() * 100.0)
        k4.metric("Couverture panier commun %", f"{coverage:.1f}%")
    else:
        k4.metric("Couverture panier commun %", "—")

    st.dataframe(data.head(1000))

    # Filtres
    cols = st.columns(4)
    with cols[0]:
        sel_period = st.multiselect(
            "Période",
            sorted(data["period"].dropna().unique()),
            default=list(sorted(data["period"].dropna().unique()))
        ) if "period" in data.columns else []
    with cols[1]:
        sel_lot = st.multiselect("Lot", sorted([x for x in data.get("lot", pd.Series([])).dropna().unique()]))
    with cols[2]:
        sel_provider = st.multiselect("Prestataire", sorted([x for x in data.get("provider", pd.Series([])).dropna().unique()]))
    with cols[3]:
        hide_outliers = st.toggle("Cacher les outliers")

    filt = pd.Series(True, index=data.index)
    if sel_period:
        filt &= data["period"].isin(sel_period)
    if sel_lot:
        filt &= data["lot"].isin(sel_lot)
    if sel_provider:
        filt &= data["provider"].isin(sel_provider)
    if hide_outliers:
        filt &= ~data["is_outlier"]

    dfv = data.loc[filt].copy()

    # =====================
    # Analyses & Graphiques
    # =====================
    if "unit_price" in dfv.columns:
        st.subheader("Écarts vs moyenne par article (regroupement fuzzy si activé)")
        if group_key:
            agg = dfv.groupby(group_key).agg(
                avg_price=("unit_price", "mean"),
                med_price=("unit_price", "median"),
                std_price=("unit_price", "std"),
                n=("unit_price", "count"),
                total_amount=("total_price", "sum") if "total_price" in dfv.columns else ("unit_price", "sum")
            ).reset_index()
            agg["cov"] = (agg["std_price"] / agg["avg_price"]).replace([np.inf, -np.inf], np.nan)
            st.dataframe(agg.sort_values("n", ascending=False).head(1000))

            st.markdown("### Graphiques")
            # Waterfall Solo → Groupement (panier commun sur médianes)
            if has_both and group_key:
                med = dfv.groupby([group_key, "period"]).agg(m=("unit_price", "median")).reset_index()
                common_keys = set(med.loc[med["period"] == "solo", group_key]) & set(med.loc[med["period"] == "groupement", group_key])
                m_solo = med[(med[group_key].isin(common_keys)) & (med["period"] == "solo")].set_index(group_key)["m"]
                m_grp = med[(med[group_key].isin(common_keys)) & (med["period"] == "groupement")].set_index(group_key)["m"]
                if not m_solo.empty and not m_grp.empty:
                    cost_solo = float(m_solo.sum()); cost_grp = float(m_grp.sum()); delta = cost_grp - cost_solo
                    fig_wf = go.Figure(go.Waterfall(
                        orientation="v",
                        measure=["absolute", "relative", "total"],
                        x=["Solo", "Δ prix (≈ panier commun)", "Groupement"],
                        text=[f"{cost_solo:,.0f}", f"{delta:,.0f}", f"{cost_grp:,.0f}"],
                        y=[cost_solo, delta, 0],
                    ))
                    fig_wf.update_layout(title="Waterfall – Coût panier commun (médiane par article)")
                    st.plotly_chart(fig_wf, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(dfv, x="unit_price", nbins=50, color="period" if "period" in dfv.columns else None)
                fig_hist.update_layout(title="Distribution des prix unitaires (par période)", xaxis_title="Prix unitaire", yaxis_title="Nombre de lignes")
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                if "quantity" in dfv.columns:
                    size_col = None
                    if "total_price" in dfv.columns:
                        dfv["_bubble_size"] = (
                            pd.to_numeric(dfv["total_price"], errors="coerce")
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0)
                            .clip(lower=0)
                        )
                        size_col = "_bubble_size"
                    fig_scatter = px.scatter(
                        dfv, x="quantity", y="unit_price", size=size_col,
                        color="is_outlier", hover_data=[group_key, "provider", "lot", "source_file"],
                    )
                    fig_scatter.update_layout(title="Prix vs Quantité (taille = montant)", xaxis_title="Quantité", yaxis_title="Prix unitaire")
                    st.plotly_chart(fig_scatter, use_container_width=True)

            # Treemap montants
            if ("total_price" in dfv.columns):
                df_tree = dfv.copy()
                df_tree["_tree_value"] = (
                    pd.to_numeric(df_tree["total_price"], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                    .clip(lower=0)
                )
                if df_tree["_tree_value"].sum() > 0:
                    fig_tree = px.treemap(
                        df_tree, path=["lot", "provider", group_key], values="_tree_value",
                        color="period" if "period" in df_tree.columns else None,
                    )
                    fig_tree.update_layout(title="Répartition des montants – Lot / Prestataire / Article")
                    st.plotly_chart(fig_tree, use_container_width=True)
                else:
                    st.info("Montants indisponibles ou nuls pour la treemap après nettoyage.")

            # Heatmap Δ% Solo→Groupement par lot
            if has_both and group_key and ("lot" in dfv.columns):
                med2 = dfv.groupby([group_key, "lot", "period"]).agg(m=("unit_price", "median")).reset_index()
                pivot = med2.pivot_table(index=[group_key], columns=["lot", "period"], values="m")
                lots = sorted({c[0] for c in pivot.columns if isinstance(c, tuple)})
                heat, index_labels = [], []
                for art, row in pivot.iterrows():
                    row_vals = []
                    for L in lots:
                        solo = row.get((L, "solo"), np.nan)
                        grp = row.get((L, "groupement"), np.nan)
                        val = (grp/solo - 1)*100 if pd.notna(solo) and pd.notna(grp) and solo != 0 else np.nan
                        row_vals.append(val)
                    heat.append(row_vals); index_labels.append(art)
                if heat:
                    fig_hm = go.Figure(data=go.Heatmap(z=heat, x=lots, y=index_labels, coloraxis="coloraxis"))
                    fig_hm.update_layout(title="Heatmap Δ% Solo→Groupement (médiane, par lot)", coloraxis_colorscale="RdBu", coloraxis_cmid=0)
                    st.plotly_chart(fig_hm, use_container_width=True)

            # Écarts individuels vs moyenne
            dfv = dfv.merge(agg[[group_key, "avg_price"]], on=group_key, how="left")
            dfv["delta_vs_avg"] = dfv["unit_price"] - dfv["avg_price"]
            top_over = dfv.sort_values("delta_vs_avg", ascending=False).head(50)
            top_under = dfv.sort_values("delta_vs_avg", ascending=True).head(50)
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**Top au-dessus de la moyenne**")
                st.dataframe(top_over[[group_key, "provider", "lot", "unit_price", "avg_price", "delta_vs_avg", "source_file"]])
            with c4:
                st.markdown("**Top en dessous de la moyenne**")
                st.dataframe(top_under[[group_key, "provider", "lot", "unit_price", "avg_price", "delta_vs_avg", "source_file"]])

            # Comparaison par lot (barres groupées + erreur σ)
            if "lot" in dfv.columns and dfv["lot"].notna().any():
                lot_stats = dfv.groupby(["lot", "period"]).agg(m=("unit_price", "mean"), s=("unit_price", "std")).reset_index()
                if not lot_stats.empty and set(lot_stats["period"].unique()) >= {"solo", "groupement"}:
                    fig_bars = px.bar(lot_stats, x="lot", y="m", color="period", barmode="group", error_y="s")
                    fig_bars.update_layout(title="Prix moyen par lot – Solo vs Groupement (barres d'erreur = σ)", yaxis_title="Prix moyen")
                    st.plotly_chart(fig_bars, use_container_width=True)
        else:
            st.info("Aucune clé d'article/description fiable pour regrouper les écarts.")

    # Articles sensibles & bench prestataires
    st.subheader("Articles sensibles & Bench prestataires")
    if group_key and "provider" in dfv.columns:
        med_by_art = dfv.groupby(group_key)["unit_price"].median()
        dfv = dfv.join(med_by_art.rename("med_art"), on=group_key)
        dfv["delta_med_%"] = ((dfv["unit_price"] - dfv["med_art"]) / dfv["med_art"]) * 100
        if "lot" in dfv.columns and dfv["lot"].notna().any():
            bench = dfv.assign(is_under=(dfv["unit_price"] <= dfv["med_art"]))
            bench = bench.groupby(["lot", "provider"]).agg(
                pct_under=("is_under", "mean"),
                avg_delta=("delta_med_%", "mean"),
                n=("unit_price", "count")
            ).reset_index()
            bench["pct_under"] *= 100
            st.markdown("**Classement prestataires (par lot) – % d'articles sous la médiane**")

            lots_list = sorted(bench["lot"].astype(str).unique())
            # Sélecteur si trop de lots
            default_sel = lots_list[:12]
            sel_lots_rank = st.multiselect("Lots à afficher (max 12)", lots_list, default=default_sel)
            sel_lots_rank = sel_lots_rank[:12] if len(sel_lots_rank) > 12 else sel_lots_rank
            bench_sel = bench[bench["lot"].astype(str).isin(sel_lots_rank)].copy()
            if bench_sel.empty:
                bench_sel = bench[bench["lot"].astype(str).isin(default_sel)]

            fig_rank = px.bar(
                bench_sel,
                x="pct_under",
                y="provider",
                facet_col="lot",
                facet_col_wrap=4,
                facet_col_spacing=0.04,
                orientation="h",
            )
            fig_rank.update_layout(title="% d'articles sous la médiane (par lot)", xaxis_title="% sous médiane")
            st.plotly_chart(fig_rank, use_container_width=True)

    # Articles volatils (CoV)
    if group_key:
        vol = dfv.groupby(group_key).agg(avg=("unit_price", "mean"), std=("unit_price", "std"), n=("unit_price", "count")).reset_index()
        vol["cov"] = (vol["std"] / vol["avg"]).replace([np.inf, -np.inf], np.nan)
        vol = vol.sort_values("cov", ascending=False).head(50)
        st.markdown("**Top articles sensibles (CoV élevé)**")
        fig_cov = px.bar(vol, x="cov", y=group_key, orientation="h")
        fig_cov.update_layout(xaxis_title="CoV (σ/μ)")
        st.plotly_chart(fig_cov, use_container_width=True)

    # Qualité & couverture
    st.subheader("Qualité & couverture des données")
    qual_cols = ["item", "description", "unit", "unit_price", "quantity", "total_price", "provider", "lot"]
    qual = pd.Series({c: 100.0 * data[c].notna().mean() if c in data.columns else 0.0 for c in qual_cols})
    fig_q = px.bar(x=qual.index, y=qual.values)
    fig_q.update_layout(title="% de valeurs non manquantes par champ", xaxis_title="Champ", yaxis_title="% non manquant")
    st.plotly_chart(fig_q, use_container_width=True)

    # Rapport de parsing
    st.subheader("Rapport de parsing par feuille")
    rep_df = pd.DataFrame([
        {
            "sheet": r.sheet_name,
            "header_row": r.header_row,
            "reconnu": ", ".join(sorted(set(r.recognized.values()))) if r.recognized else "",
            "inconnu/erreurs": ", ".join(map(str, r.unknown)) if r.unknown else "",
        }
        for r in all_reports
    ])
    st.dataframe(rep_df)

    # Export
    if save_outputs:
        out_csv = data.to_csv(index=False).encode("utf-8")
        out_parquet = io.BytesIO()
        try:
            data.to_parquet(out_parquet, index=False)
            parquet_bytes = out_parquet.getvalue()
        except Exception:
            parquet_bytes = b""
        st.download_button("Télécharger CSV consolidé", data=out_csv, file_name="ac_analyzer_consolide.csv", mime="text/csv")
        if parquet_bytes:
            st.download_button("Télécharger Parquet consolidé", data=parquet_bytes, file_name="ac_analyzer_consolide.parquet", mime="application/octet-stream")
else:
    st.info("Chargez un ou plusieurs fichiers Excel pour démarrer l'analyse.")

# =====================
# Bloc Chat LLM sur données importées
# =====================
if 'data' in locals() and not data.empty:
    st.header("💬 Assistant LLM – Analyse interactive des données")
    st.markdown("Posez vos questions en langage naturel (ex: *Quels sont les prestataires les plus chers ?*)")
    query = st.text_input("Votre question :", placeholder="Ex: Peux-tu fournir les prestataires qui sont les plus chers ?")
    if query:
        try:
            subset = data[['provider','lot','unit_price']].dropna()
            agg_df = subset.groupby(['provider','lot']).agg(moy_price=('unit_price','mean')).reset_index()

            # Si pas de clé API, on répond en local avec un classement simple
            if not os.getenv("OPENAI_API_KEY"):
                top = agg_df.sort_values('moy_price', ascending=False)
                # Affichage synthétique par lot (top 3 par lot)
                top_by_lot = top.groupby('lot').head(3)
                st.markdown("**(Mode local sans LLM)** Prestataires les plus chers (par lot, prix moyen):")
                st.dataframe(top_by_lot.rename(columns={'moy_price':'prix_moyen'}))
                # Résumé texte
                lignes = []
                for lot, grp in top_by_lot.groupby('lot'):
                    trio = ", ".join(f"{row.provider} ⇒ {row.moy_price:,.2f}" for _, row in grp.iterrows())
                    lignes.append(f"Lot {lot}: {trio}")
                st.success("\n".join(lignes))
            else:
                from openai import OpenAI
                client = OpenAI()
                data_json = agg_df.head(400).to_dict(orient='records')
                prompt = (
                    "Tu es un analyste de données. Voici des extraits du tableau de prix (provider, lot, prix_moyen): "
                    f"{data_json}\n"
                    "Réponds à la question suivante de manière claire et concise, en français:\n"
                    f"{query}"
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.0
                )
                answer = resp.choices[0].message.content.strip()
                st.success(answer)
        except Exception as e:
            st.error(f"Erreur lors de l'analyse/LLM : {e}")
