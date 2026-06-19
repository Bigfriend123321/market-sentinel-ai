"""Tableau de bord Market Sentinel AI (Streamlit) — interface dark premium.

Lancement :
    streamlit run dashboard/app.py

Navigation (barre latérale) : Opportunités, Pépites, Analyse entreprise,
Actualités, Alertes, Portefeuille virtuel, Historique.

NOTE : ce fichier ne contient QUE de la présentation (UI/UX). Toute la
logique métier (analyse, scoring, IA, données, calculs) vit dans le paquet
`market_sentinel` et n'est pas modifiée ici.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permettre les imports `market_sentinel.*` quel que soit le dossier de lancement.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from market_sentinel import DISCLAIMER, __version__  # noqa: E402
from market_sentinel.ai.ml_model import get_active_model  # noqa: E402
from market_sentinel.analysis.market_regime import (  # noqa: E402
    market_trend_score,
    regime_label,
)
from market_sentinel.analysis.opportunities import (  # noqa: E402
    analyze_ticker,
    analyze_watchlist,
)
from market_sentinel.analysis.backtest import run_backtest  # noqa: E402
from market_sentinel.analysis.screener import find_gems, opportunity_index  # noqa: E402
from market_sentinel.config import load_config  # noqa: E402
from market_sentinel.data.market_data import get_history  # noqa: E402
from market_sentinel.news.fetcher import fetch_news  # noqa: E402
from market_sentinel.storage.database import Database  # noqa: E402

st.set_page_config(
    page_title="Market Sentinel AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = load_config()
db = Database(cfg.resolve_path(cfg.get("storage.database_path", "data/market_sentinel.db")))
MODEL_PATH = str(cfg.resolve_path(cfg.get("ai.model_path", "data/predictor.joblib")))

# --- Palette ---------------------------------------------------------------
BG, BG2, BG3 = "#0A0F1C", "#101827", "#131D31"
BLUE, BLUE2, BLUE3 = "#2563EB", "#3B82F6", "#60A5FA"
T1, T2, T3 = "#F3F4F6", "#D1D5DB", "#9CA3AF"
OK, WARN, ERR = "#22C55E", "#F59E0B", "#EF4444"

# --- Thème Plotly sombre (appliqué à TOUS les graphiques) ------------------
pio.templates["ms_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T2, family="Inter, sans-serif", size=13),
        colorway=[BLUE2, BLUE3, OK, WARN, ERR, "#A78BFA"],
        xaxis=dict(gridcolor="rgba(148,163,184,0.10)", zerolinecolor="rgba(148,163,184,0.18)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.10)", zerolinecolor="rgba(148,163,184,0.18)"),
        hoverlabel=dict(bgcolor=BG3, bordercolor=BLUE, font=dict(color=T1, family="Inter")),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=10, r=10, t=44, b=10),
    )
)
pio.templates.default = "ms_dark"
PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": True, "displaylogo": False}


def plot(fig: go.Figure):
    """Affiche un graphique Plotly en pleine largeur avec la config commune."""
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


# --- Fonctions mises en cache (limitent les appels réseau) ----------------
@st.cache_data(ttl=900, show_spinner=False)
def cached_watchlist():
    """Analyse complète de l'univers (liste de dicts, triée par score)."""
    return [a.__dict__ for a in analyze_watchlist(cfg)]


@st.cache_data(ttl=300, show_spinner=False)
def cached_analysis(ticker: str):
    return analyze_ticker(ticker, cfg).__dict__


@st.cache_data(ttl=300, show_spinner=False)
def cached_history(ticker: str, period: str):
    return get_history(ticker, period=period)


@st.cache_data(ttl=600, show_spinner=False)
def cached_news(ticker: str):
    return [n.__dict__ for n in fetch_news(ticker, cfg.get("news.max_articles", 20))]


@st.cache_data(ttl=900, show_spinner=False)
def cached_market():
    score = market_trend_score(cfg.get("ai.market_index", "^GSPC"))
    return score, regime_label(score)


@st.cache_data(ttl=900, show_spinner=False)
def cached_universe_size():
    from market_sentinel.analysis.universe import get_universe

    return len(get_universe(cfg))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_backtest(top_n: int, period: str):
    res = run_backtest(cfg.watchlist, period=period, top_n=top_n)
    return {
        "equity": res.equity,
        "benchmark": res.benchmark,
        "metrics": res.metrics,
        "benchmark_metrics": res.benchmark_metrics,
        "n_tickers": res.n_tickers,
    }


def fmt_price(value, currency=""):
    return f"{value:,.2f} {currency or ''}".strip() if value is not None else "N/A"


def section(title: str, subtitle: str = "") -> None:
    """Titre de section avec hiérarchie visuelle soignée."""
    sub = f'<div class="ms-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="ms-section">{title}</div>{sub}', unsafe_allow_html=True)


def indicator_chart(history: pd.DataFrame, ticker: str) -> go.Figure:
    """Graphique multi-panneaux : cours + moyennes mobiles, RSI, volume."""
    close = history["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.19, 0.19], vertical_spacing=0.05,
        subplot_titles=("Cours + moyennes mobiles", "RSI (14)", "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=history.index, open=history["Open"], high=history["High"],
            low=history["Low"], close=history["Close"], name="Cours",
            increasing_line_color=OK, decreasing_line_color=ERR,
            increasing_fillcolor=OK, decreasing_fillcolor=ERR,
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=history.index, y=sma50, name="SMA 50",
                             line=dict(width=1.5, color=BLUE3, shape="spline")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=sma200, name="SMA 200",
                             line=dict(width=1.5, color=WARN, shape="spline")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=rsi, name="RSI",
                             line=dict(width=1.5, color=BLUE2, shape="spline")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=ERR, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=OK, row=2, col=1)
    fig.add_trace(go.Bar(x=history.index, y=history["Volume"], name="Volume",
                         marker_color="rgba(96,165,250,0.45)"), row=3, col=1)

    fig.update_layout(height=620, xaxis_rangeslider_visible=False, showlegend=True)
    return fig


# --- Style premium (CSS) ---------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#0A0F1C; --bg2:#101827; --bg3:#131D31;
  --blue:#2563EB; --blue2:#3B82F6; --blue3:#60A5FA;
  --t1:#F3F4F6; --t2:#D1D5DB; --t3:#9CA3AF;
  --ok:#22C55E; --warn:#F59E0B; --err:#EF4444;
  --card:rgba(19,29,49,0.55); --border:rgba(148,163,184,0.12);
}
html, body, [class*="css"]{ font-family:'Inter',-apple-system,sans-serif; }

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 620px at 12% -8%, rgba(37,99,235,0.12), transparent 60%),
    radial-gradient(1000px 520px at 100% 0%, rgba(96,165,250,0.07), transparent 55%),
    var(--bg);
  color:var(--t2);
}
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{ visibility:hidden; }
.block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width:1340px; animation:fadeUp .45s ease both; }

@keyframes fadeUp{ from{opacity:0; transform:translateY(12px);} to{opacity:1; transform:none;} }
@keyframes pulse{ 0%,100%{opacity:1;} 50%{opacity:.35;} }

h1,h2,h3,h4{ color:var(--t1)!important; letter-spacing:-0.01em; }
p, label, span, .stMarkdown{ color:var(--t2); }

/* ---------- Top bar (glass) ---------- */
.ms-topbar{
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  background:linear-gradient(135deg, rgba(19,29,49,0.78), rgba(16,24,39,0.62));
  border:1px solid var(--border); border-radius:18px; padding:14px 22px; margin-bottom:18px;
  backdrop-filter:blur(12px); box-shadow:0 10px 34px rgba(0,0,0,0.40); animation:fadeUp .4s ease both;
}
.ms-brand{ display:flex; align-items:center; gap:12px; font-weight:800; color:var(--t1); font-size:1.12rem; }
.ms-logo{ width:36px; height:36px; border-radius:11px; display:grid; place-items:center; font-size:18px;
  background:linear-gradient(135deg,var(--blue),var(--blue3)); box-shadow:0 6px 18px rgba(37,99,235,0.5); }
.ms-actions{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.ms-pill{ display:inline-flex; align-items:center; gap:8px; font-size:0.82rem; font-weight:600;
  padding:8px 13px; border-radius:999px; border:1px solid var(--border);
  background:rgba(255,255,255,0.03); color:var(--t2); transition:all .25s ease; }
.ms-pill:hover{ border-color:rgba(59,130,246,0.45); box-shadow:0 0 16px rgba(37,99,235,0.20); }
.ms-dot{ width:8px; height:8px; border-radius:50%; background:var(--ok); box-shadow:0 0 10px var(--ok); animation:pulse 2s infinite; }
.ms-avatar{ width:36px; height:36px; border-radius:50%; display:grid; place-items:center; font-weight:700; color:#fff;
  background:linear-gradient(135deg,var(--blue2),var(--blue)); box-shadow:0 4px 14px rgba(37,99,235,0.4); }

/* ---------- Section headers ---------- */
.ms-section{ font-size:1.22rem; font-weight:700; color:var(--t1); margin:6px 0 2px; }
.ms-sub{ color:var(--t3); font-size:0.88rem; margin-bottom:12px; }

/* ---------- Metric cards (glass) ---------- */
[data-testid="stMetric"]{
  background:var(--card); border:1px solid var(--border); border-radius:16px; padding:18px 20px;
  backdrop-filter:blur(8px); box-shadow:0 6px 22px rgba(0,0,0,0.28);
  transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stMetric"]:hover{ transform:translateY(-3px); border-color:rgba(59,130,246,0.45);
  box-shadow:0 14px 32px rgba(37,99,235,0.22); }
[data-testid="stMetricLabel"] p{ color:var(--t3); font-weight:500; }
[data-testid="stMetricValue"]{ color:var(--t1); font-weight:700; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0B1322 0%, #0A0F1C 100%); border-right:1px solid var(--border);
}
[data-testid="stSidebar"] [role="radiogroup"]{ gap:5px; display:flex; flex-direction:column; }
[data-testid="stSidebar"] [role="radiogroup"] label{
  display:flex; align-items:center; border-radius:12px; padding:10px 14px; cursor:pointer;
  color:var(--t2); font-weight:500; border:1px solid transparent; transition:all .2s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{
  background:rgba(59,130,246,0.10); border-color:rgba(59,130,246,0.22);
  box-shadow:0 0 18px rgba(37,99,235,0.16); transform:translateX(3px);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:linear-gradient(135deg, rgba(37,99,235,0.32), rgba(59,130,246,0.15));
  border-color:rgba(96,165,250,0.5); color:var(--t1); box-shadow:0 6px 20px rgba(37,99,235,0.30);
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{ display:none; }
.ms-side-brand{ display:flex; align-items:center; gap:10px; font-weight:800; color:var(--t1);
  font-size:1.05rem; margin:4px 4px 14px; }
.ms-side-status{ font-size:0.8rem; color:var(--t3); padding:10px 12px; border-radius:12px;
  border:1px solid var(--border); background:rgba(255,255,255,0.02); margin-top:8px; line-height:1.7; }

/* ---------- Tabs (résiduels) ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:6px; flex-wrap:wrap; }
.stTabs [data-baseweb="tab"]{ border-radius:10px; padding:8px 16px; background:rgba(255,255,255,0.03);
  color:var(--t2); font-weight:500; border:1px solid var(--border); transition:all .2s ease; }
.stTabs [aria-selected="true"]{ background:linear-gradient(135deg,var(--blue),var(--blue2))!important;
  color:#fff!important; border:none!important; }

/* ---------- Buttons ---------- */
.stButton > button{ border-radius:12px; font-weight:600; border:1px solid var(--border);
  background:rgba(255,255,255,0.04); color:var(--t1); transition:all .22s ease; }
.stButton > button:hover{ transform:translateY(-2px); border-color:rgba(59,130,246,0.5);
  box-shadow:0 10px 24px rgba(37,99,235,0.25); }
.stButton > button[kind="primary"]{ background:linear-gradient(135deg,var(--blue),var(--blue2)); border:none; color:#fff; }
.stButton > button[kind="primary"]:hover{ box-shadow:0 12px 30px rgba(37,99,235,0.45); }

/* ---------- Inputs ---------- */
.stTextInput input, .stNumberInput input, [data-baseweb="select"] > div{
  background:var(--bg3)!important; border-radius:10px!important; color:var(--t1)!important;
  border-color:var(--border)!important;
}
.stTextInput input:focus, .stNumberInput input:focus{ border-color:var(--blue2)!important;
  box-shadow:0 0 0 3px rgba(59,130,246,0.20)!important; }

/* ---------- DataFrame / Expander / Progress ---------- */
[data-testid="stDataFrame"]{ border-radius:14px; overflow:hidden; border:1px solid var(--border);
  box-shadow:0 6px 22px rgba(0,0,0,0.25); }
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:14px;
  background:var(--card); overflow:hidden; }
[data-testid="stProgress"] > div > div > div{ background:linear-gradient(90deg,var(--blue),var(--blue3)); }

/* ---------- Note / disclaimer ---------- */
.ms-note{ font-size:0.8rem; color:#FCD34D; background:rgba(245,158,11,0.08);
  border:1px solid rgba(245,158,11,0.30); border-radius:12px; padding:10px 14px; margin:8px 0 16px; }

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:rgba(96,165,250,0.25); border-radius:10px; }
::-webkit-scrollbar-thumb:hover{ background:rgba(96,165,250,0.45); }

/* ---------- Responsive ---------- */
@media (max-width:680px){
  .ms-topbar{ flex-direction:column; align-items:flex-start; gap:10px; }
  .block-container{ padding-left:.7rem; padding-right:.7rem; }
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# --- Données d'en-tête ----------------------------------------------------
with st.spinner("Lecture de l'état du marché…"):
    market_score, market_lbl = cached_market()
    universe_n = cached_universe_size()
ai_ready = get_active_model(MODEL_PATH) is not None
auto_mode = cfg.get("universe.mode", "auto") == "auto"
n_alerts = len(db.recent_alerts(200))

# --- Barre latérale (navigation) ------------------------------------------
PAGES = [
    "🏆  Opportunités",
    "💎  Pépites",
    "🔍  Analyse entreprise",
    "📰  Actualités",
    "🔔  Alertes",
    "💼  Portefeuille",
    "🕓  Historique",
    "📈  Backtest",
]
with st.sidebar:
    st.markdown(
        '<div class="ms-side-brand"><span class="ms-logo">📈</span> Market Sentinel</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.text_input("🔎 Rechercher un ticker", key="global_search", placeholder="ex. NVDA")
    ai_txt = "🟢 Active" if ai_ready else "⚪ Non entraînée"
    st.markdown(
        f'<div class="ms-side-status">IA locale : <b>{ai_txt}</b><br>'
        f"Régime : <b>{market_lbl}</b><br>"
        f"Valeurs scannées : <b>{universe_n}</b><br>"
        f'Mode : <b>{"Auto · marché" if auto_mode else "Watchlist"}</b></div>',
        unsafe_allow_html=True,
    )

# --- Barre supérieure (glass) ---------------------------------------------
st.markdown(
    f"""
    <div class="ms-topbar">
      <div class="ms-brand"><span class="ms-logo">📈</span> Market Sentinel AI
        <span style="color:var(--t3);font-weight:500;font-size:0.8rem;">v{__version__}</span></div>
      <div class="ms-actions">
        <span class="ms-pill"><span class="ms-dot"></span> Marchés connectés</span>
        <span class="ms-pill">🔔 {n_alerts}</span>
        <span class="ms-pill">📊 {market_lbl}</span>
        <span class="ms-avatar">A</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Cartes KPI -----------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Régime de marché", market_lbl, f"{market_score:+.2f}")
k2.metric("IA locale", "Active" if ai_ready else "Non entraînée")
k3.metric("Valeurs scannées", universe_n)
k4.metric("Sélection", "Auto · marché" if auto_mode else "Watchlist")

st.markdown(
    f'<div class="ms-note">⚠️ {DISCLAIMER} Toutes les estimations sont probabilistes.</div>',
    unsafe_allow_html=True,
)


# ==========================================================================
#  PAGE : TOP opportunités
# ==========================================================================
if page == PAGES[0]:
    section("🏆 Opportunités détectées par l'IA",
            "L'IA scanne le marché en direct et sélectionne elle-même les valeurs — aucune liste figée.")
    c1, c2, c3 = st.columns([1, 1.6, 1.4])
    limit = c1.slider("Nombre", 3, 30, 12)
    crit = c2.selectbox(
        "Classer par",
        ["Potentiel IA (mine d'or)", "Mixte (IA + fondamental)", "Score fondamental"],
        index=0,
    )
    if c3.button("🔄 Re-scanner le marché", type="primary"):
        cached_watchlist.clear()

    with st.spinner("Scan du marché et analyse en cours (en parallèle)…"):
        alla = cached_watchlist()

    if not alla:
        st.info("Aucune donnée. Vérifie ta connexion internet.")
    else:
        def _rank_key(a: dict):
            if crit.startswith("Potentiel IA"):
                return a.get("ai_up_proba") if a.get("ai_up_proba") is not None else -1.0
            if crit.startswith("Score"):
                return a["score"]["global_score"]
            return opportunity_index(a)

        ranked = sorted(alla, key=_rank_key, reverse=True)
        top = ranked[:limit]
        st.caption(f"🔎 L'IA a scanné **{len(alla)} valeurs** du marché en direct et "
                   f"retenu les meilleures ci-dessous (tri : {crit.lower()}).")
        rows = [
            {
                "Entreprise": a["name"],
                "Ticker": a["ticker"],
                "Prix": fmt_price(a.get("price"), a.get("currency")),
                "Score /100": a["score"]["global_score"],
                "Potentiel": a["score"]["potential"],
                "Risque": a["score"]["risk"],
                "Hausse %": a["scenario"]["up"],
                "🏆 Mine d'or %": round(a["ai_up_proba"] * 100, 0) if a.get("ai_up_proba") is not None else None,
                "Objectif %": a.get("upside_pct"),
                "Horizon": a["score"]["horizon"],
            }
            for a in top
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        df = pd.DataFrame({"Entreprise": [a["name"] for a in top],
                           "Score": [a["score"]["global_score"] for a in top]})
        fig = go.Figure(go.Bar(
            x=df["Score"], y=df["Entreprise"], orientation="h", text=df["Score"],
            marker=dict(color=df["Score"], colorscale=[[0, BLUE], [1, BLUE3]], showscale=False),
        ))
        fig.update_layout(title="Score global", yaxis={"categoryorder": "total ascending"}, height=460)
        plot(fig)

        for a in alla:
            db.save_analysis(a)


# ==========================================================================
#  PAGE : Pépites
# ==========================================================================
elif page == PAGES[1]:
    section("💎 Pépites", "Sociétés connues, action peu chère, fort potentiel.")
    c1, c2 = st.columns(2)
    max_price = c1.slider("Prix max d'une action", 5, 200,
                          int(cfg.get("screener.max_share_price", 60)))
    min_score = c2.slider("Score minimum", 40, 90, int(cfg.get("screener.min_score", 60)))

    with st.spinner("Recherche des pépites…"):
        alla = cached_watchlist()
    gems = find_gems(alla, max_share_price=float(max_price), min_score=float(min_score))

    if not gems:
        st.info("Aucune pépite avec ces critères. Augmente le prix max ou baisse le score min.")
    else:
        rows = [
            {
                "Entreprise": a["name"],
                "Ticker": a["ticker"],
                "Prix action": fmt_price(a.get("price"), a.get("currency")),
                "Score /100": a["score"]["global_score"],
                "Hausse %": a["scenario"]["up"],
                "Objectif analystes %": a.get("upside_pct"),
                "🏆 Mine d'or % (IA)": round(a["ai_up_proba"] * 100, 0) if a.get("ai_up_proba") is not None else None,
                "Risque": a["score"]["risk"],
            }
            for a in gems
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "« Pépite » = action dont le prix unitaire est faible et le potentiel élevé. "
            "Un prix bas ne veut pas dire « bon marché » au sens valorisation — ce sont des aides à la décision."
        )


# ==========================================================================
#  PAGE : Analyse d'une entreprise
# ==========================================================================
elif page == PAGES[2]:
    section("🔍 Analyse détaillée d'une valeur", "Prix réel, objectifs analystes, technique et IA.")
    default_ticker = (st.session_state.get("global_search") or "NVDA").strip().upper()
    colt, colp = st.columns([2, 1])
    ticker = colt.text_input("Ticker (format Yahoo Finance)", value=default_ticker).strip().upper()
    period = colp.selectbox("Période du graphique", ["6mo", "1y", "2y", "5y"], index=1)

    if st.button("Analyser", key="btn_company", type="primary") and ticker:
        with st.spinner(f"Analyse de {ticker}…"):
            analysis = cached_analysis(ticker)
            history = cached_history(ticker, period)

        score = analysis["score"]
        scenario = analysis["scenario"]
        currency = analysis.get("currency") or ""

        # ---- PRIX RÉEL D'UNE ACTION, bien en évidence ----
        st.markdown(f"### {analysis['name']}  ·  `{analysis['ticker']}`")
        p1, p2, p3, p4 = st.columns(4)
        change = analysis["quote"].get("change_pct")
        p1.metric(
            "💵 Prix d'une action",
            fmt_price(analysis.get("price"), currency),
            f"{change:+.2f}%" if change is not None else None,
        )
        p2.metric("🎯 Objectif analystes", fmt_price(analysis.get("target_price"), currency),
                  f"{analysis['upside_pct']:+.1f}%" if analysis.get("upside_pct") is not None else None)
        p3.metric("Score global", f"{score['global_score']:.0f}/100")
        p4.metric("Recommandation", (analysis.get("recommendation") or "N/A").upper())

        # ---- Plage 52 semaines ----
        low, high, price = analysis.get("week52_low"), analysis.get("week52_high"), analysis.get("price")
        if low and high and price and high > low:
            pos = min(max((price - low) / (high - low), 0.0), 1.0)
            st.progress(pos)
            st.caption(f"Plage 52 semaines — bas {fmt_price(low, currency)} · "
                       f"actuel {fmt_price(price, currency)} · haut {fmt_price(high, currency)}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Potentiel", score["potential"])
        m2.metric("Risque", score["risk"])
        m3.metric("Horizon", score["horizon"])
        m4.metric("Sentiment news", f"{analysis['news_sentiment']:+.2f}")

        # ---- Graphique technique multi-panneaux ----
        if history is not None and not history.empty:
            plot(indicator_chart(history, ticker))

        # ---- Scénarios + IA ----
        cL, cR = st.columns(2)
        with cL:
            st.markdown("#### Scénarios probabilistes")
            pie = go.Figure(go.Pie(
                labels=["Hausse", "Stabilité", "Baisse"],
                values=[scenario["up"], scenario["stable"], scenario["down"]],
                hole=0.5,
                marker=dict(colors=[OK, T3, ERR]),
            ))
            pie.update_layout(height=300)
            plot(pie)
        with cR:
            st.markdown("#### IA locale")
            ai = analysis.get("ai_up_proba")
            if ai is not None:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=ai * 100,
                    number={"suffix": "%", "font": {"color": T1}},
                    title={"text": "Potentiel mine d'or (IA)"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": BLUE2},
                           "bgcolor": "rgba(255,255,255,0.03)",
                           "steps": [{"range": [0, 45], "color": "rgba(239,68,68,0.18)"},
                                     {"range": [45, 55], "color": "rgba(245,158,11,0.18)"},
                                     {"range": [55, 100], "color": "rgba(34,197,94,0.18)"}]},
                ))
                gauge.update_layout(height=300)
                plot(gauge)
                st.caption("🏆 « Mine d'or » = ressemblance aux **gagnantes durables** du "
                           "passé (top 3/6/12 mois). Validation honnête : **lift ≈ 1,56×** "
                           "(~1,5× plus de gagnantes dans son top 10% que le hasard, AUC 0,565). "
                           "Avantage réel mais **modeste** — biais du survivant assumé, "
                           "signal de tri, **jamais une garantie de gain**.")
            elif ai_ready:
                st.info("IA active ✅, mais **données insuffisantes pour estimer cette "
                        "valeur précise** (historique trop court ou indisponible — "
                        "fréquent pour les sociétés récemment cotées).")
            else:
                st.info("IA locale non entraînée. Double-clique sur **Entrainer_IA.bat** "
                        "pour l'activer, puis recharge la page.")

        st.markdown(f"**Justification :** {analysis['justification']}")
        st.caption("Rationnel IA/scénario : " + " · ".join(scenario["rationale"]))

        with st.expander("Détail des sous-scores fondamentaux/techniques"):
            comps = {k: v for k, v in score["components"].items() if v is not None}
            st.bar_chart(pd.Series(comps, name="Sous-score /100"))


# ==========================================================================
#  PAGE : Actualités
# ==========================================================================
elif page == PAGES[3]:
    section("📰 Actualités & sentiment", "Titres récents analysés et notés automatiquement.")
    news_ticker = st.text_input("Ticker", value="AAPL", key="news_ticker").strip().upper()
    if st.button("Charger les actualités", key="btn_news") and news_ticker:
        with st.spinner("Récupération des actualités…"):
            items = cached_news(news_ticker)
        if not items:
            st.info("Aucune actualité disponible pour ce ticker.")
        for item in items:
            sent = item["sentiment"]
            emoji = "🟢" if sent.score > 0.15 else "🔴" if sent.score < -0.15 else "⚪"
            st.markdown(
                f"{emoji} **{item['title']}**  \n"
                f"*{item['publisher']}* — {sent.label} "
                f"(score {sent.score:+.2f}, confiance {sent.confidence:.0%})"
            )
            if item["link"]:
                st.caption(item["link"])
            st.divider()


# ==========================================================================
#  PAGE : Alertes
# ==========================================================================
elif page == PAGES[4]:
    section("🔔 Alertes récentes", "Opportunités et mouvements détectés par le service.")
    alerts = db.recent_alerts(100)
    if alerts:
        st.dataframe(
            pd.DataFrame(alerts)[["created_at", "ticker", "title", "message", "confidence"]],
            width="stretch", hide_index=True,
        )
    else:
        st.info("Aucune alerte enregistrée pour le moment.")


# ==========================================================================
#  PAGE : Portefeuille virtuel
# ==========================================================================
elif page == PAGES[5]:
    section("💼 Portefeuille virtuel", "Suis tes positions et leur valeur estimée.")
    with st.form("add_position", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        pf_ticker = c1.text_input("Ticker").strip().upper()
        pf_shares = c2.number_input("Quantité", min_value=0.0, step=1.0)
        pf_price = c3.number_input("Prix d'achat moyen", min_value=0.0, step=1.0)
        if st.form_submit_button("Ajouter / mettre à jour") and pf_ticker:
            db.upsert_position(pf_ticker, pf_shares, pf_price)
            st.success(f"Position {pf_ticker} enregistrée.")

    positions = db.get_portfolio()
    if positions:
        rows, total_cost, total_value = [], 0.0, 0.0
        for pos in positions:
            try:
                last = cached_analysis(pos["ticker"]).get("price")
            except Exception:  # noqa: BLE001
                last = None
            cost = pos["shares"] * pos["avg_price"]
            value = pos["shares"] * last if last else None
            total_cost += cost
            total_value += value or 0
            rows.append({
                "Ticker": pos["ticker"],
                "Quantité": pos["shares"],
                "Prix moyen": pos["avg_price"],
                "Cours actuel": round(last, 2) if last else "N/A",
                "Valeur": round(value, 2) if value else "N/A",
                "P&L": round(value - cost, 2) if value else "N/A",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.metric("Valeur totale (estimée)", f"{total_value:,.2f}",
                  f"{total_value - total_cost:+,.2f} vs coût" if total_cost else None)
        del_ticker = st.selectbox("Supprimer une position", [p["ticker"] for p in positions])
        if st.button("Supprimer"):
            db.remove_position(del_ticker)
            st.rerun()
    else:
        st.info("Portefeuille vide. Ajoute une position ci-dessus.")


# ==========================================================================
#  PAGE : Historique
# ==========================================================================
elif page == PAGES[6]:
    section("🕓 Historique des analyses", "Toutes les analyses enregistrées.")
    history_rows = db.latest_analyses(300)
    if history_rows:
        df = pd.DataFrame(history_rows)[
            ["created_at", "ticker", "name", "global_score", "risk", "potential", "horizon"]
        ]
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("Aucune analyse enregistrée. Lance une analyse depuis la page Opportunités.")


# ==========================================================================
#  PAGE : Backtest
# ==========================================================================
elif page == PAGES[7]:
    section("📈 Backtest de la stratégie",
            "Validation honnête sur l'historique — momentum + tendance, point-in-time.")
    st.markdown(
        '<div class="ms-note">⚠️ On backteste la stratégie <b>prix/tendance</b> '
        "(reconstituable dans le passé), <b>PAS</b> le score fondamental (yfinance n'a pas "
        "de fondamentaux historiques → ce serait une fuite de données). Univers = watchlist, "
        "biais du survivant assumé. Le passé ne préjuge pas du futur.</div>",
        unsafe_allow_html=True,
    )
    cc1, cc2, cc3 = st.columns([1, 1, 1.4])
    bt_n = cc1.slider("Taille du TOP", 3, 20, 10)
    bt_period = cc2.selectbox("Période", ["2y", "5y", "10y", "max"], index=1)
    launch = cc3.button("🚀 Lancer le backtest", type="primary")

    if launch:
        with st.spinner("Backtest en cours (téléchargement de l'historique)…"):
            try:
                res = cached_backtest(bt_n, bt_period)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Backtest impossible : {exc}")
                res = None
        if res:
            m, bm = res["metrics"], res["benchmark_metrics"]
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Rendement annualisé", f"{m.get('cagr', 0) * 100:.1f}%",
                      f"{(m.get('cagr', 0) - bm.get('cagr', 0)) * 100:+.1f} pts vs S&P 500")
            x2.metric("Ratio de Sharpe", f"{m.get('sharpe', 0):.2f}")
            x3.metric("Perte max (drawdown)", f"{m.get('max_drawdown', 0) * 100:.1f}%")
            x4.metric("Mois battant le S&P 500", f"{m.get('beat_benchmark', 0) * 100:.0f}%")

            eq, beq = res["equity"], res["benchmark"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Stratégie",
                                     line=dict(color=BLUE3, width=2.4, shape="spline")))
            fig.add_trace(go.Scatter(x=beq.index, y=beq.values, name="S&P 500",
                                     line=dict(color=T3, width=1.6, shape="spline")))
            fig.update_layout(title="Courbe d'équité (base 1.0)", height=430)
            plot(fig)
            st.caption(
                f"Stratégie : CAGR {m.get('cagr', 0) * 100:.1f}% · Sharpe {m.get('sharpe', 0):.2f} · "
                f"hit-rate {m.get('hit_rate', 0) * 100:.0f}% sur {m.get('months', 0)} mois "
                f"({res['n_tickers']} valeurs). "
                f"S&P 500 : CAGR {bm.get('cagr', 0) * 100:.1f}% · Sharpe {bm.get('sharpe', 0):.2f}."
            )
    else:
        st.info("Choisis les paramètres puis clique sur « Lancer le backtest ».")
