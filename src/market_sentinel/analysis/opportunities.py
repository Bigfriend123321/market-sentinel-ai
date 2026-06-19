"""Orchestration de l'analyse complète d'une valeur et construction du TOP N.

Pour chaque ticker on enchaîne : fondamentaux -> technique -> score ->
cotation -> actualités/sentiment -> IA locale -> régime de marché ->
scénario probabiliste -> justification.

`analyze_watchlist` exécute ces analyses **en parallèle** (I/O réseau), ce
qui permet de suivre une watchlist beaucoup plus large sans ralentir.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional

from ..ai.ml_model import get_active_model
from ..ai.predictor import predict_scenarios
from ..data.fundamentals import get_fundamentals
from ..data.market_data import get_history, get_quote
from ..logging_setup import get_logger
from ..news.fetcher import aggregate_sentiment, fetch_news
from .market_regime import market_trend_score
from .scoring import compute_score
from .technical import compute_technical
from .universe import get_universe

log = get_logger(__name__)


@dataclass
class Analysis:
    """Synthèse complète d'une valeur à un instant donné."""

    ticker: str
    name: str
    sector: str
    quote: dict
    score: dict
    technical: dict
    news_sentiment: float
    scenario: dict
    justification: str
    timestamp: str
    # --- Prix & objectifs (page société + screener) ---
    price: Optional[float] = None
    currency: Optional[str] = None
    target_price: Optional[float] = None
    upside_pct: Optional[float] = None       # potentiel vs objectif analystes (%)
    recommendation: Optional[str] = None
    week52_low: Optional[float] = None
    week52_high: Optional[float] = None
    ai_up_proba: Optional[float] = None      # probabilité de hausse de l'IA locale


def analyze_ticker(ticker: str, cfg, market_trend: Optional[float] = None) -> Analysis:
    """Analyse de bout en bout d'une seule valeur.

    `market_trend` peut être fourni par l'appelant (calculé une seule fois par
    cycle dans `analyze_watchlist`) pour éviter de recharger l'indice de
    référence à chaque valeur.
    """
    log.info("Analyse de %s", ticker)

    fundamentals = get_fundamentals(ticker)
    history = get_history(
        ticker,
        cfg.get("data.default_period", "1y"),
        cfg.get("data.default_interval", "1d"),
    )
    technical = compute_technical(history)
    score = compute_score(fundamentals, technical, cfg.get("scoring.weights", {}))
    quote = get_quote(ticker)

    backend = "finbert" if cfg.get("news.provider") == "finbert" else "lexicon"
    news = fetch_news(ticker, cfg.get("news.max_articles", 20), backend=backend)
    sentiment = aggregate_sentiment(news)

    # IA locale (si un modèle est entraîné et activé)
    ai_up_proba = None
    if cfg.get("ai.use_ml_model", True):
        model = get_active_model(
            str(cfg.resolve_path(cfg.get("ai.model_path", "data/predictor.joblib")))
        )
        if model is not None:
            try:
                ai_up_proba = model.predict_up_proba(history)
            except Exception as exc:  # noqa: BLE001
                log.debug("IA indisponible pour %s : %s", ticker, exc)

    # Régime de marché global (fourni par le cycle, sinon calculé à la volée).
    market = (
        market_trend
        if market_trend is not None
        else market_trend_score(cfg.get("ai.market_index", "^GSPC"))
    )

    scenario = predict_scenarios(
        score.global_score, technical.score, sentiment, technical.volatility,
        ai_up_proba=ai_up_proba, market_trend=market,
    )

    # Prix réel d'une action (priorité : cotation, repli sur fondamentaux)
    price = quote.price if quote.price is not None else fundamentals.current_price
    target = fundamentals.target_mean_price
    upside = None
    if price and target:
        upside = round((target / price - 1.0) * 100, 1)

    justification = _build_justification(fundamentals, technical, sentiment, scenario)

    return Analysis(
        ticker=ticker,
        name=fundamentals.name or ticker,
        sector=fundamentals.sector or "N/A",
        quote=asdict(quote),
        score=asdict(score),
        technical=asdict(technical),
        news_sentiment=sentiment,
        scenario=asdict(scenario),
        justification=justification,
        timestamp=datetime.now(timezone.utc).isoformat(),
        price=price,
        currency=fundamentals.currency or quote.currency,
        target_price=target,
        upside_pct=upside,
        recommendation=fundamentals.recommendation,
        week52_low=fundamentals.fifty_two_week_low,
        week52_high=fundamentals.fifty_two_week_high,
        ai_up_proba=ai_up_proba,
    )


def analyze_watchlist(cfg, max_workers: Optional[int] = None) -> List[Analysis]:
    """Analyse tout l'univers (dynamique en mode 'auto') en parallèle."""
    if max_workers is None:
        max_workers = int(cfg.get("data.max_workers", 8))
    tickers = get_universe(cfg)
    log.info("Univers d'analyse : %d valeurs", len(tickers))
    # Régime de marché calculé UNE seule fois pour tout le cycle.
    market = market_trend_score(cfg.get("ai.market_index", "^GSPC"))
    results: List[Analysis] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_ticker, t, cfg, market): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                log.exception("Échec de l'analyse de %s : %s", ticker, exc)

    results.sort(key=lambda a: a.score["global_score"], reverse=True)
    return results


def build_top_opportunities(cfg, limit: int = 10) -> List[Analysis]:
    """Analyse toute la watchlist et renvoie le TOP N trié par score global."""
    return analyze_watchlist(cfg)[:limit]


def _build_justification(fundamentals, technical, sentiment, scenario) -> str:
    parts: List[str] = []
    if fundamentals.revenue_growth is not None:
        parts.append(f"croissance des revenus {fundamentals.revenue_growth:+.0%}")
    if fundamentals.profit_margin is not None:
        parts.append(f"marge {fundamentals.profit_margin:.0%}")
    parts.append(f"tendance {technical.trend}")
    if sentiment > 0.10:
        parts.append("actualités positives")
    elif sentiment < -0.10:
        parts.append("actualités négatives")
    parts.append(f"hausse probable {scenario.up:.0f}%")
    return "; ".join(parts).capitalize() + "."
