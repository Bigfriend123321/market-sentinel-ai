"""Récupération des données fondamentales d'une entreprise via yfinance.

Les champs renvoyés par Yahoo Finance peuvent être absents selon la valeur :
toutes les propriétés sont donc Optionnelles et gérées en aval (le moteur
de score ignore les composantes manquantes et renormalise les poids).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yfinance as yf

from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Fundamentals:
    """Indicateurs fondamentaux normalisés."""

    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: Optional[str] = None
    market_cap: Optional[float] = None          # capitalisation boursière
    revenue: Optional[float] = None             # chiffre d'affaires (TTM)
    revenue_growth: Optional[float] = None      # croissance des revenus (fraction)
    profit_margin: Optional[float] = None       # marge bénéficiaire (fraction)
    pe_ratio: Optional[float] = None            # ratio cours/bénéfice
    forward_pe: Optional[float] = None          # P/E prévisionnel
    peg_ratio: Optional[float] = None           # PEG (P/E rapporté à la croissance)
    ps_ratio: Optional[float] = None            # ratio cours/ventes
    debt_to_equity: Optional[float] = None      # dette/capitaux propres (en %)
    free_cash_flow: Optional[float] = None      # flux de trésorerie disponible
    eps: Optional[float] = None                 # bénéfice par action
    # --- Prix & objectifs (pour la page société) ---
    current_price: Optional[float] = None       # PRIX RÉEL d'une action
    target_mean_price: Optional[float] = None   # objectif moyen des analystes
    target_high_price: Optional[float] = None
    target_low_price: Optional[float] = None
    num_analysts: Optional[int] = None
    recommendation: Optional[str] = None        # buy / hold / sell ...
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    beta: Optional[float] = None                # sensibilité au marché
    dividend_yield: Optional[float] = None      # rendement du dividende (fraction)


def get_fundamentals(ticker: str) -> Fundamentals:
    """Construit l'objet Fundamentals depuis `Ticker.info`."""
    info: dict = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("Fondamentaux indisponibles pour %s : %s", ticker, exc)

    g = info.get
    return Fundamentals(
        ticker=ticker,
        name=g("longName") or g("shortName"),
        sector=g("sector"),
        industry=g("industry"),
        currency=g("currency"),
        market_cap=g("marketCap"),
        revenue=g("totalRevenue"),
        revenue_growth=g("revenueGrowth"),
        profit_margin=g("profitMargins"),
        pe_ratio=g("trailingPE") or g("forwardPE"),
        forward_pe=g("forwardPE"),
        peg_ratio=g("trailingPegRatio") or g("pegRatio"),
        ps_ratio=g("priceToSalesTrailing12Months"),
        debt_to_equity=g("debtToEquity"),
        free_cash_flow=g("freeCashflow"),
        eps=g("trailingEps"),
        current_price=g("currentPrice") or g("regularMarketPrice"),
        target_mean_price=g("targetMeanPrice"),
        target_high_price=g("targetHighPrice"),
        target_low_price=g("targetLowPrice"),
        num_analysts=g("numberOfAnalystOpinions"),
        recommendation=g("recommendationKey"),
        fifty_two_week_high=g("fiftyTwoWeekHigh"),
        fifty_two_week_low=g("fiftyTwoWeekLow"),
        beta=g("beta"),
        dividend_yield=g("dividendYield"),
    )
