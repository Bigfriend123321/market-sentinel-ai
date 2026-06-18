"""Moteur de score global /100 — TRANSPARENT et explicable.

Principe : chaque critère fondamental est converti en sous-score [0, 100]
via des seuils lisibles, puis combiné par moyenne pondérée. Les composantes
manquantes sont ignorées et les poids restants renormalisés (pas de pénalité
arbitraire pour une donnée absente).

Ce moteur n'est volontairement PAS une boîte noire : chaque sous-score est
conservé dans `ScoreBreakdown.components` afin de pouvoir justifier la note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..data.fundamentals import Fundamentals
from .technical import TechnicalSignals


@dataclass
class ScoreBreakdown:
    """Résultat détaillé du scoring d'une valeur."""

    ticker: str
    global_score: float
    risk: str             # "Faible" | "Modéré" | "Élevé"
    potential: str        # "Faible" | "Moyen" | "Élevé" | "Très élevé"
    horizon: str
    components: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# --- Conversion de chaque critère en sous-score [0, 100] -----------------

def _score_revenue_growth(g: Optional[float]) -> Optional[float]:
    if g is None:
        return None
    # fraction : 0 % -> 40, +20 % -> 90, +40 % et plus -> 100
    return _clamp(40 + g * 250)


def _score_profit_margin(m: Optional[float]) -> Optional[float]:
    if m is None:
        return None
    return _clamp(30 + m * 300)  # 0 % -> 30, +20 % -> 90


def _score_pe(pe: Optional[float]) -> Optional[float]:
    if pe is None or pe <= 0:
        return None
    if pe <= 10:
        return 90.0
    if pe <= 15:
        return 80.0
    if pe <= 25:
        return 65.0
    if pe <= 40:
        return 45.0
    return 30.0


def _score_ps(ps: Optional[float]) -> Optional[float]:
    if ps is None or ps <= 0:
        return None
    if ps <= 2:
        return 90.0
    if ps <= 5:
        return 75.0
    if ps <= 10:
        return 55.0
    if ps <= 20:
        return 40.0
    return 25.0


def _score_debt(de: Optional[float]) -> Optional[float]:
    # yfinance exprime debtToEquity en pourcentage (ex. 50 = 0,5x).
    if de is None:
        return None
    if de <= 30:
        return 90.0
    if de <= 60:
        return 75.0
    if de <= 100:
        return 60.0
    if de <= 200:
        return 40.0
    return 25.0


def _score_fcf(fcf: Optional[float]) -> Optional[float]:
    if fcf is None:
        return None
    return 75.0 if fcf > 0 else 30.0


def _score_eps(eps: Optional[float]) -> Optional[float]:
    if eps is None:
        return None
    return 70.0 if eps > 0 else 30.0


def _score_market_cap(mc: Optional[float]) -> Optional[float]:
    # Sous-score de stabilité : une grande capitalisation = profil plus stable.
    if mc is None:
        return None
    if mc >= 200e9:
        return 85.0
    if mc >= 50e9:
        return 75.0
    if mc >= 10e9:
        return 65.0
    if mc >= 2e9:
        return 55.0
    return 45.0


# --- Agrégation ----------------------------------------------------------

def compute_score(
    f: Fundamentals, tech: TechnicalSignals, weights: dict
) -> ScoreBreakdown:
    """Calcule le score global et les niveaux de risque/potentiel/horizon."""
    components = {
        "revenue_growth": _score_revenue_growth(f.revenue_growth),
        "profit_margin": _score_profit_margin(f.profit_margin),
        "pe_ratio": _score_pe(f.pe_ratio),
        "ps_ratio": _score_ps(f.ps_ratio),
        "debt_to_equity": _score_debt(f.debt_to_equity),
        "free_cash_flow": _score_fcf(f.free_cash_flow),
        "eps": _score_eps(f.eps),
        "market_cap": _score_market_cap(f.market_cap),
        "technical": tech.score,
    }

    numerator = denominator = 0.0
    notes: list[str] = []
    for key, sub in components.items():
        weight = float(weights.get(key, 0) or 0)
        if sub is None:
            notes.append(f"Donnée manquante : {key}")
            continue
        numerator += weight * sub
        denominator += weight

    global_score = round(numerator / denominator, 1) if denominator > 0 else 0.0

    risk = _risk_level(f, tech)
    potential = _potential_level(global_score, tech)
    horizon = _horizon(global_score, risk)

    return ScoreBreakdown(
        ticker=f.ticker,
        global_score=global_score,
        risk=risk,
        potential=potential,
        horizon=horizon,
        components=components,
        notes=notes,
    )


def _risk_level(f: Fundamentals, tech: TechnicalSignals) -> str:
    points = 0
    if tech.volatility is not None:
        if tech.volatility > 0.60:
            points += 2
        elif tech.volatility > 0.35:
            points += 1
    if f.debt_to_equity is not None and f.debt_to_equity > 150:
        points += 1
    if f.pe_ratio is not None and f.pe_ratio > 40:
        points += 1
    if f.eps is not None and f.eps < 0:
        points += 1

    if points >= 3:
        return "Élevé"
    if points >= 1:
        return "Modéré"
    return "Faible"


def _potential_level(score: float, tech: TechnicalSignals) -> str:
    if score >= 80 and tech.trend == "haussier":
        return "Très élevé"
    if score >= 70:
        return "Élevé"
    if score >= 55:
        return "Moyen"
    return "Faible"


def _horizon(score: float, risk: str) -> str:
    if score >= 80 and risk != "Élevé":
        return "3 à 5 ans"
    if score >= 65:
        return "1 à 3 ans"
    return "6 à 12 mois (surveillance)"
