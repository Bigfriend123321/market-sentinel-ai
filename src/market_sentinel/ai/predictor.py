"""Moteur prédictif probabiliste — scénarios hausse / stabilité / baisse.

AVERTISSEMENT : ce moteur produit des PROBABILITÉS, jamais des certitudes.
Il combine de façon transparente plusieurs signaux normalisés :
    - le score fondamental global,
    - le sous-score technique,
    - le sentiment agrégé des actualités,
    - la probabilité de hausse de l'IA locale (si un modèle est entraîné),
    - le régime de marché global (S&P 500),
puis applique un softmax avec une "température" pilotée par la volatilité :
plus la volatilité est élevée, plus la distribution est aplatie (incertitude
accrue) et plus la confiance affichée est réduite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Scenario:
    """Distribution probabiliste des évolutions possibles (en %)."""

    up: float
    stable: float
    down: float
    confidence: float
    rationale: List[str]


def _softmax(values: List[float]) -> List[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def predict_scenarios(
    global_score: float,
    technical_score: float,
    news_sentiment: float,
    volatility: Optional[float],
    ai_up_proba: Optional[float] = None,
    market_trend: float = 0.0,
) -> Scenario:
    """Estime les probabilités de hausse / stabilité / baisse.

    `ai_up_proba` : sortie de l'IA locale dans [0, 1] (None si pas de modèle).
    `market_trend` : orientation du marché global dans [-1, 1].
    """
    # Normalisation de chaque signal autour de 0.
    fundamental = (global_score - 60) / 40.0   # ~[-1.5, 1.0]
    technical = (technical_score - 50) / 50.0  # [-1, 1]
    sentiment = max(-1.0, min(1.0, news_sentiment))
    market = max(-1.0, min(1.0, market_trend))

    # Poids des signaux. L'IA technique n'ayant PAS d'avantage prouvé en
    # validation honnête (AUC ≈ 0,50), elle ne reçoit qu'un poids modeste :
    # les fondamentaux et le sentiment pilotent l'essentiel de la décision.
    if ai_up_proba is not None:
        ai_signal = (ai_up_proba - 0.5) * 2.0  # [0,1] -> [-1,1]
        composite = (
            0.42 * fundamental
            + 0.22 * technical
            + 0.14 * sentiment
            + 0.12 * ai_signal
            + 0.10 * market
        )
    else:
        composite = (
            0.42 * fundamental
            + 0.25 * technical
            + 0.18 * sentiment
            + 0.15 * market
        )

    # Logits bruts pour chaque issue.
    up_logit = 2.0 * composite
    down_logit = -2.0 * composite
    stable_logit = 0.6 - 1.5 * abs(composite)

    # Température : la volatilité augmente l'incertitude (aplatit la distribution).
    vol = volatility if volatility is not None else 0.30
    temperature = 1.0 + min(vol, 1.0)
    up, stable, down = _softmax(
        [up_logit / temperature, stable_logit / temperature, down_logit / temperature]
    )

    # La confiance grimpe avec la force du signal et la présence de l'IA.
    base_conf = 0.5 + 0.4 * abs(composite) - 0.15 * min(vol, 1.0)
    if ai_up_proba is not None:
        base_conf += 0.05
    confidence = round(max(0.30, min(0.95, base_conf)), 2)

    rationale = [
        f"Score fondamental : {global_score:.0f}/100",
        f"Signal technique : {technical_score:.0f}/100",
        f"Sentiment des actualités : {news_sentiment:+.2f}",
    ]
    if ai_up_proba is not None:
        rationale.append(f"IA locale — surperformance probable : {ai_up_proba:.0%}")
    rationale.append(f"Régime de marché : {market:+.2f}")
    rationale.append(f"Volatilité annualisée estimée : {vol:.0%}")

    return Scenario(
        up=round(up * 100, 1),
        stable=round(stable * 100, 1),
        down=round(down * 100, 1),
        confidence=confidence,
        rationale=rationale,
    )
