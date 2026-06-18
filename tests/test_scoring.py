"""Tests unitaires (sans réseau) du moteur de score et du moteur prédictif.

Lancer :
    pip install pytest
    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_sentinel.ai.predictor import predict_scenarios  # noqa: E402
from market_sentinel.analysis.scoring import compute_score  # noqa: E402
from market_sentinel.analysis.technical import TechnicalSignals  # noqa: E402
from market_sentinel.data.fundamentals import Fundamentals  # noqa: E402
from market_sentinel.news.sentiment import analyze_text_lexicon  # noqa: E402

WEIGHTS = {
    "revenue_growth": 0.15, "profit_margin": 0.15, "pe_ratio": 0.12,
    "ps_ratio": 0.08, "debt_to_equity": 0.12, "free_cash_flow": 0.13,
    "eps": 0.10, "market_cap": 0.05, "technical": 0.10,
}


def _strong_company() -> Fundamentals:
    return Fundamentals(
        ticker="TEST", name="Test Corp", sector="Tech", market_cap=300e9,
        revenue=100e9, revenue_growth=0.30, profit_margin=0.25, pe_ratio=18,
        ps_ratio=4, debt_to_equity=25, free_cash_flow=20e9, eps=6.0,
    )


def _weak_company() -> Fundamentals:
    return Fundamentals(
        ticker="WEAK", name="Weak Inc", sector="Misc", market_cap=1e9,
        revenue=5e8, revenue_growth=-0.10, profit_margin=-0.05, pe_ratio=120,
        ps_ratio=25, debt_to_equity=400, free_cash_flow=-1e8, eps=-0.5,
    )


def test_score_in_range():
    tech = TechnicalSignals(trend="haussier", volatility=0.25, score=70)
    result = compute_score(_strong_company(), tech, WEIGHTS)
    assert 0 <= result.global_score <= 100


def test_strong_beats_weak():
    tech = TechnicalSignals(trend="haussier", volatility=0.25, score=70)
    strong = compute_score(_strong_company(), tech, WEIGHTS)
    weak = compute_score(_weak_company(), tech, WEIGHTS)
    assert strong.global_score > weak.global_score
    assert strong.risk == "Faible"
    assert weak.risk in {"Modéré", "Élevé"}


def test_missing_data_is_renormalised():
    # Une entreprise sans aucune donnée fondamentale ne doit pas crasher.
    empty = Fundamentals(ticker="NA")
    tech = TechnicalSignals(score=50)
    result = compute_score(empty, tech, WEIGHTS)
    assert 0 <= result.global_score <= 100
    assert result.notes  # des données manquantes sont signalées


def test_scenarios_sum_to_100():
    scenario = predict_scenarios(85, 75, 0.4, 0.25)
    total = scenario.up + scenario.stable + scenario.down
    assert abs(total - 100.0) < 0.5
    assert 0.30 <= scenario.confidence <= 0.95


def test_high_score_favours_upside():
    bullish = predict_scenarios(90, 80, 0.5, 0.20)
    bearish = predict_scenarios(35, 30, -0.5, 0.20)
    assert bullish.up > bullish.down
    assert bearish.down > bearish.up


def test_sentiment_lexicon():
    assert analyze_text_lexicon("Company beats earnings, shares surge").score > 0
    assert analyze_text_lexicon("Company misses estimates, stock plunges").score < 0
    assert analyze_text_lexicon("Company holds annual meeting").label == "Neutre"
